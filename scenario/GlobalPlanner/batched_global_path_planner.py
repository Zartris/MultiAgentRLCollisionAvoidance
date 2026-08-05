# benchmark matrix multiplication
from os import environ
from typing import List

environ["OMP_NUM_THREADS"] = "7"

import time

import cv2
import numba
import numpy as np
import os

import torch
from numba import njit, prange
from colorama import Fore, Style


# To jit it, we will move it out of the class


@njit
def get_neighbor(s, move_set):
    """
    find neighbors of state s that not in obstacles.
    :param s: state
    :return: neighbors
    """

    n = [(s[:] + u) for u in move_set]
    return n


@njit(parallel=False)
def is_collision(
    s_start: numba.int64[:, :, :],
    s_end: numba.int64[:, :, :],
    grid_map: numba.uint8[:, :],
):
    """
    check if the line segment (s_start, s_end) is collision.
    :param s_start: start node
    :param s_end: end node
    :return: True: is collision / False: not collision
    """
    is_wall_collision = np.zeros(s_start.shape[0])
    # index: int = 0
    for i in prange(s_start.shape[0]):
        s = s_start[i]
        e = s_end[i]
        if grid_map[s[1], s[0]] != 0 or grid_map[e[1], e[0]] != 0:
            is_wall_collision[i] = 1
    return is_wall_collision


@njit()
def snap_to_free(points, grid_map):
    """Move any point sitting in an occupied cell to the nearest free cell.

    Grid discretization + obstacle inflation can map a position that is
    geometrically clear of an obstacle into an occupied cell. Rather than failing
    planning for the whole batch, snap that start/goal to the closest free cell so
    A* still has a reachable endpoint within ~one cell of the true position.
    """
    h, w = grid_map.shape
    out = points.copy()
    for i in range(points.shape[0]):
        # Clamp into the grid before indexing: a start/goal can map slightly out of
        # bounds (an agent at the world edge), and an unchecked grid_map[y, x] would
        # be an out-of-bounds read. The ring search below then snaps it to free.
        x = min(max(int(points[i, 0]), 0), w - 1)
        y = min(max(int(points[i, 1]), 0), h - 1)
        out[i, 0] = x
        out[i, 1] = y
        if grid_map[y, x] == 0:
            continue
        found = False
        for r in range(1, h + w):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if -r < dx < r and -r < dy < r:
                        continue  # only the ring at Chebyshev distance r
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < w and 0 <= ny < h and grid_map[ny, nx] == 0:
                        out[i, 0] = nx
                        out[i, 1] = ny
                        found = True
                        break
                if found:
                    break
            if found:
                break
    return out


@njit()
# @profile
def cost(s_start, s_goal, grid_map, heuristic_type, same_move=False):
    """
    Calculate Cost for this motion
    :param s_start: starting node
    :param s_goal: end node
    :return:  Cost for this motion
    :note: Cost function could be more complicate!
    """
    # c = np.hypot(s_goal[:, 0] - s_start[:, 0], s_goal[:, 1] - s_start[:, 1])
    c = heuristic(s_start, s_goal, heuristic_type)
    col = is_collision(s_start, s_goal, grid_map)
    colliding = np.where(col)
    if colliding[0].size > 0:
        c[colliding] = np.inf

    return c


def extract_path(PARENT, s_goal, s_start, grid_map):
    """
    Extract the path based on the PARENT set.
    :return: The planning path
    """
    max_path_length = grid_map.shape[0] * grid_map.shape[1]

    paths = [[s] for s in s_goal[:]]
    s = s_goal.copy()
    dones = np.zeros(s.shape[0], dtype=np.uint8)
    batch_index = np.arange(s.shape[0])
    sp_flat = np.zeros((1, 1), dtype=np.int64)
    path_length = 1
    while True:
        s_flat = get_flat_index(s, grid_map)

        # Have to do forloop as we have different path length
        for i in range(s_flat.shape[0]):
            if dones[i] == 1:
                continue
            sp_flat[0] = PARENT[i, s_flat[i]]
            sp = get_map_index(sp_flat, grid_map).astype(np.int64)
            paths[i].append(sp[0])
            if (sp[0] == s_start[i]).all():
                dones[i] = 1
            s[i] = sp[0]
        if dones.all():
            break
        path_length += 1
        # Make sure that the while loops ends
        if path_length > max_path_length:
            print("EXTRACT PATH LOOP FOUND!")
            return None

    results = []
    for i in range(len(paths)):
        path = np.zeros((len(paths[i]), 2))
        for index, coord in enumerate(paths[i]):
            path[-(index + 1)] = coord
        results.append(path)
    return results


@njit()
def nb_reverse_array(l):
    new_l = np.zeros_like(l)
    for i in range(len(l)):
        new_l[i] = l[-i - 1]
    return new_l


# @profile
def check_neighbors_and_update_cost(
    s_flat, move_set, g, PARENT, CLOSED, inflated_map, s_goal, heuristic_type
):
    tol = 1e-6
    start = time.perf_counter()
    batch_index = np.arange(s_flat.shape[0])
    estimated_costs = []
    s = get_map_index(s_flat, inflated_map)
    for s_n in get_neighbor(s, move_set):
        s_n_flat = get_flat_index(s_n, inflated_map)
        current_cost = g[batch_index, s_flat]
        move_cost = cost(s, s_n, inflated_map, heuristic_type)
        new_cost = current_cost + move_cost

        # Check if we have been here before and if the new cost is lower
        old_cost = g[batch_index, s_n_flat]  # inf if not visited
        update_mask = np.argwhere(new_cost + tol < old_cost).squeeze()
        if update_mask.size > 0:
            if CLOSED[update_mask, s_n_flat[update_mask]].any():
                debug = 0
                print("found a node that is already visited")
            g[update_mask, s_n_flat[update_mask]] = new_cost[update_mask]
            PARENT[update_mask, s_n_flat[update_mask]] = s_flat[update_mask]
            estimated_cost = (
                f_value(s_n, g, inflated_map, s_goal, heuristic_type)[update_mask],
                s_n_flat[update_mask],
                update_mask,
            )
            estimated_costs.append(estimated_cost)
    # print(f"\tcheck_neighbors_and_update_cost time: {time.perf_counter() - start:.6f}s")
    return estimated_costs, PARENT, g


@njit()
def _heap_sift_up(hp_node, hp_f, hp_pos, i):
    node = hp_node[i]
    f = hp_f[i]
    while i > 0:
        parent = (i - 1) // 2
        if hp_f[parent] <= f:
            break
        hp_node[i] = hp_node[parent]
        hp_f[i] = hp_f[parent]
        hp_pos[hp_node[i]] = i
        i = parent
    hp_node[i] = node
    hp_f[i] = f
    hp_pos[node] = i


@njit()
def _heap_sift_down(hp_node, hp_f, hp_pos, i, size):
    node = hp_node[i]
    f = hp_f[i]
    while True:
        smallest = i
        sf = f
        left = 2 * i + 1
        right = 2 * i + 2
        if left < size and hp_f[left] < sf:
            smallest = left
            sf = hp_f[left]
        if right < size and hp_f[right] < sf:
            smallest = right
            sf = hp_f[right]
        if smallest == i:
            break
        hp_node[i] = hp_node[smallest]
        hp_f[i] = hp_f[smallest]
        hp_pos[hp_node[i]] = i
        i = smallest
    hp_node[i] = node
    hp_f[i] = f
    hp_pos[node] = i


@njit()
def astar_search_heap_njit(s_flat_start, s_flat_goal, s_goal, inflated_map, move_set,
                           manhattan):
    """Batched A* using a per-agent decrease-key binary min-heap for OPEN.

    The committed planner scans the whole OPEN array (O(n) over ~36k cells) to pop
    the next node every iteration; this replaces that with a heap (O(log n) pops /
    decrease-key), each agent searched independently to its goal. Same cost
    (octile/manhattan), collision rule and tie-break (f = g + h + 1e-4*h). With an
    admissible, consistent heuristic the result is still an optimal-cost path, but
    f-ties may resolve differently than the linear scan, so paths can differ from
    the reference while having identical length/cost (validated in tests).

    Returns (PARENT, failed); failed is True if any agent cannot reach its goal.
    """
    h = inflated_map.shape[0]
    w = inflated_map.shape[1]
    n = h * w
    batch = s_flat_start.shape[0]
    sqrt2 = 1.4142135623730951
    tol = 1e-6
    PARENT = np.full((batch, n), -1, dtype=np.int64)
    failed = False

    for b in range(batch):
        start = s_flat_start[b]
        goal = s_flat_goal[b]
        gx = s_goal[b, 0]
        gy = s_goal[b, 1]
        g = np.full(n, np.inf)
        closed = np.zeros(n, dtype=np.bool_)
        hp_node = np.empty(n, dtype=np.int64)
        hp_f = np.empty(n, dtype=np.float64)
        hp_pos = np.full(n, -1, dtype=np.int64)

        g[start] = 0.0
        PARENT[b, start] = start
        sx = start // h
        sy = start % h
        hdx = sx - gx if sx >= gx else gx - sx
        hdy = sy - gy if sy >= gy else gy - sy
        if manhattan:
            hn = float(hdx + hdy)
        else:
            hmn = hdx if hdx < hdy else hdy
            hmx = hdx if hdx > hdy else hdy
            hn = hmn * sqrt2 + (hmx - hmn)
        hp_node[0] = start
        hp_f[0] = hn + 0.0001 * hn
        hp_pos[start] = 0
        size = 1

        found = False
        while size > 0:
            cur = hp_node[0]
            size -= 1
            hp_pos[cur] = -1
            if size > 0:
                hp_node[0] = hp_node[size]
                hp_f[0] = hp_f[size]
                hp_pos[hp_node[0]] = 0
                _heap_sift_down(hp_node, hp_f, hp_pos, 0, size)
            if cur == goal:
                found = True
                break
            if closed[cur]:
                continue
            closed[cur] = True
            cx = cur // h
            cy = cur % h
            gcur = g[cur]
            for m in range(move_set.shape[0]):
                nx = cx + move_set[m, 0]
                ny = cy + move_set[m, 1]
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue
                if inflated_map[ny, nx] != 0:
                    continue
                nf = nx * h + ny
                if closed[nf]:
                    continue
                adx = nx - cx if nx >= cx else cx - nx
                ady = ny - cy if ny >= cy else cy - ny
                if manhattan:
                    mc = float(adx + ady)
                else:
                    mn = adx if adx < ady else ady
                    mx = adx if adx > ady else ady
                    mc = mn * sqrt2 + (mx - mn)
                ng = gcur + mc
                if ng + tol < g[nf]:
                    g[nf] = ng
                    PARENT[b, nf] = cur
                    hdx = nx - gx if nx >= gx else gx - nx
                    hdy = ny - gy if ny >= gy else gy - ny
                    if manhattan:
                        hh = float(hdx + hdy)
                    else:
                        hmn = hdx if hdx < hdy else hdy
                        hmx = hdx if hdx > hdy else hdy
                        hh = hmn * sqrt2 + (hmx - hmn)
                    fv = ng + hh + 0.0001 * hh
                    pos = hp_pos[nf]
                    if pos == -1:
                        hp_node[size] = nf
                        hp_f[size] = fv
                        hp_pos[nf] = size
                        _heap_sift_up(hp_node, hp_f, hp_pos, size)
                        size += 1
                    else:
                        hp_f[pos] = fv  # decrease-key (g only ever decreases)
                        _heap_sift_up(hp_node, hp_f, hp_pos, pos)
        if not found:
            failed = True

    return PARENT, failed


@njit()
def expand_neighbors_njit(s_flat, move_set, g, PARENT, OPEN, inflated_map, s_goal,
                          manhattan):
    """Fully-compiled equivalent of check_neighbors_and_update_cost + the OPEN
    update loop that followed it.

    The original was plain Python (a per-neighbor loop building a list of tuples,
    calling numba helpers piecemeal) and was the planner's #1 hotspot — most of
    its cost was interpreter overhead, not compute. This does the identical work
    in one njit pass, updating g / PARENT / OPEN in place:
      for each agent's current node, for each move, relax the neighbour if the new
      g-cost is lower, and set OPEN = g + h + 1e-4*h (the original tie-break).
    Cost / heuristic are the octile (or manhattan) distance, exactly as cost() and
    heuristic(); a neighbour in an occupied cell has infinite move cost (skipped),
    matching is_collision(). Moves are all +/-1 and the inflated map has occupied
    borders, so neighbours always stay in bounds (guarded anyway).
    """
    tol = 1e-6
    h = inflated_map.shape[0]
    w = inflated_map.shape[1]
    sqrt2 = 1.4142135623730951
    for b in range(s_flat.shape[0]):
        cur = s_flat[b]
        sx = cur // h
        sy = cur % h
        if inflated_map[sy, sx] != 0:
            continue  # current in obstacle -> every move cost is inf
        gcur = g[b, cur]
        gx = s_goal[b, 0]
        gy = s_goal[b, 1]
        for m in range(move_set.shape[0]):
            nx = sx + move_set[m, 0]
            ny = sy + move_set[m, 1]
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            if inflated_map[ny, nx] != 0:
                continue  # neighbour occupied -> inf move cost (is_collision)
            adx = nx - sx if nx >= sx else sx - nx
            ady = ny - sy if ny >= sy else sy - ny
            if manhattan:
                move_cost = float(adx + ady)
            else:
                mn = adx if adx < ady else ady
                mx = adx if adx > ady else ady
                move_cost = mn * sqrt2 + (mx - mn)
            new_cost = gcur + move_cost
            n_flat = nx * h + ny
            if new_cost + tol < g[b, n_flat]:
                g[b, n_flat] = new_cost
                PARENT[b, n_flat] = cur
                hdx = nx - gx if nx >= gx else gx - nx
                hdy = ny - gy if ny >= gy else gy - ny
                if manhattan:
                    hn = float(hdx + hdy)
                else:
                    hmn = hdx if hdx < hdy else hdy
                    hmx = hdx if hdx > hdy else hdy
                    hn = hmn * sqrt2 + (hmx - hmn)
                OPEN[b, n_flat] = new_cost + hn + 0.0001 * hn


@njit()
# @profile
def f_value(s, g, inflated_map, s_goal, heuristic_type):
    """
    f = g + h. (g: Cost to come, h: heuristic value)
    :param s: current state
    :return: f
    """
    # f = g[np.arange(len(s)), get_flat_index(s, inflated_map)] + heuristic(s, s_goal, heuristic_type)
    f2 = np.zeros(len(s))
    h = heuristic(s, s_goal, heuristic_type)
    # to make it a little bit faster we are prioritizing nodes closer to the goal

    # h2 = heuristic(s, s_goal, "manhattan" if heuristic_type != "manhattan" else "")
    flatten = get_flat_index(s, inflated_map)
    for i in range(flatten.shape[0]):
        f2[i] = g[i, flatten[i]] + h[i] + 0.0001 * h[i]
    # nbp_fill_f2(flatten, f2, g, h)

    return f2


@njit(parallel=False)
def nbp_fill_f2(flatten, f2, g, h):
    for i in prange(flatten.shape[0]):
        f2[i] = g[i, flatten[i]] + h[i] + 0.0001 * h[i]


@njit()
# @profile
def heuristic(s, s_goal, heuristic_type):
    """
    Calculate heuristic.
    :param s: current node (state)
    :return: heuristic function value
    """
    # s_goal = s_goal_i.astype(np.float64)
    # s = s_i.astype(np.float64)
    if heuristic_type == "manhattan":
        # c = np.abs(s_goal[:, 0] - s[:, 0]) + np.abs(s_goal[:, 1] - s[:, 1])
        c = np.abs(s_goal - s)
        c = c[:, 0] + c[:, 1]
        c = c.astype(np.float64)
    else:
        # TOO NAIVE we are still moving in grids
        c = s_goal - s
        c = np.abs(c)
        min_c = np.minimum(c[:, 0], c[:, 1])
        max_c = np.maximum(c[:, 0], c[:, 1])
        cross_move_cost = 1.4142135623730951
        c = min_c * cross_move_cost + max_c - min_c
        # c = np.hypot(c[:, 0], c[:, 1])

        # we can only move

    return c


@njit(parallel=False)
def inflate_map(grid_map, radius):
    """
    Inflate the obstacles in the map
    :param grid_map: The grid map
    :param radius: The radius of the robot
    :return: The inflated map
    """
    inflated_map = np.zeros(grid_map.shape)
    for i in range(grid_map.shape[0]):
        for j in prange(grid_map.shape[1]):
            if grid_map[i, j] < 255:
                for k in range(-radius, radius + 1):
                    for l in range(-radius, radius + 1):
                        if (
                            0 <= i + k < grid_map.shape[0]
                            and 0 <= j + l < grid_map.shape[1]
                        ):
                            inflated_map[i + k, j + l] = 1.0
    return inflated_map


def inflate_map_np(grid_map, radius):
    return circular_dilation(grid_map, radius)


@njit()
def circular_dilation(grid_map, radius):
    output_map = np.zeros_like(grid_map, dtype=np.uint8)
    rows, cols = grid_map.shape
    r_squared = radius**2  # Pre-compute radius squared for circle equation

    for x in prange(rows):
        for y in prange(cols):
            if grid_map[x, y] < 255:  # If it's an obstacle
                # Check surrounding cells within the radius distance
                min_x = max(0, x - radius)
                max_x = min(rows, x + radius + 1)
                min_y = max(0, y - radius)
                max_y = min(cols, y + radius + 1)

                for i in range(min_x, max_x):
                    for j in range(min_y, max_y):
                        if (x - i) ** 2 + (y - j) ** 2 <= r_squared:
                            output_map[i, j] = 255

    return output_map


@njit()
def get_flat_index(s, grid_map):
    """
    Hash the state
    :param s: The state
    :return: The hash value
    """
    return s[:, 0] * grid_map.shape[0] + s[:, 1]


@njit()
def get_map_index(flat_index, grid_map):
    """
    Get the state from hash value
    :param flat_index: The hash value
    :return: The state
    """
    s = np.zeros((flat_index.shape[0], 2))
    s[:, 0] = flat_index // grid_map.shape[0]
    s[:, 1] = flat_index % grid_map.shape[0]
    return s.astype(np.int64)
    # x = flat_index // grid_map.shape[0]
    # y = flat_index % grid_map.shape[0]
    # s = np.array([x, y])


@njit()
def nb_argmin(s_flat, OPEN, is_not_found):
    for i in range(len(is_not_found)):
        if not is_not_found[i]:
            continue
        not_inf = np.where(OPEN[i] != np.inf)[0]
        s_flat[i] = not_inf[OPEN[i, not_inf].argmin()]
    return s_flat


@njit(parallel=True)
def nbp_argmin(s_flat, OPEN, is_not_found):
    for i in prange(len(is_not_found)):
        if not is_not_found[i]:
            continue
        not_inf = np.where(OPEN[i] != np.inf)[0]
        if OPEN[i, not_inf].size == 0:
            debug = 0
        s_flat[i] = not_inf[OPEN[i, not_inf].argmin()]
    return s_flat


class BatchAStar:
    """AStar set the cost + heuristics as the priority
    MAP = 1 or above for obstacles and 0. for free space
    """

    def __init__(
        self,
        grid_map: np.ndarray,
        s_start: np.ndarray,
        s_goal: np.ndarray,
        inflate_radius,
        heuristic_type,
        verbose=False,
        draw=False,
    ):
        self.verbose = verbose
        self.s_start = np.round(s_start).copy().astype(int)
        self.s_goal = np.round(s_goal).copy().astype(int)
        # heuristic_type = "eculidean"

        self.heuristic_type = heuristic_type
        self.batch_size = len(self.s_start)

        self.grid_map = grid_map
        self.inflated_map = inflate_map_np(self.grid_map, inflate_radius)
        # make border on inflated map so we stop the search at edge
        self.inflated_map[0, :] = 255
        self.inflated_map[-1, :] = 255
        self.inflated_map[:, 0] = 255
        self.inflated_map[:, -1] = 255
        # self.inflated_map = np.stack([self.inflated_map] * len(self.s_start), axis=2)
        self.s_flat_goal = get_flat_index(self.s_goal, self.grid_map)
        self.s_flat_start = get_flat_index(self.s_start, self.grid_map)
        if heuristic_type == "manhattan":
            self.move_set = np.array(
                [
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                    (1, 0),
                ],
                dtype=int,
            )  # feasible input set
        else:
            self.move_set = np.array(
                [(-1, 0), (0, 1), (0, -1), (1, 0), (-1, 1), (1, -1), (1, 1), (-1, -1)],
                dtype=int,
            )  # feasible input set

        self.OPEN = np.array(
            [np.full((grid_map.shape[0] * grid_map.shape[1]), np.inf)] * self.batch_size
        )  # priority queue / OPEN set
        self.CLOSED = np.zeros(
            (self.batch_size, grid_map.shape[0] * grid_map.shape[1]), dtype=bool
        )  # CLOSED set / VISITED order
        self.PARENT = (
            np.zeros(
                (self.batch_size, grid_map.shape[0] * grid_map.shape[1]), dtype=int
            )
            - 1
        )  # recorded parent
        self.g = np.array(
            [np.full((grid_map.shape[0] * grid_map.shape[1]), np.inf)] * self.batch_size
        )  # cost to come # prediction

        # For debugging and visualization
        self.canvas_index = 0
        if draw:
            self.canvas = self.grid_map.copy().astype(np.uint8)
            cv2.circle(self.canvas, self.s_goal[self.canvas_index], 3, 150, -1)
            cv2.circle(self.canvas, self.s_start[self.canvas_index], 6, 200, -1)
        self.draw = draw

        # self.warmup_func()

    def reset_canvas(self, canvas_index=None):
        if canvas_index is not None:
            self.canvas_index = canvas_index

        self.canvas = self.grid_map.copy().astype(np.uint8)
        closed_nodes = np.argwhere(self.CLOSED[self.canvas_index].squeeze())
        for flat_node in closed_nodes:
            s = get_map_index(flat_node, self.grid_map)
            self.canvas[s[0, 1], s[0, 0]] = 150

            # cv2.circle(self.canvas, s, 1, 150, -1)

        cv2.circle(self.canvas, self.s_goal[self.canvas_index], 3, 150, -1)
        cv2.circle(self.canvas, self.s_start[self.canvas_index], 6, 200, -1)

        # self.show_im(self.canvas, "canvas")

    def draw_node(self, s_flat, value=150):
        # if len(s_flat) != 1:
        #     s_flat = s_flat[self.canvas_index].unsqueeze(0)
        try:
            ss = get_map_index(s_flat, self.grid_map)
            if ss.ndim == 2:
                if len(s_flat) != 1:
                    s = ss[self.canvas_index]
                else:
                    s = ss[0]
            else:
                s = ss
            self.canvas[s[1], s[0]] = value
        except:
            print("error")
            debug = 0
        # cv2.circle(self.canvas, s[self.canvas_index], 1, 150, -1)

    @staticmethod
    def show_im(im, name):
        im = cv2.resize(im, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
        if im.max() > 1.0:
            im = im / 255.0
        cv2.imshow(name, im)
        key = cv2.waitKey(1)
        if key == ord("s"):
            pass

    # @profile
    def searching(self, print_time=False, test=False) -> (List[np.ndarray], np.ndarray):
        """
        A_star Searching.
        :return: path, visited order
        """
        # Grid discretization + obstacle inflation can land a start/goal in an
        # occupied cell even when it is geometrically clear. Snap such endpoints
        # to the nearest free cell instead of aborting the whole batch (one bad
        # agent used to fail planning for every agent in the world, causing the
        # reset-retry stalls seen in the corridor scenario).
        self.s_start = snap_to_free(self.s_start, self.inflated_map)
        self.s_goal = snap_to_free(self.s_goal, self.inflated_map)
        self.s_flat_start = get_flat_index(self.s_start, self.grid_map)
        self.s_flat_goal = get_flat_index(self.s_goal, self.grid_map)

        # Fast path: heap-based A* (O(log n) pops). The Python loop below is kept
        # only for the interactive draw/visualization case.
        if not self.draw:
            PARENT, failed = astar_search_heap_njit(
                self.s_flat_start.astype(np.int64),
                self.s_flat_goal.astype(np.int64),
                self.s_goal.astype(np.int64),
                self.inflated_map,
                self.move_set.astype(np.int64),
                self.heuristic_type == "manhattan",
            )
            self.PARENT = PARENT
            if failed:
                if self.verbose:
                    print("Cannot find the path")
                return None, None
            return (
                extract_path(self.PARENT, self.s_goal, self.s_start, self.grid_map),
                self.CLOSED,
            )

        flat_index = self.s_flat_start
        batch_index = np.arange(self.batch_size)
        f_i = np.stack(
            [np.arange(self.batch_size), get_flat_index(self.s_start, self.grid_map)],
            axis=1,
        )
        self.PARENT[batch_index, flat_index] = flat_index
        self.g[batch_index, flat_index] = 0
        # self.g[get_flat_index(self.s_goal, self.grid_map)] = np.inf

        # insert the start node
        self.OPEN[batch_index, flat_index] = f_value(
            self.s_start, self.g, self.inflated_map, self.s_goal, self.heuristic_type
        )

        global_start = time.perf_counter()
        epoch = 0
        render_time = 1
        is_not_found = np.ones(self.batch_size, dtype=bool)
        s_flat = np.argmin(self.OPEN, axis=1)
        close_nodes = [s_flat[0]]
        OPEN_NOT_INF = [[s_flat[i]] for i in range(self.batch_size)]
        while self.OPEN[batch_index, s_flat].min() < np.inf:
            # print(f"=============== epoch: {epoch} ===============")
            if self.draw:
                # print(f"close nodes: {close_nodes}")
                # print(s_flat[0], self.OPEN[self.canvas_index, s_flat[0]])
                if s_flat[0] not in close_nodes:
                    debug = 0
                else:
                    close_nodes.remove(s_flat[0])
                self.reset_canvas()
                for i, node in enumerate(self.OPEN[self.canvas_index]):
                    if node != np.inf:
                        self.draw_node(np.array([i]), int((node - 140) * 5))
                if self.OPEN[self.canvas_index, s_flat[0]] != np.inf:
                    best_nodes = np.where(
                        np.isclose(
                            self.OPEN[self.canvas_index],
                            self.OPEN[self.canvas_index, s_flat[0]],
                            0.001,
                        )
                    )[0]

                    best_nodes = np.expand_dims(best_nodes, axis=-1)
                    close_nodes = []
                    for best_node in best_nodes:
                        self.draw_node(best_node, 100)
                        if best_node != s_flat[0] and best_node not in close_nodes:
                            close_nodes.append(best_node[0])

                self.draw_node(np.array([s_flat[0]]), 150)

                if epoch % render_time == 0:
                    self.show_im(self.canvas, "canvas")

            if np.any(s_flat == self.s_flat_goal):  # stop condition
                done = np.argwhere(s_flat == self.s_flat_goal)
                #
                is_not_found[done] = False

                if np.logical_not(is_not_found).all():
                    break
                if self.draw and not is_not_found[self.canvas_index]:
                    # find first not found
                    self.reset_canvas(np.argwhere(is_not_found)[0][0])

            # Compiled neighbour expansion + OPEN update (identical logic to the
            # old Python check_neighbors_and_update_cost; see expand_neighbors_njit).
            expand_neighbors_njit(
                s_flat,
                self.move_set,
                self.g,
                self.PARENT,
                self.OPEN,
                self.inflated_map,
                self.s_goal,
                self.heuristic_type == "manhattan",
            )

            # set current node as done (not open)
            self.OPEN[batch_index, s_flat] = np.inf
            # print(f"\t\tself.OPEN[batch_mask, s_n]: {self.OPEN[batch_mask, s_n]}")
            self.CLOSED[batch_index, s_flat] = True
            s_flat[:] = 0

            try:
                if test:
                    # s_flat is being updated in place
                    nb_argmin(s_flat, self.OPEN, is_not_found)
                else:
                    nbp_argmin(s_flat, self.OPEN, is_not_found)
            except Exception as e:
                if self.verbose:
                    print(e)
                    print(
                        f"{Fore.YELLOW}DEV warning: Most likely objects are splitting the map into two so there is not path connecting start and goal{Style.RESET_ALL}"
                    )
                return None, None
                # s_flat[is_not_found] = self.OPEN[is_not_found].argmin(1)

            # print(len(np.where(self.OPEN != np.inf)[0]))
            epoch += 1
            # Find new lowest cost node at the end to only run this once
        if print_time:
            print(
                f"check_neighbors_and_update_cost time: {time.perf_counter() - global_start:.6f}s"
            )
        if not np.logical_not(is_not_found).all():
            if self.verbose:
                print("Cannot find the path")
            return None, None
        return (
            extract_path(self.PARENT, self.s_goal, self.s_start, self.grid_map),
            self.CLOSED,
        )

    def warmup_func(self):
        batch = 3
        m_size = 10
        s = np.random.randint(0, m_size * m_size, size=batch)
        g = np.random.rand(batch, m_size * m_size)
        c = check_neighbors_and_update_cost(
            s,
            self.move_set,
            g,
            np.zeros((batch, m_size * m_size), dtype=int) - 1,
            self.inflated_map,
            np.zeros((batch, 2)),
            self.heuristic_type,
        )
