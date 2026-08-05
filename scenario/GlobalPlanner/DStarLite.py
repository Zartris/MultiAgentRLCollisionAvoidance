import numpy as np
from queue import PriorityQueue
from numba import njit

@njit
def heuristic(s1, s2, heuristic_type='euclidean'):
    if heuristic_type == 'euclidean':
        return np.linalg.norm(np.array(s1) - np.array(s2))
    elif heuristic_type == 'manhattan':
        return abs(s1[0] - s2[0]) + abs(s1[1] - s2[1])

@njit
def get_neighbors(s, grid_shape):
    neighbors = []
    move_set = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for move in move_set:
        neighbor = (s[0] + move[0], s[1] + move[1])
        if 0 <= neighbor[0] < grid_shape[0] and 0 <= neighbor[1] < grid_shape[1]:
            neighbors.append(neighbor)
    return neighbors

class DStarLite:
    def __init__(self, grid_map, s_start, s_goal, heuristic_type='euclidean'):
        self.grid_map = grid_map
        self.s_start = tuple(s_start)
        self.s_goal = tuple(s_goal)
        self.heuristic_type = heuristic_type

        self.km = 0
        self.rhs = {}
        self.g = {}
        self.U = PriorityQueue()

        self.initialize()

    def initialize(self):
        self.g[self.s_goal] = 0
        self.rhs[self.s_goal] = 0
        self.U.put((self.calculate_key(self.s_goal), self.s_goal))
        for row in range(self.grid_map.shape[0]):
            for col in range(self.grid_map.shape[1]):
                s = (row, col)
                if s != self.s_goal:
                    self.g[s] = float('inf')
                    self.rhs[s] = float('inf')

    def calculate_key(self, s):
        g_rhs = min(self.g.get(s, float('inf')), self.rhs.get(s, float('inf')))
        return (g_rhs + heuristic(self.s_start, s, self.heuristic_type) + self.km, g_rhs)

    def update_vertex(self, u):
        if u != self.s_goal:
            self.rhs[u] = min([self.g.get(s, float('inf')) + 1 for s in get_neighbors(u, self.grid_map.shape)])
        if u in self.U.queue:
            self.U.queue.remove((self.calculate_key(u), u))
        if self.g[u] != self.rhs[u]:
            self.U.put((self.calculate_key(u), u))

    def compute_shortest_path(self):
        while not self.U.empty() and (self.U.queue[0][0] < self.calculate_key(self.s_start) or
                                      self.rhs[self.s_start] != self.g[self.s_start]):
            k_old, u = self.U.get()
            if k_old < self.calculate_key(u):
                self.U.put((self.calculate_key(u), u))
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for s in get_neighbors(u, self.grid_map.shape):
                    self.update_vertex(s)
            else:
                g_old = self.g[u]
                self.g[u] = float('inf')
                self.update_vertex(u)
                for s in get_neighbors(u, self.grid_map.shape):
                    self.update_vertex(s)

    def plan(self, s_start):
        self.s_start = s_start
        self.compute_shortest_path()
        path = [self.s_start]
        current = self.s_start
        while current != self.s_goal:
            min_cost = float('inf')
            next_step = None
            for s in get_neighbors(current, self.grid_map.shape):
                if self.g.get(s, float('inf')) + 1 < min_cost:
                    min_cost = self.g.get(s, float('inf')) + 1
                    next_step = s
            if next_step is None:
                return None
            path.append(next_step)
            current = next_step
        return path

    def replan(self, current_position):
        self.km += heuristic(self.s_start, current_position, self.heuristic_type)
        self.s_start = current_position
        self.update_vertex(current_position)
        self.compute_shortest_path()
        return self.plan(current_position)