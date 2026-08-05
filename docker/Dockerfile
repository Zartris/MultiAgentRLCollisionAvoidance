# syntax=docker/dockerfile:1
# MALP Development Container
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# ── Build args ──────────────────────────────────────────────────────
ARG USERNAME=zartris
ARG CONTAINER_USER=${USERNAME}
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# ── 1. Core system packages ────────────────────────────────────────
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y \
        git \
        curl \
        python3 \
        python3-pip \
        python3-venv \
        cmake \
        build-essential \
        libgl1 \
        libglu1-mesa \
        libglib2.0-0 \
        xvfb \
        sudo \
        tmux

# ── 2. GitHub CLI ──────────────────────────────────────────────────
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y gh

# ── 3. Create non-root user ───────────────────────────────────────
RUN (userdel -r ubuntu 2>/dev/null || true) \
    && (groupdel ubuntu 2>/dev/null || true) \
    && (groupadd -g ${USER_GID} ${CONTAINER_USER} 2>/dev/null || true) \
    && adduser --disabled-password --gecos "" --uid ${USER_UID} --gid ${USER_GID} ${CONTAINER_USER} \
    && echo "${CONTAINER_USER} ALL=(ALL:ALL) NOPASSWD: ALL" >> /etc/sudoers

# ── 4. Switch to non-root user ────────────────────────────────────
USER ${CONTAINER_USER}
WORKDIR /home/${CONTAINER_USER}

ENV PATH=${PATH}:/home/${CONTAINER_USER}/.local/bin

# ── 5. Python dependencies ────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --break-system-packages -r /tmp/requirements.txt

# ── 6. Python-RVO2 (baseline; source build, needs cmake<4) ────────
# mit-acl fork: rvo_policy.py needs setAgentCollabCoeff (NH-ORCA), absent from
# the standard sybrenstuvel fork. --no-build-isolation so the build reuses the
# Cython/cmake from step 5 (its setup.py imports Cython without declaring it, so
# an isolated build fails with "No module named 'Cython'").
RUN git clone https://github.com/mit-acl/Python-RVO2.git /tmp/Python-RVO2 \
    && pip3 install --break-system-packages --no-build-isolation /tmp/Python-RVO2 \
    && rm -rf /tmp/Python-RVO2

# ── Entry ──────────────────────────────────────────────────────────
WORKDIR /repo
CMD ["/bin/bash"]
