# docker/

Compose-based dev environment. `docker-compose.yml` defines the main
service with the CUDA/torch/vmas stack pre-installed.

    cp .env.example .env      # fill in WANDB_API_KEY
    docker compose up -d
    docker compose exec giant bash

Training inside the container needs a virtual display:

    xvfb-run -a python3 -m train.LidarSingleStep --config configs/smoke.yaml
