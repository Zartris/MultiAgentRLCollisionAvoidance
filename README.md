# Cooperative Multi-Agent Collision Avoidance with Attentive Graph Networks and Global Path Integration
| <img src="docs/doorway_800x600.gif" alt="GIF 1" width="380"/> | <img src="docs/circle_800x600.gif" alt="GIF 2" width="380"/> |
|:---------------------------------------------------------:|:---------------------------------------------------------:|
| <img src="docs/hallway_800x600.gif" alt="GIF 3" width="380"/> | <img src="docs/random_800x600.gif" alt="GIF 4" width="380"/> |
## 🚨 **Paper Under Review** 🚨
### **Everything will be visible when accepted.**
This repository contains the official implementation of the paper **"Cooperative multi-agent collision avoidance with attentive graph networks and global path integration"**. This work introduces a novel approach to multi-robot collision avoidance, integrating global path planning with local navigation strategies.

The main contributions of this work include:
- We introduce a local navigation model that incorporates global path information within the observation space, enabling the agent to maintain adherence to pre-planned routes while reacting to dynamic changes in the environment.
- The model employs graph structures to represent and manage interactions with neighboring agents using attentive graph neural networks, improving the robots’ ability to navigate dense environments.
- The ability to navigate in complex, dynamic environments with noisy sensor data.
- Superior performance when compared to other baselines like NH-OCRA, DLR-NAV, and GA3C-CADRL in multiple simulated environments.

  


## Table of Contents
- [Cooperative Multi-Agent Collision Avoidance with Attentive Graph Networks and Global Path Integration](#cooperative-multi-agent-collision-avoidance-with-attentive-graph-networks-and-global-path-integration)
  - [Table of Contents](#table-of-contents)
  - [Problem Statement](#problem-statement)
  - [Our Solution](#our-solution)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Project Structure](#project-structure)
  - [Links](#links)
  - [Contributing](#contributing)
  - [License](#license)

## Problem Statement
Multi-robot systems face significant challenges when navigating complex environments without effective global planning. Global planners are essential in predicting efficient paths and guiding robots toward their goals. However, without a global planner, robots can become stuck in local minima, unable to navigate through dense or dynamic environments effectively.

While many navigation methods are trained with a single goal point, they depend on running waypoints from a global planner to prevent them from becoming trapped in these local minima. The lack of coordination between global objectives and local navigation often results in inefficiencies, as robots focus too narrowly on immediate targets, missing the broader context of the overall path.
|<img src="docs/LocalMinima_github_800x450.gif" alt="GIF 1" width="450"/>|<img src="docs/GPFocus_github_800x670.gif" alt="GIF 2" width="300"/>|
|-------------------|-------------------|

## Our Solution
Our approach addresses these challenges by integrating global path planning with local navigation in a more balanced and efficient way. First, we separate the goal point from the global path in the observation space, allowing the network to learn when to prioritize the global path versus immediate local goals. This distinction helps the robot balance its focus, avoiding over-fixation on temporary waypoints and reducing the risk of getting stuck in new local minima.

Additionally, our model uses both raw sensor data for understanding the complexities of the environment and processed data for tracking dynamic objects like neighboring agents. By incorporating attentive graph neural networks, our model efficiently handles interactions with multiple agents. The attention mechanism allows the network to dynamically focus on the most relevant agents in the environment, ensuring that the robot can navigate safely and efficiently even in dense, dynamic scenarios.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/collision-avoidance.git
   cd collision-avoidance
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Install other environment dependencies such as simulation environments (e.g., VMAS) by following the instructions in `docs/INSTALL.md`.

## Usage

To run the training script, use the following command:
```bash
python3 train.py --config config.yaml
```

To evaluate the trained model in a specific environment, run:
```bash
python3 eval.py --env random
```

## Project Structure
```
├── evaluate/                  # Evaluation scripts and results
│   ├── results/               # Evaluation results (created during runtime)
│   ├── compare_models.py      # Print table of selected results from the results folder
│   ├── evaluate.py            # Running the evaluation script (remember to change the config in this file if needed)
│   └── generate_videos.py     # Just focusing on videos, not the results.
├── models/                    # Models and policies for training and evaluation
│   ├── baseline/              # Baseline models (GA3C_CADRL, RVO2, DRLNav)
│   ├── checkpoints/           # Checkpoints for saving models
│   ├── distributions.py       # Distributions utility for models
│   ├── model_loader.py        # Utility to load models
│   └── MultiAgentModel.py     # Our neural networks
├── scenario/                  # Scenarios for simulation and evaluation
│   ├── GlobalPlanner/         # Global path planning algorithm used in scenarios
│   ├── PaperScenarios/        # Scenarios used for showcasing the problem and not used in the evaluation
|   └── all scenarios .py      # All scenarios used in this paper (separated into different files).
├── sensors/                   # Sensor handling scripts (e.g., LiDAR for collision avoidance)
├── train/                     # Training scripts and utilities
│   ├── results/               # Training results (created during runtime)
│   └── utils/                 # Utility scripts for logging, loss functions, etc.
│   └── train.py               # Main script to start training (remember to change the config in this file if needed)
└── requirements.txt           # Required Python packages
```

## Links

- **Paper**: [Download the paper here](./path-to-your-paper.pdf)  
- **Video**: [Watch the demonstration video here](link-to-video)

## Contributing

Contributions are welcome! Please submit a pull request with any improvements or bug fixes. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.
