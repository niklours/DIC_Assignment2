# DIC_Assignment2

This is the repository containing the Data Intelligence Challenge-2AMC15 challenge environment code.

## Table of Contents
- [DIC\_Assignment2](#dic_assignment2)
  - [Table of Contents](#table-of-contents)
  - [Quickstart](#quickstart)
  - [Examples for quick testing](#examples-for-quick-testing)
  - [Code Guide](#code-guide)
    - [The `agent` Module](#the-agent-module)
    - [The Environment](#the-environment)
    - [Image Generation](#image-generation)
    - [Helper Module](#helper-module)
  - [Usage of `train.py`](#usage-of-trainpy)

## Quickstart
Follow these steps to set up and start using the simulation environment:

1. **Directory**
    
    Change to the directory of the project: ``cd <path/DIC_Assignment2>``. If you want to create a virtual environment for this course with Python >= 3.10. Using conda, you can do: conda create -n dic2025 python=3.11. Use conda activate dic2025 to activate it conda deactivate to deactivate it.
1. **Clone Repository**
    
    Clone this repository into the local directory you prefer ``https://github.com/niklours/DIC_Assignment2.git``
1. **Dependencies and Environment Setup**
    - Install dependencies using ``pip install -r requirements.txt``
    - If you use an environment activate it and install dependencies there.
2. **Run Demo**
      
      To run our demo with some of the experiments and the generated plots, run on Git Bash terminal:
    ```bash
        bash demo.sh
    ```
      Or, if you only want to see a single training example run on PowerShell terminal:
      ```bash 
          $ python train.py --agent_type=DQN --episodes=500 --steps=100 --gamma=0.95 --lr=1e-3 --sa --br=4 --tol=300 --env=1
      ```


## Examples for quick testing
**IMPORTANT**: For the execution of the scripts it is necessary to be in the directory
``DIC_Assignment2``

Different configurations of the parameters can affect agent's performance. Keep the default parameters for fast for convergence of the agents. Below you can find some configurations for running quick tests:

   1. For agent ``DQNAgent``: You can train: `pythonpython train.py --agent_type=DQN --sa --env=0`
   2. For agent ``DDQNAgent``: You can train: `pythonpython train.py --agent_type=DuelingDQN --sa --env=0`

## Code Guide

### The `agent` Module
This module contains the agents for this project include:
   1. `dqn_agent.py` - Deep Q-Network Agent
   2. `dueling_dqn.py` - Dueling Deep Q-Network Agent


### The Environment
- **Purpose**: The `new_environment` class is where the agent acts. It includes key methods like `move_agent_direction()`, `get_state_vector()` and `step_with_reward()`

### Image Generation
- **Purpose**: The purpose of `img_gen.py` module is the visualization of the 2D space with the target and obstacles, where the agents acts and the path that selects.

### Helper Module
- **Purpose**: The purpose of `helper.py` module is to provide the creation of the two ready-to-use environments and evaluate agent's performance.
- **Environment Setup**: Use `helper.setup_env()` and `helper.setup_env_hard()` to create the easy and harder one environment, respectively.
- **Evaluation** : Use `helper.eval_agent()` to evaluate agent's performance (after training). Results are saved in `logs` file

## Usage of `train.py`
`train.py` does the training of our Reinforcement Learning Agents with customizable parameters:


```bash


DIC Reinforcement Learning Trainer.

OPTIONS:
  -h:  show this help message and exit
  --agent_type: Agent type to train. Possible choices: DQN/DuelingDQN. (str, default="DQN")
  --episodes: Number of iterations to go through. (int, default=500)
  --steps: Number of steps per episode. (int, default=100)
  --epsilon: The epsilon parameter. If not specified, defaults to a decaying rate. (float, default=None)
  --gamma: The discount factor parameter. (float, default=0.95)
  --lr: The learning rate. (float, default=0.001)
  --sa: Enable Strategic Adaptation approach. Add --sa to activate (boolean, default=False)
  --br: Number of brackets for SA. (int, default=4)
  --tol: Stopping criterion success history size (int, default=300)
  --env: Selecting which of the two environment functions to use 0 for easy, 1 for hard."

```
**Example**: ```python train.py --agent_type=DQN --episodes=500 --steps=100 --gamma=0.95 --lr=1e-3 --sa --br=4 --tol=300 --env=1```

