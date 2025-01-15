# Lunar Lander DDPG Reinforcement Learning

This project implements a Deep Deterministic Policy Gradient (DDPG) agent to solve the LunarLander-v2 continuous control environment from OpenAI Gym.

## Project Overview

The DDPG algorithm combines deep learning with reinforcement learning to handle continuous action spaces. This implementation trains an agent to successfully land a lunar module on a landing pad.

## Key Components

- `ddpg.py`: Implementation of the DDPG algorithm with Actor and Critic networks
- `train_lunarlander_continuous.py`: Script to train the DDPG agent
- `test_lunarlander_continuous.py`: Script to evaluate the trained agent

## Results

The repository includes:
- Trained model weights for both actor and critic networks
- Training progress visualization
- Performance metrics

### Demo Video

Watch the trained agent in action: [Watch Demo](https://github.com/muhabdullahd/lunarlander-ddpg/blob/main/lunarlander_demo.mov)


## Requirements

You can set up the environment using one of the following methods:

### Using Conda
```bash
conda env create -f environment.yml
conda activate pa4-torch
```

### Using Pip
```bash
pip install -r requirements.txt
```

## Usage

To train a new agent:
```bash
python train_lunarlander_continuous.py
```

To test a trained agent:
```bash
python test_lunarlander_continuous.py
```

## Learning Outcomes

This project demonstrates:
- Implementation of a deep reinforcement learning algorithm (DDPG)
- Training agents for continuous control tasks
- Hyperparameter tuning and optimization
- Performance analysis and visualization

## License

This project is licensed under the GNU General Public License v2.0 - see the [LICENSE](LICENSE) file for details.
