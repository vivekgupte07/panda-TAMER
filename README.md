# TAMER Framework for the Panda Robot Arm

This repository is an implementation of an interactive learning framework 'TAMER' for training an end-to-end position controller on the Panda robot. A Gym-Pybullet interface is used for training the tasks. This project is my thesis work, where we compare the traditional reward engineering and the interactive TAMER framework to train the robot to reach a certain position goal. We ask users who are not robotics or learning experts to develop simple conditions that act as reward signal to the RL agent and also ask them to use a simple key-pressing interface to give rewards to the TAMER agent. In both cases, the agent is the Panda robot. 

Find the thesis on this entire project here: [Human in the Loop Reinforcement Learning for Robot Controllers](https://drive.google.com/file/d/1K-A3tTsfCGSUG_VyQyUD87Vf8tP68jSC/view?usp=sharing)
## Environment and Agent setup

The Pybullet simulator with the robot looks as shown in the figure below:

![env](https://github.com/vivekgupte07/panda-TAMER/assets/67819132/eaf5841d-4f6c-46bb-84f3-a5a119c4f275) 

This gym-pybullet interface used in this tutorial was taken from a [tutorial series](https://www.etedal.net/2020/04/pybullet-panda.html).
Tutorial from the same series on [making your own environment](https://www.etedal.net/2020/04/pybullet-panda_2.html).

To install `gym-panda`:
```ruby
pip install gym-panda
```
Or build from source as shown [here](https://www.etedal.net/2020/04/pybullet-panda_2.html#:~:text=Building%20from%20Source).

For this environment to work, a change in the `info` struct in the `step` method of `gym-panda/envs/panda-env.py` needs to be initialized as type *dict*:
```ruby
return np.array(self.observation).astype(np.float32), self.reward, done, {}

```
If you want to use the envs from this repo, replace the folder in `.../site-packages/` with the `/gym-panda` folder in this repo after installing `gym-panda` package.

In this environment, where the base of the arm is fixed on the table $(0,0,0)$. The state space is the end effector coordinates in the robot reference frame. The orientation of the end-effector is fixed for simplicity. The action space is the target co-ordinates, and the arm takes a small step towards the target at each time step. These definitions are present in `gym-panda/envs/panda-env.py`. All changes or definitions at the agent or environment level go into this script. The `panda-envs` folder has examples of differently set MDPs used for various tests in the project.

The reward function can be introduced in the step function.

## More on TAMER 
TAMER stands for "**T**raining an **Ag**ent **M**annually using **E**valuative **R**einforcement". It is an interactive reward shaping method introduced by [Knox and Stone (2008)]([https://dl.acm.org/doi/abs/10.1145/1597735.1597738](https://ieeexplore.ieee.org/abstract/document/4640845)). This method is inspired from a need to allow easier method for humans to transfer their knowledge to robots, allowing robots to learn complex tasks more efficiently. This framework also provides an alternative to problem of reward engineering for learning complex tasks.

In this framework, the user gives reward, as feedback, to the agent's actions (using some simple interface). The agent's state, the action taken and the user's feedback on this action is used by the agent to learn a model of the human's reward function. The agent then uses this model to greedily choose actions that it predicts would receive the maximum reward. As the user continues to give feedback during the training, the model continuously improves, and eventually, it can directly give rewards to the agent. This framework thus provides a simple method for the user to shape a reward function by using a simple interace. By changing the way the rewards are given, the agent can also be trained to follow a different policy.

Various function approximators such as regression, nearest neighbours and artificial as well as deep neural networks can be used to learn the reward function. The reward model can also be combined with an environmental reward signal. The actions can be chosen to maximize immediate reward or discounted long term returns. The rewards given by the humans can be binary or otherwise, and the framework can be used in discrete as well as continuous tasks. In a discrete space task, the agent can wait for the human user's feedback before taking the next step. For continuous task spaces, where the agent cannot wait for the human user's rewards, a credit assigner is used in [Knox and Stone (2009)](https://dl.acm.org/doi/abs/10.1145/1597735.1597738).

Go to [Knox's website](https://www.bradknox.net/human-reward/) to read about his entire work on human generated rewards.
Watch [this](https://www.youtube.com/watch?v=xzqCzX5ExZA) video where a team from the MIT Media Lab train a robot to show different behaviours as per the human user's reward.

## Training (RL AND TAMER)

### Algorithms
`stable-baselines3 ` has been used for making vecotrized environments and training. This repo has several scripts (could have made them into one) that are used to make the environment and then start the training episodes. The algorithms are also provided by stable-baselines3. The hyperparameters are detailed in the thesis (see above). 

Find the `stable-baselines3` installation instructions [here](https://stable-baselines3.readthedocs.io/en/master/guide/install.html#installation:~:text=Edit%20on%20GitHub-,Installation,-%EF%83%81).

> Stable-Baselines3 requires python 3.8+ and PyTorch >= 1.13

For installation using `pip`:
```ruby
pip install stable-baselines3
```
Run the `controlller*.py` scipts for training.
### TAMER

For TAMER, we convert the environment into a discreet env using the pybullet simulator and increase the step size. This is done to allow the human user sufficient time to reward the actions. 

**Reward models**

We use differnet approximators - linear regression, k-nearest neighbours and ANN based binary classification. The best results are obtained using the neural network. The approximated models are updated after a certain number of human responses. Find the reward models in `reward-functions/`.


**Actions**

Actions are selected greedily based on different reward models used to approximate the human reward funciton. The different `action-select` scripts in the repo are used for that purpose.

For more information feel free to contact me at vgupte@ucsd.edu
