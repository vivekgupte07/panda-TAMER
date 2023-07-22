# TAMER Framework for the Panda Robot Arm

This repository is an implementation of an interactive learning framework 'TAMER' for training an end-to-end position controller on the Panda robot. A Gym-Pybullet interface is used for training the tasks. This project is my thesis work, where we compare the traditional reward engineering and the interactive TAMER framework to train the robot to reach a certain position goal. We ask users who are not robotics or learning experts to develop simple conditions that act as reward signal to the RL agent and also ask them to use a simple key-pressing interface to give rewards to the TAMER agent. In both cases, the agent is the Panda robot. 

The Pybullet simulator with the robot looks as shown in the figure below:

![env](https://github.com/vivekgupte07/panda-TAMER/assets/67819132/eaf5841d-4f6c-46bb-84f3-a5a119c4f275) 

This gym-pybullet interface used in this tutorial was taken from [this](https://www.etedal.net/2020/04/pybullet-panda.html) tutorial.

Using this environment, where the base of the arm is fixed on the table $(0,0,0)$. The state space is the end effector co-ordinates $(x, y, z)$. The orientation of the end-effector is fixed for simplicity. The action space is the target co-ordinates, and the arm takes a small step towards the target at each time step.




## More on TAMER 
TAMER stands for "**T**raining an **Ag**ent **M**annually using **E**valuative **R**einforcement". It is an interactive reward shaping method introduced by [Knox and Stone (2008)]([https://dl.acm.org/doi/abs/10.1145/1597735.1597738](https://ieeexplore.ieee.org/abstract/document/4640845)). This method is inspired from a need to allow easier method for humans to transfer their knowledge to robots, allowing robots to learn complex tasks more efficiently. This framework also provides an alternative to problem of reward engineering for learning complex tasks.

In this framework, the user gives reward, as feedback, to the agent's actions (using some simple interface). The agent's state, the action taken and the user's feedback on this action is used by the agent to learn a model of the human's reward function. The agent then uses this model to greedily choose actions that it predicts would receive the maximum reward. As the user continues to give feedback during the training, the model continuously improves, and eventually, it can directly give rewards to the agent. This framework thus provides a simple method for the user to shape a reward function by using a simple interace. By changing the way the rewards are given, the agent can also be trained to follow a different policy.

Various function approximators such as regression, nearest neighbours and artificial as well as deep neural networks can be used to learn the reward function. The reward model can also be combined with an environmental reward signal. The actions can be chosen to maximize immediate reward or discounted long term returns. The rewards given by the humans can be binary or otherwise, and the framework can be used in discrete as well as continuous tasks. In a discrete space task, the agent can wait for the human user's feedback before taking the next step. For continuous task spaces, where the agent cannot wait for the human user's rewards, a credit assigner is used in [Knox and Stone (2009)](https://dl.acm.org/doi/abs/10.1145/1597735.1597738).

Go to [Knox's website](https://www.bradknox.net/human-reward/) to read about his entire work on human generated rewards.
Watch [this](https://www.youtube.com/watch?v=xzqCzX5ExZA) video where a team from the MIT Media Lab train a robot to show different behaviours as per the human user's reward. 
