import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.human_response import human_response
import os
import pybullet as p
import pybullet_data
import math
import numpy as np
from numpy import savetxt, loadtxt
import random
from gym_panda.envs.autorewards import autorewards
from rewardmodel_classification import train_model, predict


EPISODE_LEN = 24
THRESHOLD = 0.3


class PandaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # Misc coutners
        self.step_counter = 0
        self.ep_reward = 0
        self.ep_count = 0
        self.train = False
        self.episode = 0
        self.counter = 0
        self.ep_len = []
        self.episode_len = EPISODE_LEN
        self.features = []
        self.h_ = []
        self.success = []
        ## Gym
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(np.array([-1] * 6), np.array([1] * 6))

        # Pybullet
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=50, cameraPitch=-20,
                                     cameraTargetPosition=[0.75, -0.25, 0.2])


    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        orientation = p.getQuaternionFromEuler([0., -math.pi, math.pi / 2.])
        dv = 0.5
        fingers = 0.08
        dx = 0
        dy = 0
        dz = 0
        label = "Didnt Move"
        if action == 1:
            dy = 1 * dv  # Left
            label = "Left"
        if action == 3:
            dy = -1 * dv  # Right
            label = "Right"
        if action == 2:
            dz = -1 * dv  # Down
            label = "Down"
        if action == 4:
            dx = 1 * dv  # Forward
            label = "Forward"
        if action == 0:
            dx = -1 * dv  # Back
            label = "Back"

        currentPose = p.getLinkState(self.pandaUid, 11)
        currentPosition = currentPose[0]
        newPosition = [currentPosition[0] + dx,
                       currentPosition[1] + dy,
                       currentPosition[2] + dz]

        jointPoses = p.calculateInverseKinematics(self.pandaUid, 11, newPosition, orientation)[0:7]

        state_object, _ = p.getBasePositionAndOrientation(self.objectUid)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid, 9)[0], p.getJointState(self.pandaUid, 10)[0])

        err = []
        for i in range(3):
            err.append(round(state_object[i] - state_robot[i], 2))
        dist = round(math.sqrt(err[0] ** 2 + err[1] ** 2 + err[2] ** 2), 2)

        p.setJointMotorControlArray(self.pandaUid, list(range(7)) + [9, 10], p.POSITION_CONTROL,
                                    list(jointPoses) + 2 * [fingers])

        past_state_robot = list(state_robot)

        p.stepSimulation()

        state_object, _ = p.getBasePositionAndOrientation(self.objectUid)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        new_state_robot = list(state_robot)
        for i in range(len(state_robot)):
            new_state_robot[i] = round(state_robot[i], 2)

        new_err = []

        for i in range(3):
            new_err.append(round(state_object[i] - state_robot[i], 2))
        new_dist = round(math.sqrt(new_err[0] ** 2 + new_err[1] ** 2 + new_err[2] ** 2), 2)

        new_feature = past_state_robot + [action] + list(state_object)

        if os.path.exists('/home/turtledan/Projects/pandarl/PandaRL/model_1.h5'):
           reward = predict(np.array(new_feature))
           self.reward = 10  if reward >= 0.5 else -10
        else:
            self.reward = 0

        self.step_counter += 1

        if self.step_counter > EPISODE_LEN:
            done = True
        elif state_robot[2] < 0:
            done = False
            print("CLASH!")
        elif dist < 0.25:
            done = True
            print("SUCCESS!")
        else:
            done = False
        self.observation = new_state_robot + list(state_object)
        return np.array(self.observation).astype(np.float32), self.reward, done, {}


    def reset(self):
        print("------- RESET -------")

        self.ep_count += 1
        self.step_counter = 0
        self.counter = 0
        self.reward = 0

        p.resetSimulation()

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # we will enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        # after we loaded everything
        urdfRootPath = pybullet_data.getDataPath()

        p.setGravity(0, 0, -9.81)

        planeUid = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])

        rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08]
        self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), useFixedBase=True)

        for i in range(7):
            p.resetJointState(self.pandaUid, i, rest_poses[i])

        p.resetJointState(self.pandaUid, 9, 0.08)
        p.resetJointState(self.pandaUid, 10, 0.08)
        tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.65])

        state_object = [random.uniform(0.5, 0.8), random.uniform(-0.25, 0.25), 0.001]

        self.objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=state_object,
                                    useFixedBase=True)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid, 9)[0], p.getJointState(self.pandaUid, 10)[0])
        self.observation = state_robot + (1., 1., 1.)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        return np.array(self.observation).astype(np.float32)

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7, 0, 0.05],
                                                          distance=.7,
                                                          yaw=90,
                                                          pitch=-70,
                                                          roll=0,
                                                          upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=90,
                                                   aspect=float(960) / 720,
                                                   nearVal=0.1,
                                                 farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                            height=720,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720, 960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _get_state(self):
        return self.observation

    def close(self):
        p.disconnect()
