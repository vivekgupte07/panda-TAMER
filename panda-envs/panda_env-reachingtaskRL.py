import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random


EPISODE_LEN = 1000
THRESHOLD = 0.3
class PandaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # Misc coutners
        self.step_counter = 0
        self.reward = 0
        self.episode = 0
        self.t = 0
        self.counter = 0
        self.threshold= THRESHOLD
        self.episode_len = EPISODE_LEN
        ## Gym
        self.action_space = spaces.Box(np.array([-1] * 4), np.array([1] * 4))
        self.observation_space = spaces.Box(np.array([-1] * 6), np.array([1] * 6))

        # Pybullet
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        orientation = p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])
        dv = 0.05
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        fingers = 0.08 # action[3]

        currentPose = p.getLinkState(self.pandaUid, 11)
        currentPosition = currentPose[0]

        newPosition = [currentPosition[0] + dx,
                       currentPosition[1] + dy,
                       currentPosition[2] + dz]
        if newPosition[2]< 0.01:
            newPosition[2] = 0.01
        jointPoses = p.calculateInverseKinematics(self.pandaUid,11,newPosition, orientation)[0:7]

        p.setJointMotorControlArray(self.pandaUid, list(range(7))+[9,10], p.POSITION_CONTROL, list(jointPoses)+2*[fingers])

        p.stepSimulation()

        state_object, _ = p.getBasePositionAndOrientation(self.objectUid)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])

        past_err = [abs(state_object[0] - currentPosition[0]), abs(state_object[1] - currentPosition[1]), abs(state_object[2] - currentPosition[2])]

        err = [abs(newPosition[0]-state_object[0]), abs(newPosition[1]-state_object[1]), abs(newPosition[2]-state_object[2])]

        self.step_counter += 1
        self.reward -= 1

        ### Reward Function ####
        for f in range(3):
            if err[f] < past_err[f]:
                self.reward += 1
            else:
                self.reward += -1

        if err[0] < self.threshold and err[1] < self.threshold and err[2] < self.threshold:
            self.t += 1
            self.reward +=5
            done = False
            if self.t > 5:
                self.threshold += -0.05
                done = True if self.threshold < 0.05 else False
        else:
            self.t = 0
            done = False

        if err[0] > (0.3 + self.threshold) or err[1] > (0.3 + self.threshold) or err[2] > (0.3 + self.threshold):
            self.reward -=  5
            done = False

        '''if newPosition[2] < 0.01:
            self.counter += 1
            self.reward -= 5
            if self.counter > 5:
                self.reward -= 15
                self.counter = 0
                print("!!!!!!!Hit the table!!!!!!")'''
        ##### REWARDS DONE #####

        if self.step_counter > EPISODE_LEN:
           done = True

        # print(self.reward)

        self.observation = state_robot +  (err[0], err[1], err[2])
        return np.array(self.observation).astype(np.float32), self.reward, done, {}

    def reset(self):
        print("0000000Reset0000000")
        self.step_counter = 0
        self.counter = 0
        self.t = 0
        self.counter = 0
        self.reward = 0
        self.threshold = THRESHOLD

        p.resetSimulation()

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything
        urdfRootPath=pybullet_data.getDataPath()

        p.setGravity(0,0,-10)

        planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,-0.65])

        rest_poses = [0,-0.215,0,-2.57,0,2.356,2.356,0.08,0.08]
        self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),useFixedBase=True)

        for i in range(7):
            p.resetJointState(self.pandaUid,i, rest_poses[i])

        p.resetJointState(self.pandaUid, 9, 0.08)
        p.resetJointState(self.pandaUid,10, 0.08)

        tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])

        state_object= [random.uniform(0.5,0.8),random.uniform(-0.2,0.2),0.001]
        self.objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=state_object, useFixedBase=True)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
        self.observation = state_robot + (state_object[0], state_object[1], state_object[2])
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        return np.array(self.observation).astype(np.float32)

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
                                                            distance=.7,
                                                            yaw=90,
                                                            pitch=-70,
                                                            roll=0,
                                                            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                              height=720,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _get_state(self):
        return self.observation

    def close(self):
        p.disconnect()
