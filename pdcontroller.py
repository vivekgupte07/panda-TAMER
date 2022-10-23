
import gym
import gym_panda

env = gym.make("panda-v0")

done = False
err = 0.01
fingers = 1

info = {'object_position': [0.7, 0.0, 0.1]}

k_p = 10
k_d = 1
dt = 1./240
t = 0
dx = 0
dy = 0
dz = 0

for i_ep in range(20):
    obs = env.reset()
    fingers = 1

    for t in range(100):
        env.render()
        print(obs)
        print(' ')

        dx = info['object_position'][0] - obs[0]
        dy = info['object_position'][1] - obs[1]

        target_z = info['object_position'][2]

        if abs(dx) < err and abs(dy) < err and abs(dz) < err:
            fingers = 0
        if obs[3]+obs[4] < err+0.02 and fingers == 0:
            target_z = 0.5
        dz = target_z-obs[2]
        pd_x = k_p * dx + k_d * dx / dt
        pd_y = k_p * dy + k_d * dy / dt
        pd_z = k_p * dz + k_d * dz / dt

        action = [pd_x, pd_y, pd_z, fingers]
        obs, reward, done, info = env.step(action)

        if done:
            print(f"Episode finished after {t+1} steps")
            break
env.close()
