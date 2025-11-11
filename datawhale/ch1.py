import numpy as np
import gymnasium as gym

# For Codespaces/no-GUI use render_mode=None or "rgb_array"
env = gym.make('MountainCar-v0', render_mode=None)
print('观测空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('观测范围 = {} ~ {}'.format(env.observation_space.low, env.observation_space.high))
print('动作数 = {}'.format(env.action_space.n))

class SimpleAgent:
    def __init__(self, env):
        pass

    def decide(self, observation):  # 决策
        position, velocity = observation  # observation 是形如 [pos, vel] 的 ndarray
        # 下面这些都是标量运算，保持不变
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03,
                 0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2  # push right
        else:
            action = 0  # push left
        return action

    def learn(self, *args):
        pass

agent = SimpleAgent(env)

def play(env, agent, render=False, train=False, seed=None, max_steps=10000):
    total_reward = 0.0
    obs, info = env.reset(seed=seed)

    for _ in range(max_steps):
        if render and env.render_mode == "human":
            env.render()

        action = agent.decide(obs)
        # Gymnasium: step returns 5 values
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if train:
            agent.learn(obs, action, reward, terminated or truncated)

        if terminated or truncated:
            break

    return total_reward

# 推荐的播种方式
env.action_space.seed(3)          # 可选：给动作空间也播种
episode_reward = play(env, agent, render=False, seed=3)
print('回合奖励 = {}'.format(episode_reward))

# 多回合评估
episode_rewards = [play(env, agent, render=False) for _ in range(100)]
print('平均回合奖励 = {}'.format(np.mean(episode_rewards)))

env.close()
