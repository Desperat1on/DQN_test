import gym
import numpy as np
import matplotlib.pyplot as plt


def run_sarsa(env, num_episodes, alpha, gamma, epsilon):
    # 初始化 Q 表格为全零矩阵
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    # 用于可视化的每个回合的奖励列表
    rewards_per_episode = []

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        # 使用 ε-贪心策略选择动作
        epsilon = epsilon / (episode+1)
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        while not done:
            # 执行动作，观察下一个状态和奖励
            next_state, reward, done, truncated, info = env.step(action)

            # 使用 ε-贪心策略选择下一个动作
            if np.random.uniform(0, 1) < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state, :])

            # 使用 SARSA 更新规则更新 Q 表格
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

            state = next_state
            action = next_action
            total_reward += reward

        rewards_per_episode.append(total_reward)

    return Q, rewards_per_episode


if __name__ == '__main__':
    desc = ["SFFHF", "FFFHF", "HHFFG", "FFFFF", "FFHHH"]
    # 创建 FrozenLake 游戏环境
    env = gym.make('FrozenLake-v1', desc=desc, is_slippery=False, render_mode="human")

    # 设置超参数
    num_episodes = 25  # 回合数
    alpha = 0.8  # 学习率
    gamma = 0.95  # 折扣因子
    epsilon = 0.1  # 探索率

    # 运行 SARSA 算法
    Q, rewards = run_sarsa(env, num_episodes, alpha, gamma, epsilon)
    env.close()
    # 绘制每个回合的奖励图表
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Learning Curve')
    # 保存图表为PNG文件
    plt.savefig('./output/learning_curve_sarsa_1.png')
    plt.show()
