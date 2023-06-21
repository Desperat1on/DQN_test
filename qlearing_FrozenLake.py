import gym
import numpy as np
import matplotlib.pyplot as plt


def q_learning(env, num_episodes, alpha, gamma, epsilon, epsilon_decay, min_epsilon):
    # 初始化Q表，将其所有值设为0
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    # 存储每个回合的奖励
    rewards = []

    # 执行Q-learning算法
    for episode in range(num_episodes):
        # 初始化回合的奖励
        total_reward = 0

        # 重置环境，获取初始状态
        state, info = env.reset()
        # print("state:")
        # print(state)

        # 根据epsilon-greedy策略选择动作
        while True:
            # 以epsilon的概率进行随机探索，以1-epsilon的概率进行利用已学到的最优策略
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            # 执行动作，观察环境返回的下一个状态、奖励和是否终止
            # print(action)
            next_state, reward, done, truncated, info = env.step(action)

            # 更新Q表的值
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

            # 累计回合的奖励
            total_reward += reward

            # 更新当前状态为下一个状态
            state = next_state

            # 如果达到终止状态，结束回合
            if done:
                print('=' * 8)
                break

        # 将回合的奖励添加到列表中
        rewards.append(total_reward)

        # 每隔5个回合减小探索度
        if episode % 5 == 0:
            if epsilon > min_epsilon:
                epsilon *= epsilon_decay

        # 每隔100个回合打印一次累计奖励的平均值
        if episode % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode: {episode}, Average Reward: {avg_reward}")

    return Q, rewards


if __name__ == '__main__':
    # 创建FrozenLake环境
    desc = ["SFFHF", "FFFHF", "HHFFG", "FFFFF", "FFHHH"]
    env = gym.make('FrozenLake-v1', desc=desc, is_slippery=False, render_mode="human")
    # print(env.action_space)
    # print(env.action_space.n)
    # print(env.action_space.sample())
    # print(env.observation_space)
    # print(env.observation_space.n)
    # env.reset()
    # env.render()

    # 创建FrozenLake环境
    # env = gym.make('FrozenLake-v1', desc=desc, is_slippery=False, render_mode="human")

    # 设置超参数
    num_episodes = 200  # 回合数
    alpha = 0.01  # 学习率
    gamma = 0.9  # 折扣因子
    epsilon = 0.8  # 探索率
    epsilon_decay = 0.25
    min_epsilon = 0

    # 运行Q-learning算法
    Q, rewards = q_learning(env, num_episodes, alpha, gamma, epsilon, epsilon_decay, min_epsilon)

    env.close()
    # 绘制学习曲线
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Learning Curve')
    plt.savefig('./output/learning_curve_qlearning_3.png')
    plt.show()
