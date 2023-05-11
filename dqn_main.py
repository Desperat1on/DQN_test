from maze_env2 import Maze
from dqn import DeepQNetwork
import time


def run_maze():
    step = 0  # 记录步数，用来提示学习的时间
    for episode in range(1000):
        # # 初始化环境
        # print("episode: %d" % episode)
        observation = env.reset()
        # print("observation: {0}".format(observation))
        # observation=list(observation)
        while True:
            env.render()  # 渲染一帧环境画面
            # print("observation:{0}".format(observation))
            action = RL.choose_action(observation)  # DQN根据当前状态s选择行为a
            # print("action:{0}".format(action))
            observation_, reward, done = env.step(action)  # 与环境进行交互，获得下一状态s'、奖励R和是否到达终态
            # print("observation_:{0}".format(observation_))
            RL.store_transition(observation, action, reward, observation_)  # 将当前的采样序列存储到RF中（s, a, R, s'）
            # 200步之后开始学习，每隔5步学习一次，更新Q网络参数（第一个网络）
            if (step > 200) and (step % 5 == 0):
                RL.learn()
            observation = observation_  # 转移至下一状态
            if done:  # 如果终止, 就跳出循环
                print("eposide：%d " % episode)
                # print("回报为：{0}".format(R))
                break
            step += 1  # 总步数 + 1

    # 游戏结束
    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()  # 创建环境
    # sleeptime = 0.5
    # terminate_states = env.env.getTerminate_states()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,  # 每 200 步替换一次 target_net 的参数
                      memory_size=2000,  # 记忆上限
                      # output_graph=True   # 是否输出 tensorboard 文件
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()  # 神经网络的误差曲线
