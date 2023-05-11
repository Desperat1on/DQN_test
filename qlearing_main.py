from maze_env1 import Maze
from rl import QLearningTable, SarsaTable


def update():
    for episode in range(20):
        observation = env.reset()
        while True:
            env.render()
            action = RL.choose_action(str(observation))
            # print("action:{0}".format(action))
            observation_, reward, done = env.step(action)
            # (s, a, r, s, a) == Sarsa
            RL.learn(str(observation), action, reward, str(observation_))
            observation = observation_
            if done:
                print("episode: %d" % episode)
                break
    print('game over')
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    # RL = SarsaTable(actions=list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()
