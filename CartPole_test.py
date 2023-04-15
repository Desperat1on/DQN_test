import gym

env = gym.make('CartPole-v1', render_mode="human")  # v0已过时，指定render_mode为human以显示画面
env.reset()
env.render()