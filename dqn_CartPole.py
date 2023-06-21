import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# env = gym.make("CartPole-v1", render_mode="human")  # 创建 CartPole 游戏环境
env = gym.make("CartPole-v1")  # 创建 CartPole 游戏环境

plt.ion()  # 打开交互式绘图模式

# 如果可以使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备为 GPU（如果可用）或 CPU

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state',
                         'reward'))  # 定义一个名为 Transition 的命名元组，包含了状态（state）、动作（action）、下一个状态（next_state）和奖励（reward）这四个属性


# 定义经验回放
class ReplayMemory(object):

    # 初始化方法，设置回放内存的容量
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)  # 使用 deque 数据结构来存储回放内存的内容，并设置最大长度为 capacity
        # deque（双端队列）是一种具有队列和栈性质的数据结构，它可以在两端进行元素的插入和删除操作
        # 可以指定队列的最大长度，超过最大长度时，新插入的元素会将最老的元素删除。

    # 保存经验
    def push(self, *args):
        self.memory.append(Transition(*args))  # 将传入的参数转换为 Transition 对象，并将其添加到经验回放内存中

    # 随机采样方法，从回放内存中随机选择一批数据进行训练
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)  # 使用 random.sample 方法从回放内存中随机选择 batch_size 个样本，并返回选择的样本列表

    # 获取回放内存的长度
    def __len__(self):
        return len(self.memory)  # 返回回放内存中存储的样本数量

# 定义网络结构
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # 被用来确定下一步动作的函数，可以输入一个元素以确定下一个动作，或者输入一个批次用于优化。返回一个 tensor([[left0exp,right0exp]...])。
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


BATCH_SIZE = 128  # BATCH_SIZE 是从经验回放区中采样的经验数量
GAMMA = 0.99  # GAMMA 是折扣因子，控制agent是否具有远见能力，0则表示只关注当前奖励
EPS_START = 0.9  # EPS_START 是 epsilon 的初始值
EPS_END = 0.05  # EPS_END 是 epsilon 的最终值
EPS_DECAY = 1000  # EPS_DECAY 控制 epsilon 的指数衰减速率，数值越大衰减越慢
TAU = 0.005  # TAU 是目标网络的更新速率
LR = 1e-4  # LR 是 AdamW 优化器的学习率

# 从 gym 的动作空间获取动作数量
n_actions = env.action_space.n
# 获取状态空间的状态数量
state, info = env.reset()
n_observations = len(state)

# 创建策略网络和目标网络
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

# 创建 AdamW 优化器
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
# 创建经验回放区
memory = ReplayMemory(6000)

# 用于记录总步数的变量
steps_done = 0

# 定义根据状态选择下一个动作
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) 将返回每行的最大列值。
            # max 结果的第二列是最大元素的索引，因此我们选择具有较大期望奖励的动作。
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


# 用于记录每个回合的持续时间
episode_durations = []

# 绘制持续时间图表
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # 计算每100个回合的平均持续时间并绘制
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.savefig('./output/learning_curve_DQN_CartPole_5.png')

    # plt.pause(0.001)  # 暂停一段时间以更新图形

# 定义模型优化
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # 转置批次数据 (see https://stackoverflow.com/a/19343/3343043 for detailed explanation).
    # 这将批次数据数组转换为具有批次数组的 Transition。
    batch = Transition(*zip(*transitions))
    # original = [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
    # result = (['a', 'b', 'c', 'd'], [1, 2, 3, 4])

    # 计算非终止状态的标识并连接批次元素
    # （终止状态是指模拟结束后的状态）
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # 计算 Q(s_t, a) - 模型计算 Q(s_t)，然后选择采取的动作的列。
    # 这些动作是根据 policy_net 对每个批次状态计算的
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 计算所有下一个状态的 V(s_{t+1})。
    # 对于非终止状态，基于“旧”目标网络计算预期动作值的期望值，选择最大奖励的动作（max(1)[0]）。
    # 根据掩码进行合并，这样我们将在状态是终止状态时，要么得到预期的状态值，要么得到 0。
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # 计算预期的 Q 值
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # 计算 Huber 损失
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # 优化模型
    optimizer.zero_grad()
    loss.backward()
    # 梯度裁剪防止梯度爆炸
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available():
    num_episodes = 1000
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # 初始化环境并获取其状态
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        # 如果终止，则将下一个状态设置为 None；否则将下一个状态转换为张量并添加一个维度。
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # 将经验存储在经验回放区中
        memory.push(state, action, next_state, reward)

        # 状态变为下一个状态
        state = next_state

        # 进行一步优化（在策略网络上）
        optimize_model()

        # 更新目标网络的权重（软更新）
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
