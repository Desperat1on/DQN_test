"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -100].
Yellow bin circle:      paradise    [reward = +100].
All other states:       ground      [reward = -1].
This script is the environment part of this example. The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import time
import sys

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 100  # 单元长宽
MAZE_H = 5  # 格子长度数目
MAZE_W = 5  # 格子宽度数目


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))  # 窗口大小
        self._build_maze()
        self.n_features = 4
        self.zuobiao = [50, 150, 250, 350, 450]

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',  # 画布
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)
        # 网格
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)
        # create origin
        origin = np.array([50, 50])
        # 陷阱
        hell1_center = origin + np.array([UNIT * 3, 0])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 40, hell1_center[1] - 40,
            hell1_center[0] + 40, hell1_center[1] + 40,
            fill='black')
        # hell
        hell2_center = origin + np.array([UNIT * 3, UNIT])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 40, hell2_center[1] - 40,
            hell2_center[0] + 40, hell2_center[1] + 40,
            fill='black')
        # hell
        hell3_center = origin + np.array([0, UNIT * 2])
        self.hell3 = self.canvas.create_rectangle(
            hell3_center[0] - 40, hell3_center[1] - 40,
            hell3_center[0] + 40, hell3_center[1] + 40,
            fill='black')
        # hell
        hell4_center = origin + np.array([UNIT, UNIT * 2])
        self.hell4 = self.canvas.create_rectangle(
            hell4_center[0] - 40, hell4_center[1] - 40,
            hell4_center[0] + 40, hell4_center[1] + 40,
            fill='black')
        # hell
        hell5_center = origin + np.array([UNIT * 2, UNIT * 4])
        self.hell5 = self.canvas.create_rectangle(
            hell5_center[0] - 40, hell5_center[1] - 40,
            hell5_center[0] + 40, hell5_center[1] + 40,
            fill='black')
        # hell
        hell6_center = origin + np.array([UNIT * 3, UNIT * 4])
        self.hell6 = self.canvas.create_rectangle(
            hell6_center[0] - 40, hell6_center[1] - 40,
            hell6_center[0] + 40, hell6_center[1] + 40,
            fill='black')
        # hell
        hell7_center = origin + np.array([UNIT * 4, UNIT * 4])
        self.hell7 = self.canvas.create_rectangle(
            hell7_center[0] - 40, hell7_center[1] - 40,
            hell7_center[0] + 40, hell7_center[1] + 40,
            fill='black')
        # 奖励点
        oval_center = origin + np.array([UNIT * 4, UNIT * 2])
        self.oval = self.canvas.create_oval(
            oval_center[0] - 40, oval_center[1] - 40,
            oval_center[0] + 40, oval_center[1] + 40,
            fill='yellow')
        # create 玩家
        self.rect = self.canvas.create_rectangle(
            origin[0] - 40, origin[1] - 40,
            origin[0] + 40, origin[1] + 40,
            fill='red')
        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.rect)

        origin = np.array([50, 50])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 40, origin[1] - 40,
            origin[0] + 40, origin[1] + 40,
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)
        # return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (
        #             MAZE_H * UNIT)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
        self.canvas.move(self.rect, base_action[0], base_action[1])  # 移动玩家
        s_ = self.canvas.coords(self.rect)  # 下一个状态
        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
            print("成功")
        # 是否掉入黑色
        elif s_ in [self.canvas.coords(self.hell1),  # [310,10,390,90]
                    self.canvas.coords(self.hell2),  # [310,110,390,190]
                    self.canvas.coords(self.hell3),  # [10,210,90,290]
                    self.canvas.coords(self.hell4),  # [110,210,90,190]
                    self.canvas.coords(self.hell5),  # [210,410,290,490]
                    self.canvas.coords(self.hell6),  # [310,410,390,490]
                    self.canvas.coords(self.hell7)]:  # [410,410,490,490]
            reward = -1  # reward
            done = True
            print("失败")
        else:
            reward = 0
            done = False
            # print("继续")
        s_ = self.canvas.coords(self.rect)
        return s_, reward, done

    def render(self):

        # time.sleep(0.1)
        self.update()

# def update():
#     for t in range(10):
#         s = env.reset()
#         while True:
#             env.render()
#             a = 1
#             s, r, done = env.step(a)
#             if done:
#                 break

# if __name__ == '__main__':
#     env = Maze()
#     env.after(100, update)
#     env.mainloop()
