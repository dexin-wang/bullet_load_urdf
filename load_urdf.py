'''
Description: 
Author: wangdx
Date: 2021-09-06 22:09:19
LastEditTime: 2022-03-31 17:15:46
'''

import pybullet as p
import pybullet_data
import os
import sys
sys.path.append(os.curdir)
from simEnv import SimEnv

if __name__ == "__main__":
    cid = p.connect(p.GUI)  # 连接服务器
    # cid = p.connect(p.DIRECT)  # 连接服务器
    env = SimEnv(p) # 初始化虚拟环境类
    env.loadURDF('007_tuna_fish_can/007_tuna_fish_can.urdf')
    while True:
        p.stepSimulation()
