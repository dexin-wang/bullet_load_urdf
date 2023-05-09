import pybullet as p
import pybullet_data
import time
import math
import os
import glob
import random
import numpy as np


class SimEnv(object):
    """
    虚拟环境类
    """
    def __init__(self, bullet_client, model_list=None, plane=0, load_tray=True):
        """
        path: 模型路径
        """
        self.p = bullet_client
        self.p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, lightPosition=[-4, 4, 10])    # 不显示控件
        self.p.setPhysicsEngineParameter(enableFileCaching=0)   # 不压缩加载的文件
        # self.p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=38, cameraPitch=-22, cameraTargetPosition=[0, 0, 0])
        self.p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0, 0, 0.2])
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.setGravity(0, 0, -10) # 设置重力

        # 读取urdf列表
        self.urdfs_list = []
        if model_list is not None:
            with open(model_list, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if not line:
                        break
                    self.urdfs_list.append(line.strip('\n'))

        self.urdfs_id_bullet = []   # 存储由pybullet系统生成的模型id
        self.urdfs_id_list = []     # 存储模型在model_list文件列表中的索引
        self.urdfs_scale = []       # 记录模型的尺度
        self.urdfs_vis_xyz = []       # 记录模型的偏移坐标，在urdf中设置
        self.start_idx = 0
        self.EulerRPList = [[0, 0], [math.pi/2, 0], [-1*math.pi/2, 0], [math.pi, 0], [0, math.pi/2], [0, -1*math.pi/2]]
        
        self.trayId = None
        self.planeId = None
        self.camera = None
        if load_tray:
            self.loadTray()     # 加载托盘
        self.loadPlane(plane=plane)    # 加载地面


    def calViewMatrix(self, position, qua):
        '''相机视图矩阵

        position: (xyz)
        euler: (rpy)
        
        视图矩阵(ViewMatrix)，是右手系坐标系转换为左手系， 传递给OpenGL进行渲染。
        X -> Y
        Y -> -X
        Z -> Z 
        '''
        cameraEyePosition = position
        cameraQuaternion = qua
        # 四元数转旋转矩阵 3x3矩阵
        cameraRotationMatrix = self.p.getMatrixFromQuaternion(cameraQuaternion)
        cameraRotationMatrix = np.array(cameraRotationMatrix).reshape((3,3))
        # camera_focus_distance 相机对焦距离
        # 指的是在当前的姿态角下, 相机要对准摄像头前方多远距离的物体.
        # 摄像头镜头指向的方向为Z轴方向。
        
        # 计算摄像头对焦点的坐标
        # T_world2cam = [position, quaternion]
        # T_cam2target : cam是相机坐标系, target是相机拍摄点的目标坐标系
        # 与相机坐标系， 只存在Z轴方向上的平移关系, 平移距离为camera_focus_distance
        # T_world2target = T_world2cam * T_cam2target
        # pybullet.multiplyTransforms 返回的T_world2target 格式 (位置, 四元数)
        self.camera_focus_distance = 0.1
        cameraTargetTransform = self.p.multiplyTransforms(positionA=cameraEyePosition, orientationA=cameraQuaternion, positionB=[0, 0, self.camera_focus_distance], orientationB=[0,0,0,1])
        # cameraTargetTransform = pybullet.multiplyTransforms(positionA=cameraEyePosition, orientationA=cameraQuaternion, positionB=[0, 0,self.camera_focus_distance], orientationB=[0,0,0,1])        
        
        # cameraTargetPosition =[0, 0, 0] # 指向坐标系原点
        cameraTargetPosition = cameraTargetTransform[0]
        # 相机坐标系Y轴正方向, 单位向量
        cameraUpVector = cameraRotationMatrix[:, 1] * -1
        
        # viewMatrix:  我理解的视图矩阵，是相机在世界坐标系下的位姿？
        # cameraEyePosition: 相机在世界坐标系下位置
        # cameraTargetPosition: 希望相机正前方， 即Z轴朝向的位置(应该设置为Z轴的坐标)
        # cameraUpVector:  相机坐标系Y轴正方向.
        # v_z 与相机坐标系Z轴正方向同向
        # v_z = cameraTargetPosition - cameraEyePosition
        # u_y = 相机坐标系在y轴正方向在世界坐标系下的描述
        # u_y = cameraUpVector 
        # cameraTargetPosition 与cameraUpVector
        viewMatrix = self.p.computeViewMatrix(cameraEyePosition=cameraEyePosition,
                                                    cameraTargetPosition=cameraTargetPosition,
                                                    cameraUpVector=cameraUpVector)
        return viewMatrix

    def urdfs_num(self):
        """
        返回环境中已加载的物体个数(包括托盘和地面)
        return: int
        """
        return len(self.urdfs_id_bullet)
    
    def urdfs_obj_num(self):
        """
        返回环境中已加载的物体个数(不包括托盘和地面)
        return: int
        """
        return len(self.urdfs_id_bullet) - self.start_idx
    
    def urdf_list_num(self):
        """
        返回模型列表中模型的数量
        """
        return len(self.urdfs_list)

    def urdfs_path(self):
        """
        返回环境中已加载的物体的路径
        return: list
        """
        paths = []
        for idx in self.urdfs_id_list:
            paths.append(self.urdfs_list[idx])
        return paths

    def sleep(self, n=100):
        t = 0
        while True:
            self.p.stepSimulation()
            t += 1
            if t == n:
                break

    def loadTray(self):
        """
        加载托盘
        """
        path = 'models/tray/tray.urdf'
        self.trayId = self.p.loadURDF(path, [0, 0, 0])
        # 记录信息
        self.urdfs_id_list += [-1]
        self.urdfs_id_bullet.append(self.trayId)
        inf = self.p.getVisualShapeData(self.trayId)[0]
        self.urdfs_scale.append(inf[3][0]) 
        self.urdfs_vis_xyz.append(inf[5])
        self.start_idx += 1
    
    def loadPlane(self, plane=0):
        """
        加载地面
        plane: 0:蓝白网格地面，1:灰黄色地面
        """
        if plane == 0:
            path = 'plane.urdf'               # 蓝白格地面路径：D:\developers\anaconda3-5.2.0\envs\sim_grasp\Lib\site-packages\pybullet_data
        elif plane == 1:
            path = 'models/plane/plane.urdf'    # 橙白地面，生成数据集用
        elif plane < 0:
            return

        if path == 'plane.urdf':
            self.planeId = self.p.loadURDF(path, [0, 0, 0])
        else:
            self.planeId = self.p.loadURDF(path, [0, 0, 0], flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        # 记录信息
        self.urdfs_id_list += [-2]
        self.urdfs_id_bullet.append(self.planeId)
        inf = self.p.getVisualShapeData(self.planeId)[0]
        self.urdfs_scale.append(inf[3][0]) 
        self.urdfs_vis_xyz.append(inf[5])
        self.start_idx += 1

    
    def loadURDF(self, urdf_file:str, idx=-1, pos=None, euler=None, scale=1):
        """
        加载单个urdf模型
        当idx为负数时，加载的物体不计入模型列表

        urdf_file: urdf文件
        idx: 物体id， 等于-1时，采用file；否则加载模型列表中索引为idx的模型
        pos: 加载位置，如果为None，则随机位置
        euler: 加载欧拉角，如果为None，则随机欧拉角
        scale: 缩放倍数
        """
        # 获取物体文件
        if idx >= 0:
            urdf_file = self.urdfs_list[idx]
            self.urdfs_id_list += [idx]

        # 位置
        if pos is None:
            _p = 0.1
            pos = [random.uniform(-1 * _p, _p), random.uniform(-1 * _p, _p), random.uniform(0.1, 0.4)]

        # 方向
        if euler is None:
            euler = [random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi)]
        Ori = self.p.getQuaternionFromEuler(euler)

        # 加载物体
        flags = p.URDF_USE_IMPLICIT_CYLINDER
        if idx >= 0:
            mtl_files = glob.glob( os.path.join( os.path.dirname(self.urdfs_list[idx]), '*.mtl') )
        else:
            mtl_files = []
        if len(mtl_files) > 0:
            flags = flags | p.URDF_USE_MATERIAL_COLORS_FROM_MTL
        urdf_id = self.p.loadURDF(urdf_file, pos, Ori, globalScaling=scale, flags=flags)
        print('urdf = ', urdf_file)

        if idx >= 0:
            # 记录信息
            self.urdfs_id_bullet.append(urdf_id)
            inf = self.p.getVisualShapeData(urdf_id)[0]
            self.urdfs_scale.append(inf[3][0]) 
            self.urdfs_vis_xyz.append(inf[5]) 
    