import open3d as o3d
import numpy as np

class Camera():
    def __init__(self, Fx, Fy, Cx, Cy, width, height, intrinsic_matrix):

        self.Fx = Fx
        self.Fy = Fy
        self.Cx = Cx
        self.Cy = Cy
        self.width = width
        self.height = height
        self.intrinsic_matrix = intrinsic_matrix
        self.extrinsic_matrix = np.eye(4)

    def translate(self, tx, ty, tz):
        self.extrinsic_matrix[:3, 3] = [tx, ty, tz]
        # print('Extrinsic matrix:\n',self.extrinsic_matrix)

    def rotate(self, roll, pitch, yaw):

        rotation_x = np.array([[1, 0, 0],
                            [0, np.cos(roll), -np.sin(roll)],
                            [0, np.sin(roll), np.cos(roll)]])

        rotation_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                            [0, 1, 0],
                            [-np.sin(pitch), 0, np.cos(pitch)]])

        rotation_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                            [np.sin(yaw), np.cos(yaw), 0],
                            [0, 0, 1]])

        # Combine rotation matrices
        rotation_matrix = np.dot(np.dot(rotation_z, rotation_y), rotation_x)

        self.extrinsic_matrix[:3, :3] = rotation_matrix

    def raycasting(self):
        return o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix=self.intrinsic_matrix,
            extrinsic_matrix=self.extrinsic_matrix,
            width_px=self.width,
            height_px=self.height
        )