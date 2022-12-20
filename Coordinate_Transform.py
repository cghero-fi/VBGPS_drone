import numpy as np
import cv2
import quaternion
import math
from numpy.linalg import inv

old_coor = np.array([
    [1.0, 0.0, 0.0]
    ,[0.0, 0.45, 0.0]
    ,[0.0, 0.0, 1.0]
])


class Pose():
    def __init__(self, pose_txt):
        _tmp = pose_txt.split(' ')
        self.id = str(_tmp[0])
        self.qw = float(_tmp[1])
        self.qx = float(_tmp[2])
        self.qy = float(_tmp[3])
        self.qz = float(_tmp[4])
        self.x = float(_tmp[5])
        self.y = float(_tmp[6])
        self.z = float(_tmp[7])

    def abs_pos(self):
        return np.dot(-self.abs_rot_mtx(), self.pos())

    def pos(self):
        return np.array([self.x, self.y, self.z])

    def abs_rot_mtx(self):
        return self.rot_mtx().T

    def rot_mtx(self):
        return quaternion.as_rotation_matrix(self.quat())

    def quat(self):
        return np.quaternion(self.qw, self.qx, self.qy, self.qz)

    def tmp(self):
        return quaternion.from_rotation_matrix(quaternion.as_rotation_matrix(self.quat()).T)

def Get_M(new_coor, new_original):                 #  M * X + t = X'
    old_mul_M = new_coor - new_original            #  M * X = X' - t
    M = np.dot(old_mul_M.T, inv(old_coor.T))       #  M = (X' - t) * X^-1
    return M

def Trans_coor(point, new_original, M):                       
    float_formatter = "{:.6f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    return (np.matmul((inv(M)), (point - new_original).T)) # (Point' - t) * M^-1 = Point

def main():
    # Load traj
    sfm_abs_fname = 'Coordinate_demo.txt'
    f = open(sfm_abs_fname, 'r')
    lines = f.readlines()
    cv_file = cv2.FileStorage("./init.xml", cv2.FILE_STORAGE_WRITE)

    cam_n = []
    cam_position = {}
    cam_trans = []

    cam_dirt = {}

    for line in lines:
        if line[0] == '#':
            continue
        p = Pose(line)
        print(p.id)
        # print(p.abs_pos())
        print(p.abs_rot_mtx())
        print()
        if p.id == "original":
            cv_file.write( p.id + "_rot_mat", p.abs_rot_mtx())
        cam_position[p.id] = p.abs_pos()

    new_coor = np.empty((3,3)) 
    new_coor[0] = cam_position['right']
    new_coor[1] = cam_position['up']
    new_coor[2] = cam_position['forward']

    M = Get_M(new_coor, cam_position['original'])
    cv_file.write( "new_original", cam_position['original'])
    cv_file.write( "M_rot", M)

    cv_file.release()

if __name__ == '__main__':
    main()