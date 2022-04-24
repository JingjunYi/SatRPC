import numpy as np
import math
from scipy.spatial.transform import Rotation as R


def load_direct_data(direct_pth):
    '''
    load direction angle of pixels 
    input: 
        file path
    output:
        direct_data(np.array(point_number, 2)): direction angle phi_x, phi_y
    '''
    direct_data = np.loadtxt(direct_pth, usecols=(1, 2), skiprows=1)
    
    return direct_data


def load_u_rot(u_rot_pth):
    '''
    load calibration matrix, namely camera to body matrices
    input:
        file path
    output:
        u_rot(np.array(3, 3)): calibration matrix
    '''
    with open(u_rot_pth, 'r') as f:
        lines = f.readlines()
        pitch_phi = float(lines[1].split('=')[1].split(' ')[1])
        roll_omega = float(lines[3].split('=')[1].split(' ')[1])
        yaw_kappa = float(lines[5].split('=')[1].split(' ')[1])
        R_phi = np.array([[math.cos(pitch_phi), 0, -math.sin(pitch_phi)],
                          [0, 1, 0],
                          [math.sin(pitch_phi), 0, math.cos(pitch_phi)]])
        R_omega = np.array([[1, 0, 0],
                            [0, math.cos(roll_omega), -math.sin(roll_omega)],
                            [0, math.sin(roll_omega), math.cos(roll_omega)]])
        R_kappa = np.array([[math.cos(yaw_kappa), -math.sin(yaw_kappa), 0],
                            [math.sin(yaw_kappa), math.cos(yaw_kappa), 0],
                            [0, 0, 1]])
        
        return np.dot(np.dot(R_phi, R_omega), R_kappa)
    
    
def load_j2w_rot(j2w_rot_pth):
    '''
    load J200 to WGS84 rotation matrices from matlab result j2w_rot.txt
    input:
        file path
    output:
        j2w_rot(np.array(time_number, 3, 3)): J200 to WGS84 rotation matrices
    '''
    with open(j2w_rot_pth, 'r') as f:
        lines = f.readlines()
        j2w_rot = []
        for line in lines:
            j2w_rot.append(line.split(' ')[0: -1])
        j2w_rot = np.array(j2w_rot).reshape(-1, 3, 3).astype(np.float)
        return j2w_rot    
    
    
def get_u_vec(direct_data):
    '''
    transform direction angle to visual vector u
    input:
        direct_data(np.array(point_number, 2)): direction angle
    output:
        visual vector(np.array(3, point_number))
    '''
    return np.concatenate([np.tan(direct_data[:, 1]).reshape(1, len(direct_data)),
                           np.tan(direct_data[:, 0]).reshape(1, len(direct_data)),
                           -np.ones((1, len(direct_data)))])
    

def get_m_factor(final_u_vec, gps_p, height):
    '''
    Obtain the scale factor for the final coordinate conversion
    input:
        final_u_vec(np.array(3, 1)): rotated visual vector
        gps_p(np.array(3, 1)): gps position of ZY3
        height(float): elevation of grid point
    output:
        m(float): scale factor m = u * f 
    '''
    # WGS84 ellipsoid parameters
    A = 6378137 + height
    B = 6356752.3142 + height
    # construct and solve quadratic function, coefficients a b c
    a = (final_u_vec[0] * final_u_vec[0] + final_u_vec[1] * final_u_vec[1]) / (A * A) + final_u_vec[2] * final_u_vec[2] / (B * B)
    b = 2 * ((final_u_vec[0] * gps_p[0] + final_u_vec[1] * gps_p[1]) / (A * A) + final_u_vec[2] * gps_p[2]/(B*B))
    c = (gps_p[0] * gps_p[0] + gps_p[1] * gps_p[1]) / (A * A) + gps_p[2] * gps_p[2] / (B * B) - 1
    delta = b * b - 4 * a * c
    if a == 0:
        if b != 0:
            x = -c / b
            return x
        else:
            print('No solution for scale factor m, assume m as 0.')
            return 0
    else:
        if delta < 0:
            print('No solution for scale factor m, assume m as 0.')
            return 0
        elif delta == 0:
            x = -b / (2 * a)
            return x
        else:
            x1 = (-b + math.sqrt(delta)) / (2 * a)
            x2 = (-b - math.sqrt(delta)) / (2 * a)
            return max(x1, x2)            
        

    
def get_wgs84_vec(u_vec, u_rot, b2j_rot, j2w_rot, gps_p, height):
    '''
    transform visual vector to coordinate vectors of grid points in WGS84
    input:
        u_vec(np.array(3, 1)): visual vector
        u_rot(np.array(3, 3)): camera to body rotation matrix
        b2j_rot(np.array(3, 3)): body to J200(orbit) rotation matrix
        j2w_rot(np.array(3, 3)): J2000 to WGS84 rotation matrix
        gps_p(np.array(1, 3)): gps position of ZY3
        height(float): elevation of grid point
    output:
        wgs84_vec(np.array(3, 1)): coordinate vector in WGS84
    '''
    u_vec = u_vec.reshape(-1, 1)
    gps_p = gps_p.reshape(-1, 1)
    inter_u_vec = np.dot(u_rot, u_vec)
    inter_u_vec = np.dot(b2j_rot, inter_u_vec)
    final_u_vec = np.dot(j2w_rot, inter_u_vec)
    m = get_m_factor(final_u_vec, gps_p, height) 
    wgs84_vec = gps_p + final_u_vec * m
    
    return wgs84_vec
    

def wgs2blh(ground_points):
    '''
    transform WGS84 coordinates to BLH coordinates
    input:
        ground_points: WGS coordinates
    output:
        BLH: BLH coordinates
    '''
    a = 6378137
    b = 6356752.314
    pi= 3.1415926
    e2 = (a * a - b * b) / (a * a)
    L = np.arctan(ground_points[:, 1] / ground_points[:, 0])
    B = np.arctan(ground_points[:, 2] / np.sqrt(ground_points[:, 0] ** 2 + ground_points[:, 1] ** 2))
    
    for i in range(0, 100):
        N = a / np.sqrt(1 - e2 * (np.sin(B) ** 2))
        H = ground_points[:, 2] / np.sin(B) - N * (1 - e2)

        Bn = np.arctan(ground_points[:, 2] * (N + H) / ((N * (1 - e2) + H) * np.sqrt(ground_points[:, 0] ** 2 + ground_points[:, 1] ** 2)))
        
        if np.max(np.abs(B - Bn)) < 1e-7:
            break
        B = Bn

    BLH = np.zeros((len(B), 3))
    B = B / pi * 180
    L = L / pi * 180
    B[B < 0] += 180
    L[L < 0] += 180
    BLH[:, 0] = B
    BLH[:, 1] = L
    BLH[:, 2] = H

    return BLH


def blh2wgs(BLH):
    '''
    transform BLH coordinates to WGS84 coordinates
    input:
        BLH: BLH coordinates
    output:
        ground_points: WGS coordinates
    '''
    a = 6378137.0000
    b = 6356752.3141
    pi = 3.1415926
    e2 = 1 - (b / a) ** 2
    
    B = BLH[:, 0]
    L = BLH[:, 1]
    H = BLH[:, 2]
    
    L = L * pi / 180
    B = B * pi / 180
    
    N = a / np.sqrt(1 - e2 * np.sin(B) ** 2)
    x = (N + H) * np.cos(B) * np.cos(L)
    y = (N + H) * np.cos(B) * np.sin(L)
    z = (N * (1 - e2) + H) * np.sin(B)
    return np.vstack((x, y, z)).transpose()



if __name__ == '__main__':
    direct_pth = 'E:/Code/SatPhotogrammetry/data/data_ZY3/NAD.cbr'
    j2w_rot_pth = 'E:/Code/SatPhotogrammetry/output/inter/j2w_rot.txt'
    direct_data = load_direct_data(direct_pth)
    u = get_u_vec(direct_data)
    # print(u[:, 0])
    j2w_rot = load_j2w_rot(j2w_rot_pth)
    # print(j2w_rot.shape)
    # print(j2w_rot)
    