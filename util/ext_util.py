import numpy as np
from matplotlib import pyplot as plt
        
        
def load_att(att_pth):
    '''
    load camera pose data from att_pth
    input: 
        file path
    ouput: 
        att_data(np.array((group_number, 4))): q1, q2, q3, q4
        att_time(np.array((group_number, 1))): timecode
    '''
    with open(att_pth, 'r') as f:
        lines = f.readlines()
        group_number = int(lines[6].split('=')[1].split(' ')[1])
        data = np.zeros((group_number, 4))
        time = np.zeros((group_number, 1))
        for i in range(group_number):
            time[i] = lines[i * 15 + 9].split('=')[1].split(' ')[1]
            for j in range(4):
                data[i][j] = lines[i * 15 + 17 + j].split('=')[1].split(' ')[1]
        
    return data, time
    
    
def load_gps(gps_pth):
    '''
    load gps imformation from gps_pth
    input: 
        file path
    output:
        gps_data(np.array((group_number, 6))): px, py, pz, vx, vy, vz
        gps_time(np.array((group_number, 1))): timecode
    '''
    with open(gps_pth, 'r') as f:
        lines = f.readlines()
        group_number = int(lines[4].split('=')[1].split(' ')[1])
        data = np.zeros((group_number, 6))
        time = np.zeros((group_number, 1))
        for i in range(group_number):
            time[i] = lines[i * 11 + 7].split('=')[1].split(' ')[1]
            for j in range(6):
                data[i][j] = lines[i * 11 + 9 + j].split('=')[1].split(' ')[1]
                    
    return data, time
    
    
def load_imgtime(imgtime_pth):
    '''
    load imaging time from imgtime_pth
    input: 
        file path
    output:
        time(np.array(time_number, 1)): timecode
    '''
    time = np.loadtxt(imgtime_pth, usecols=1, skiprows=1)
    
    return time
            
            
def interpolate_att(imgtime, att_data, att_time):
    '''
    interpolation of camera pose to produce att of timecode in imgtime
    input:
        imgtime(np.array): imaging time
        att_data(np.array): camera pose
        att_time(np.array): camera pose corresponding time
    output:
        att_img(np.array((time_number, 4))): camera pose of imgtime
    '''
    dt = att_time[1] - att_time[0]
    t0_index = ((imgtime - att_time[0]) / dt).astype(np.int32)
    t1_index = t0_index + 1
    t0 = att_time[t0_index]
    t = imgtime.reshape(len(imgtime), 1)
    t1 = att_time[t1_index]
    q0 = att_data[t0_index]
    q1 = att_data[t1_index]
    
    # spherical linear interpolation
    theta = np.arccos(np.sum(np.abs(q0 * q1), axis=1)).reshape(-1, 1)
    eta0 = np.true_divide(np.sin(theta * np.true_divide(t1 - t, t1 - t0)), np.sin(theta))
    eta1 = np.true_divide(np.sin(theta * np.true_divide(t - t0, t1 - t0)), np.sin(theta))
    qt = eta0 * q0 + eta1 * q1
    
    return qt


def interpolate_gps(imgtime, gps_data, gps_time):
    '''
    interpolation of gps data to produce gps of timecode in imgtime
    input:
        imgtime(np.array):imaging time
        gps_data(np.array): gps position and velocity
        gps_time: timecode of gps data
    output:
        p(np.array((time_number, 3))):gps position of imgtime
        v(np.array((time_number, 3))):gps velocity of imgtime
    '''
    dt = gps_time[1] - gps_time[0]
    t0_index = ((imgtime - gps_time[0]) / dt).astype(np.int32)    
    t = imgtime.reshape(len(imgtime), 1)
    
    # lagrange interpolation (former 4 points + latter 4 points)
    tl = np.zeros((len(imgtime), 1, 9))
    pl = np.zeros((len(imgtime), 3, 9))
    vl = np.zeros((len(imgtime), 3, 9))    
    for i in range(9):
        tl[:, :, i] = gps_time[t0_index + i - 4]
        pl[:, :, i] = gps_data[t0_index + i - 4, 0:3]
        vl[:, :, i] = gps_data[t0_index + i - 4, 3:6]
    p = np.zeros((len(imgtime), 3))
    v = np.zeros((len(imgtime), 3))
    for j in range(9):
        s = np.ones((len(imgtime), 1))
        for i in range(9):
            if j == i:
                continue
            s = np.multiply(s, np.true_divide(t - tl[:, :, i], tl[:, :, j] - tl[:, :, i]))
        sp = s * pl[:, :, j]
        sv = s * vl[:, :, j]
        p = p + sp
        v = v + sv
        
    return p, v


def trans_time_utc(imgtime):
    '''
    convert timecode to UTC
    input: 
        imgtime(np.array): imaging timecode
    output:
        imgtime_utc(np.array): imaging UTC time
    '''
    timecode_base = 131862356.2500000000;
    time_utc_base = np.array([[2013, 3, 7 + 4.0 / 24 + 25 / (24 * 60) + 56.25 / (24 * 60 * 60)]])
    time_utc = time_utc_base.repeat(len(imgtime), axis=0)
    time_utc[:, -1] = time_utc[:, -1] + (imgtime - timecode_base)/(24 * 60 * 60)
    
    return time_utc
            

if __name__ == '__main__':
    att_pth = 'E:/Code/SatPhotogrammetry/data_ZY3/DX_ZY3_NAD_att.txt'
    gps_pth = 'E:/Code/SatPhotogrammetry/data_ZY3/DX_ZY3_NAD_gps.txt'
    imgtime_pth = 'E:/Code/SatPhotogrammetry/data_ZY3/DX_ZY3_NAD_imagingTime.txt'
    att_data, att_time = load_att(att_pth)
    gps_data, gps_time = load_gps(gps_pth)
    imgtime = load_imgtime(imgtime_pth)
    # print(att_data[0][0])
    # print(att_time[0][0])
    # print(gps_data[0])
    # print(gps_time[0])
    # print(imgtime.shape[0])
    att = interpolate_att(imgtime, att_data, att_time)
    gps_p, gps_v = interpolate_gps(imgtime, gps_data, gps_time)
    # print(att[0])
    # print(att.shape)
    # print(gps_p[0])
    # print(gps_v[0])
    # print(gps_p.shape)
    # print(gps_v.shape)
    time_utc = trans_time_utc(imgtime)
    # print(time_utc[0])
    # print(time_utc.shape)
    
    