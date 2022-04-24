import os
from tabnanny import check
import numpy as np
from osgeo import gdal
from util import rpc_util, rot_util
from main import parser
from time import time


def rever_affine_trans(transformer, x, y):
    T = np.array([[transformer[1], transformer[2]], 
                      [transformer[4], transformer[5]]])
    C = np.array([[x - transformer[0]], [y - transformer[3]]])
    out = np.dot(np.linalg.inv(T), C)
    
    return out[0], out[1]


def affine_trans(transformer, w, h):
    x = transformer[0] + w * transformer[1] + h * transformer[2]
    y = transformer[3] + w * transformer[4] + h * transformer[5]
    
    return x, y


def get_checkpoints(ground_area, dem, grid_m, grid_n):
    dem_width = dem.RasterXSize
    dem_height = dem.RasterYSize
    dem_data = dem.ReadAsArray(0, 0, dem_width, dem_height)
    transformer = dem.GetGeoTransform()
    
    y_max, y_min, x_max, x_min = ground_area
    y_max -= 0.05
    y_min += 0.05
    x_max -= 0.05
    x_min += 0.05
    
    w_max, h_max = rever_affine_trans(transformer, x_max, y_min)
    w_min, h_min = rever_affine_trans(transformer, x_min, y_max)
    
    photo_loc = [[int((h_max - h_min) / grid_n * j + h_min),
                  int((w_max - w_min) / grid_m * i + w_min)] 
                 for j in range(0, grid_n + 1) for i in range(0, grid_m + 1)]
    
    check_points = []
    for point in photo_loc:
        x, y = affine_trans(transformer, point[1], point[0])
        b = y
        l = x
        h = dem_data[point[0], point[1]]
        check_points.append([b, l, h])
    check_points = np.array(check_points)
        
    return check_points
        


if __name__ == '__main__':
    print('------------------------------------------------------')
    print('Start evaluating RPC parameters!')
    print('------------------------------------------------------')
    time0 = time()
    args = parser.parse_args()
    args.eval_dir = 'output/eval'
    gps_p = np.load(os.path.join(args.inter_dir, 'gps_p.npy'))
    j2w_rot = np.load(os.path.join(args.inter_dir, 'j2w_rot.npy'))
    b2j_rot = np.load(os.path.join(args.inter_dir, 'b2j_rot.npy'))
    u_vec = np.load(os.path.join(args.inter_dir, 'u_vec.npy'))
    ground_area = np.load(os.path.join(args.inter_dir, 'ground_area.npy'))
    dem = gdal.Open(args.dem_pth)
    grid_m = args.check_grid_m
    grid_n = args.check_grid_n
    
    check_points = get_checkpoints(ground_area, dem, grid_m, grid_n)
    np.save(os.path.join(args.eval_dir, 'checkgcp.npy'), check_points)
    
    offset = np.load(os.path.join(args.final_dir, 'offset.npy'))
    paras = np.load(os.path.join(args.final_dir, 'paras.npy'))
    rpc = rpc_util.RPC()
    rpc.load_paras(offset, paras)
    
    check_points = rot_util.blh2wgs(check_points)
    
    # window dichotomy check
    photo_check, ground_check = rpc.dichotomy_check(check_points, gps_p, j2w_rot, b2j_rot, u_vec)
    
    ground_check = rot_util.wgs2blh(ground_check)
    photo_check, ground_check = rpc.norm_offset(photo_check, ground_check, offset)
    l = photo_check.shape[0]
    
    pred_photo_check = rpc.pred(ground_check)
    
    # precision evaluation
    pred_photo_check, ground_check = rpc.denorm_offset(pred_photo_check, ground_check, offset)
    photo_check, _ = rpc.denorm_offset(photo_check, np.array([]), offset)
    
    # mean absolute error
    error_r = np.mean(np.abs(pred_photo_check[:, 0] - photo_check[:, 0]))
    error_c = np.mean(np.abs(pred_photo_check[:, 1] - photo_check[:, 1]))
    ground_check_points_origin = np.load(os.path.join(args.eval_dir, 'checkgcp.npy'))
    error_X = np.mean(np.abs(ground_check[:, 0] - ground_check_points_origin[:, 0]))
    error_Y = np.mean(np.abs(ground_check[:, 1] - ground_check_points_origin[:, 1]))
    error_Z = np.mean(np.abs(ground_check[:, 2] - ground_check_points_origin[:, 2]))
    ground_check = rot_util.blh2wgs(ground_check)
    ground_check_points_origin = rot_util.blh2wgs(ground_check_points_origin)
    error_X_ = np.mean(np.abs(ground_check[:, 0] - ground_check_points_origin[:, 0]))
    error_Y_ = np.mean(np.abs(ground_check[:, 1] - ground_check_points_origin[:, 1]))
    error_Z_ = np.mean(np.abs(ground_check[:, 2] - ground_check_points_origin[:, 2]))
    
    print('Evaluation report:')
    print('mae_r(image space): {}'.format(error_r))
    print('mae_c(image space): {}'.format(error_c))
    print('mae_X(object space-BLH): {}'.format(error_X))
    print('mae_Y(object space-BLH): {}'.format(error_Y))
    print('mae_Z(object space-BLH): {}'.format(error_Z))
    print('mae_X(object space-WGS84): {}'.format(error_X_))
    print('mae_Y(object space-WGS84): {}'.format(error_Y_))
    print('mae_Z(object space-WGS84): {}'.format(error_Z_))
    print('\n')
    
    time1 = time()
    print('Finsh evaluation in {:.3f}ms'.format((time1 - time0)* 1000))
    print('------------------------------------------------------')
    print('\n')
    
    