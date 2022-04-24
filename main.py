import argparse
import numpy as np
from tqdm import tqdm
from time import time
import cv2
import os
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from osgeo import gdal
from multiprocessing.dummy import Pool
from functools import partial
from util import ext_util, rot_util, rpc_util


parser = argparse.ArgumentParser(description='RPC')
parser.add_argument('--ctrl_grid_m', type=int, default=10, help='number of control grid(m,n)')
parser.add_argument('--ctrl_grid_n', type=int, default=10, help='number of control grid(m,n)')
parser.add_argument('--ctrl_layer', type=int, default=6, help='layer number of control grid')
parser.add_argument('--check_grid_m', type=int, default=20)
parser.add_argument('--check_grid_n', type=int, default=20)
parser.add_argument('--max_iter', type=int, default=20, help='Maximum number of RPC iterations')
parser.add_argument('--threshold', type=float, default=0.001, help='threshold to exit iteration')

parser.add_argument('--data_dir', type=str, default='data/data_ZY3', help='path of data')
parser.add_argument('--dem_pth', type=str, default='data/dem/n35_e114_1arc_v3.tif', help='path of dem dir')
parser.add_argument('--inter_dir', type=str, default='output/inter', help='path of itermdiate result')
parser.add_argument('--final_dir', type=str, default='output/final', help='path of final result')


def preprocess(args):
    print('------------------------------------------------------')
    print('Start preprocessing!')
    print('------------------------------------------------------')
    time0 = time()
    # load data
    time1 = time()
    imgpth = os.path.join(args.data_dir, 'zy3.tif')
    img = cv2.imread(imgpth)
    args.img_row, args.img_col, _ = img.shape
    att_pth = os.path.join(args.data_dir, 'DX_ZY3_NAD_att.txt')
    gps_pth = os.path.join(args.data_dir, 'DX_ZY3_NAD_gps.txt')
    imgtime_pth = os.path.join(args.data_dir, 'DX_ZY3_NAD_imagingTime.txt')
    att_data, att_time = ext_util.load_att(att_pth)
    gps_data, gps_time = ext_util.load_gps(gps_pth)
    imgtime = ext_util.load_imgtime(imgtime_pth)
    time2 = time()
    print('Data loading completed in {:.3f}ms.'.format((time2 - time1) * 1000))
    print('\n')
    
    # data interpolation
    args.att = ext_util.interpolate_att(imgtime, att_data, att_time)
    args.gps_p, args.gps_v = ext_util.interpolate_gps(imgtime, gps_data, gps_time)
    time3 = time()
    print('Data interpolation completed in {:.3f}ms.'.format((time3 - time2) * 1000))
    print('\n')
    
    # timecdoe to UTC time
    time_utc = ext_util.trans_time_utc(imgtime)
    time_inday = time_utc[:, 2] - (time_utc[:, 2]).astype(np.int32)
    hour = (time_inday * 24).astype(np.int32)
    minute = ((time_inday * 24-hour) * 60).astype(np.int32)
    second = (((time_inday * 24-hour) * 60) - minute) * 60
    with open(os.path.join(args.inter_dir, 'time_utc.txt'), 'w') as f:
        for i in range(len(time_utc)):
            f.write(str(int(time_utc[i][0])) + ' ' + str(int(time_utc[i][1])) + ' ' + 
                    str(int(time_utc[i][2])) + ' ' + str(hour[i]) + ' ' + str(minute[i]) + ' ' + str(second[i]) + '\n')
    print('UTC imaging time saved.')
    print('\n')
    
    # obtain J2000->WGS84 rotation matrix according to UTC time (matlab)
    time4 = time()
    time_pth = os.path.join(os.getcwd(), args.inter_dir, 'time_utc.txt').replace('/', '\\')
    j2wrot_pth = os.path.join(os.getcwd(), args.inter_dir, 'j2w_rot.txt').replace('/', '\\')
    import matlab
    import matlab.engine
    print('Matlab core activated.')
    engine = matlab.engine.start_matlab()
    print('Calculating J200 to WGS84 rotation matrices...')
    engine.j2wrot(time_pth, j2wrot_pth, nargout=0)
    time5 = time()
    print('Rotation matrices saved, matlab processing {:.3f}ms.'.format((time5 - time4) * 1000))
    print('\n')
    
    # save gps position of zy3
    np.save(os.path.join(args.inter_dir, 'gps_p.npy'), args.gps_p)
    print('GPS position of ZY3 saved.')
    print('\n')
    
    # obtain range of elevation
    dem = gdal.Open(args.dem_pth)
    dem_width = dem.RasterXSize
    dem_height = dem.RasterYSize
    dem_data = dem.ReadAsArray(0, 0, dem_width, dem_height)
    args.height_max = np.max(dem_data)
    args.height_min = np.min(dem_data)
    print('DEM information: Max Elevation {:.3f}, Min Elevation {:.3f}'.format(args.height_max, args.height_min))
    # affine transformation
    transformer = dem.GetGeoTransform()
    top_left_x = transformer[0]
    top_left_y = transformer[3]
    bottom_right_x = transformer[0] + (dem_width - 1) * transformer[1] + (dem_height - 1) * transformer[2]
    bottom_right_y = transformer[3] + (dem_width - 1) * transformer[4] + (dem_height - 1) * transformer[5]
    print('DEM range from top left ({:.3f}, {:.3f}) to bottom right ({:.3f}, {:.3f}).'
          .format(top_left_x, top_left_y, bottom_right_x, bottom_right_y))
    print('\n')
    
    time6 = time()
    print('Finish preprocessing in {:.3f}ms!'.format((time6 - time0) * 1000))
    print('------------------------------------------------------')
    print('\n')
    

def generate_control(args):
    print('------------------------------------------------------')
    print('Start generating control grid!')
    print('------------------------------------------------------')
    time0 = time()
    # load data and format transformation
    time1 = time()
    direct_pth = os.path.join(args.data_dir, 'NAD.cbr')
    u_rot_pth = os.path.join(args.data_dir, 'NAD.txt')
    j2w_rot_pth = os.path.join(args.inter_dir, 'j2w_rot.txt')
    args.u_rot = rot_util.load_u_rot(u_rot_pth)
    args.b2j_rot = R.from_quat(args.att).as_matrix()
    args.j2w_rot = rot_util.load_j2w_rot(j2w_rot_pth)
    args.direct_data = rot_util.load_direct_data(direct_pth)
    args.u_vec = rot_util.get_u_vec(args.direct_data)
    time2 = time()
    print('Data loading completed in {:.3f}ms.'.format((time2 - time1) * 1000))
    print('\n')
    
    # save j2w_rot, b2j_rot, u_vec
    np.save(os.path.join(args.inter_dir, 'j2w_rot.npy'), args.j2w_rot)
    np.save(os.path.join(args.inter_dir, 'b2j_rot.npy'), args.b2j_rot)
    np.save(os.path.join(args.inter_dir, 'u_vec'), args.u_vec)
    print('J2W matrices, B2J matrices, u vector saved.')
    print('\n')
    
    # even selection of image space coordinates r,c (align along the scan line)
    time3 = time()
    args.photo_loc = [[int((args.img_row - 1) / args.ctrl_grid_n * j),
                  int((args.img_col - 1) / args.ctrl_grid_m * i)]
                 for j in range(args.ctrl_grid_n + 1)
                 for i in range(args.ctrl_grid_m + 1)]
    args.photo_loc = np.array(args.photo_loc)
    time4 = time()
    print('Image space coordinates selection completed in {:.3f}ms.'.format((time4 - time3) * 1000))
    print('\n')
    
    # set control grid points
    time5 = time()
    iterator = [[int((args.img_row -1) / args.ctrl_grid_n * j),
             int((args.img_col - 1)/ args.ctrl_grid_m * i)]
            for j in range(args.ctrl_grid_n + 1)
            for i in range(args.ctrl_grid_m + 1)]
    h_layer = [args.height_min + i * (args.height_max - args.height_min) / (args.ctrl_layer - 1)
               for i in range(args.ctrl_layer)]
    pool = Pool(6)
    ground_points = np.array([])
    count = 0
    def grid_function(range_zip, height):
        i, j = range_zip
        return rot_util.get_wgs84_vec(args.u_vec[:, j], args.u_rot ,args.b2j_rot[i, :, :],
                                      args.j2w_rot[i, :, :], args.gps_p[i, :], height)
    # get WGS84 coordinates of grid points on every layer
    for h_i in h_layer:
        partial_func = partial(grid_function, height=h_i)
        # map -> data in iterator as input arg rang_zip
        base_points = np.array(pool.map(partial_func, iterator)).squeeze()
        if count:
            ground_points = np.concatenate([ground_points, base_points])
        else:
            ground_points = base_points
        count += 1
    # transform WGS84 coordinates to BLH coordinates
    args.ground_points = rot_util.wgs2blh(ground_points)
    np.save(os.path.join(args.inter_dir, 'gcp_wgs.npy'), ground_points)
    np.save(os.path.join(args.inter_dir, 'gcp_blh.npy'), args.ground_points)
    time6 = time()
    print('Control grid points BLH coordinates obtained in {:.3f}ms.'.format((time6 - time5) * 1000))
    print('\n')

    time7 = time()
    print('Finish generating control points in {:.3f}ms!'.format((time7 - time0) * 1000))
    print('------------------------------------------------------')
    print('\n')
    
    
def solve_rpc(args):
    time0 = time()
    print('------------------------------------------------------')
    print('Start RPC parameters solving!')
    print('------------------------------------------------------')
    # expand image space coordinates (r,c) to layer dimension (len(photo_loc) * len(h_layer)) to match ground_points
    args.photo_loc = np.tile(args.photo_loc, (args.ctrl_layer, 1))
    rpc = rpc_util.RPC()
    a, b, c, d, solve = rpc.solve(args.photo_loc, args.ground_points, args.max_iter, args.threshold)
    if solve:
        print('RPC parameters solved after {} iterations.'.format(args.max_iter))
    else:
        print('RPC parameters unsolved after {} iterations.'.format(args.max_iter))
    time1 = time()
    print('RPC parameters solvement completed in {:.3f}ms!'.format((time1 - time0) * 1000))
    print('\n')
    if solve:
        rpc.save_result(os.path.join(args.final_dir, 'result.txt'))
        rpc.save_data(args.final_dir)
    print('Solved result saved.')
    time2 = time()
    print('Finish RPC parameters solving in {:.3f}ms!'.format((time2 - time0) * 1000))
    print('------------------------------------------------------')
    print('\n')



if __name__ == '__main__':
    args = parser.parse_args()
    time0 = time()
    preprocess(args)
    generate_control(args)
    solve_rpc(args)
    time1 = time()
    print('Total processing time {:.3f}ms'.format((time1 - time0) * 1000))
