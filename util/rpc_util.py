import numpy as np
import os


class RPC():

    def __init__(self):
        self.a = np.ones(20)
        self.b = np.ones(20)
        self.c = np.ones(20)
        self.d = np.ones(20)
        
    
    def gravity_centralize(self, photo_loc, ground_points):
        self.c_g = photo_loc[:, 1].flatten().mean()
        self.r_g = photo_loc[:, 0].flatten().mean()
        self.c_s = max(np.max(photo_loc[:, 1]) - self.c_g, self.c_g - np.min(photo_loc[:, 1]))
        self.r_s = max(np.max(photo_loc[:, 0]) - self.r_g, self.r_g - np.min(photo_loc[:, 0]))
        self.X_g = ground_points[:, 0].flatten().mean()
        self.Y_g = ground_points[:, 1].flatten().mean()
        self.Z_g = ground_points[:, 2].flatten().mean()
        self.X_s = max(np.max(ground_points[:, 0]) - self.X_g, self.X_g - np.min(ground_points[:, 0]))
        self.Y_s = max(np.max(ground_points[:, 1]) - self.Y_g, self.Y_g - np.min(ground_points[:, 1]))
        self.Z_s = max(np.max(ground_points[:, 2]) - self.Z_g, self.Z_g - np.min(ground_points[:, 2]))
        
        photo_n = photo_loc.astype(np.float)
        ground_n = ground_points.astype(np.float)
        # image space coordinates r,c
        photo_n[:, 1] = (photo_n[:, 1] - self.c_g) / self.c_s
        photo_n[:, 0] = (photo_n[:, 0] - self.r_g) / self.r_s
        # ground points coordinates x,y,z/B,L,H
        ground_n[:, 0] = (ground_points[:, 0] - self.X_g) / self.X_s
        ground_n[:, 1] = (ground_points[:, 1] - self.Y_g) / self.Y_s
        ground_n[:, 2] = (ground_points[:, 2] - self.Z_g) / self.Z_s
        
        return photo_n, ground_n
        

    def init_abcd(self, photo_n, ground_n):
        # ground space coordinates
        X = ground_n[:, 0].reshape(-1, 1)
        Y = ground_n[:, 1].reshape(-1, 1)
        Z = ground_n[:, 2].reshape(-1, 1)
        # image space coordinates
        R = photo_n[:, 0].reshape(-1, 1)
        C = photo_n[:, 1].reshape(-1, 1)
        # obtain coefficient matrices for 19 parameters based on third polynomial
        l = len(X)
        vec = np.hstack([np.ones((l, 1)), 
                        Z, Y, X,
                        Z * Y, Z * X, Y * X,
                        Z * Z, Y * Y, X * X,
                        Z * Y * X, Z * Z * Y, Z * Z * X,
                        Y * Y * Z, Y * Y * X, Z * X * X,
                        Y * X * X, Z * Z * Z, Y * Y * Y,
                        X * X * X])
        M = np.concatenate([vec, -np.multiply(R, vec[:, 1:])], axis=1)
        N = np.concatenate([vec, -np.multiply(C, vec[:, 1:])], axis=1)
        # merge M, N to A; R, C to L
        A = np.block([[M, np.zeros(N.shape)], 
                  [np.zeros(M.shape), N]])
        L = np.concatenate([R, C])
        # tmp = (A.t*A)^(-1)A.t*L
        tmp = np.dot(np.dot(np.linalg.inv(np.dot(A.transpose(), A)), A.transpose()), L)
        # calculate initial value of a, b, c, d
        self.a = tmp[0: 20].flatten(order='F') #flatten by col
        self.b[1:] = tmp[20: 39].flatten(order='F')
        self.c = tmp[39: 59].flatten(order='F')
        self.d[1:] = tmp[59: 78].flatten(order='F')
        
        
    def real_solve(self, photo_n, ground_n):
        # ground space coordinates
        X = ground_n[:, 0].reshape(-1, 1)
        Y = ground_n[:, 1].reshape(-1, 1)
        Z = ground_n[:, 2].reshape(-1, 1)
        # image space coordinates
        R = photo_n[:, 0].reshape(-1, 1)
        C = photo_n[:, 1].reshape(-1, 1)
        l = len(X)
        vec = np.hstack([np.ones((l, 1)), 
                         Z, Y, X, 
                         Z * Y, Z * X, Y * X, 
                         Z * Z, Y * Y, X * X, 
                         Z * Y * X, Z * Z * Y, Z * Z * X, 
                         Y * Y * Z, Y * Y * X, Z * X * X, 
                         Y * X * X, Z * Z * Z, Y * Y * Y, 
                         X * X * X])
        # construct linear error equation B, J, D, K -> V_r = W_r*M*J-W_r*R, V_c = W_c*N*K-W_c*C
        B = np.dot(vec, self.b.reshape(-1, 1))
        M = np.concatenate([vec, -np.multiply(R, vec[:, 1:])], axis=1)
        W_r = np.true_divide(np.identity(l), B)
        D = np.dot(vec, self.d.reshape(-1, 1))
        N = np.concatenate([vec, -np.multiply(C, vec[:, 1:])], axis=1)
        W_c = np.true_divide(np.identity(l), D)
        # construct normal equation M.t*W_r^2*M*J = M.t*W_r^2*R, N.t*W_c^2*N*K = N.t*W_c*C
        # merge J, K to U; M, N to A; R, C to L -> A.t*W^2*A*U = A.t*W^2*L
        A = np.block([[M, np.zeros(N.shape)], 
                      [np.zeros(M.shape), N]])
        W = np.block([[W_r, np.zeros(W_c.shape)], 
                      [np.zeros(W_r.shape), W_c]])
        L = np.concatenate([R, C])
        T = np.dot(np.dot(np.dot(A.transpose(), W), W), A) # T = A.t*W^2*A
        det_T = np.linalg.det(T)
        if det_T < 0.01:
            print('The coefficient matrix may be a singular matrix. Eigenvalue: {}'.format(det_T))
        U = np.dot(np.dot(np.dot(np.dot(np.linalg.inv(T), A.transpose()), W), W), L) # U = (A.t*W^2*A)^(-1)*A.t*W^2*L
        V = np.dot(np.dot(W, A), U) - np.dot(W, L) # V = W*A*U-W*L
    
        return U, V
    
        
    def solve(self, photo_loc, ground_points, max_iter, threshold):
        photo_n, ground_n = self.gravity_centralize(photo_loc, ground_points)
        self.init_abcd(photo_n, ground_n)
        # iteration solve
        count = 1
        while(count <= max_iter):
            tmp_u, v = self.real_solve(photo_n, ground_n)
            tmp_a = tmp_u[0:20].flatten(order='F')
            tmp_b = tmp_u[20:39].flatten(order='F')
            tmp_c = tmp_u[39:59].flatten(order='F')
            tmp_d = tmp_u[59:78].flatten(order='F')
            # adjust quit condition
            print('Iteration {}: residuals {:.3f}'.format(count, np.max(abs(v))))
            if(np.max(abs(v)) < threshold):
                print('\n')
                print('Final iteration number: {}'.format(count))
                return tmp_a, tmp_b, tmp_c, tmp_d, True
            else:
                self.a = tmp_a
                self.b[1:] = tmp_b
                self.c = tmp_c
                self.d[1:] = tmp_d
            count += 1
            unsolve = np.array([])
        return unsolve, unsolve, unsolve, unsolve, False


    def save_result(self, result_pth):
        with open(result_pth, 'w') as f:
            self.ncenter = np.ones(10)
            self.ncenter[0] = self.r_g
            self.ncenter[1] = self.c_g
            self.ncenter[2] = self.X_g
            self.ncenter[3] = self.Y_g
            self.ncenter[4] = self.Z_g
            self.ncenter[5] = self.r_s
            self.ncenter[6] = self.c_s
            self.ncenter[7] = self.X_s
            self.ncenter[8] = self.Y_s
            self.ncenter[9] = self.Z_s
            # save offset
            list_name = ['LINE_OFF: ', 'SMAP_OFF: ', 'LAT_OFF: ', 'LONG_OFF: ', 'HEIGHT_OFF: ', 
                        'LINE_SCALE: ', 'SAMP_SCALE: ', 'LAT_SCALE: ', 'LONG_SCALE: ', 'HEIGHT_SCALE: ']
            list_unit = ['pixels', 'pixels', 'degrees', 'degrees', 'meters',
                        'pixels', 'pixels', 'degrees', 'degrees', 'meters']
            for i in range(10):
                f.write(list_name[i] + str(self.ncenter[i]) + list_unit[i] + '\n')
            # save RPC parameters
            list_name2 = ['LINE_NUM_COEFF_', 'LINE_DEN_COEFF_', 'SAMP_NUM_COEFF_', 'SAMP_DEN_COEFF_']
            for i in range(20):
                f.write(list_name2[0] + str(i + 1) + ": " + str(self.a[i]) + "\n")
            for i in range(20):
                f.write(list_name2[1] + str(i + 1) + ": " + str(self.b[i]) + "\n")
            for i in range(20):
                f.write(list_name2[2] + str(i + 1) + ": " + str(self.c[i]) + "\n")
            for i in range(20):
                f.write(list_name2[3] + str(i + 1) + ": " + str(self.d[i]) + "\n")
            
            
    def save_data(self, final_dir):
        # save offset
        offset_pth = os.path.join(final_dir, 'offset.npy')
        np.save(offset_pth, self.ncenter)
        # save RPC parameters
        paras_pth = os.path.join(final_dir, 'paras.npy')
        self.a = self.a.reshape(-1, 1)
        self.b = self.b.reshape(-1, 1)
        self.c = self.c.reshape(-1, 1)
        self.d = self.d.reshape(-1, 1)
        paras = np.hstack((self.a, self.b, self.c, self.d))
        np.save(paras_pth, paras)