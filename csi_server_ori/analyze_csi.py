import scipy.io as scio
import matlab.engine as eng
import numpy as np
import math


if __name__ == '__main__':
    sample_rate = 1000
    T = 100
    F = 30
    A = 3
    L = 10
    N = 100
    FI = 2 * 312.5e3
    TI = 1 / sample_rate
    AS = 0.5
    TR = np.arange(-100, 100, 1) * 1e-9
    AR = np.arange(0, 180, 1) / 180 * math.pi
    DR = np.arange(-20, 20, 1)
    UR = 1

    inputfile = './MATLAB_data/S01.mat'
    data = scio.loadmat(inputfile)
    csi_data = data['csi_data']

    en = eng.start_matlab()
    en.sage_set_const(T, F, A, L, N, FI, TI, AS, TR, AR, DR, UR)
