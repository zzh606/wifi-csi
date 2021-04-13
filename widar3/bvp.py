from scipy import signal
import numpy as np
from get_scale_csi import csi_get_all
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.io as scio
import multiprocessing as mul
from scipy.optimize import minimize
import copy
import time


def get_doppler_spectrum(spfx_ges, rx_cnt, rx_acnt, freq_bin_len, method):
    # 设置参数
    sample_rate = 1000
    half_rate = sample_rate / 2
    upper_order = 6
    upper_stop = 60
    lower_order = 3
    lower_stop = 2
    lu, ld = signal.butter(upper_order, upper_stop / half_rate, 'low')
    hu, hd = signal.butter(lower_order, lower_stop / half_rate, 'high')
    freq_bins_unwrap = np.concatenate([np.arange(sample_rate / 2), np.arange(-sample_rate / 2, 0)]) / sample_rate
    freq_lpf_sele = (freq_bins_unwrap <= upper_stop / sample_rate) & (freq_bins_unwrap >= -upper_stop / sample_rate)
    freq_lpf_positive_max = np.sum(freq_lpf_sele[1: int(len(freq_lpf_sele) / 2) - 1])
    freq_lpf_negative_min = np.sum(freq_lpf_sele[int(len(freq_lpf_sele) / 2) - 1:])

    for ii in range(rx_cnt):
        try:
            csi_data = csi_get_all(spfx_ges + '-r' + str(ii + 1) + '.dat')
        except:
            print('获取csi_data出错')
            continue
        # Camera-Ready: Down Sample
        csi_data = csi_data[np.round(np.arange(np.size(csi_data, 0))), :]

        # Select Antenna Pair[WiDance]
        csi_mean = np.mean(np.abs(csi_data), axis=0)
        csi_var = np.sqrt(np.var(np.abs(csi_data), axis=0))
        csi_mean_var_ratio = np.divide(csi_mean, csi_var)
        csi_mean_var_ratio_mean = np.mean(np.transpose(np.reshape(csi_mean_var_ratio, [rx_acnt, 30])), axis=0)
        idx = int(np.where(csi_mean_var_ratio_mean == np.max(csi_mean_var_ratio_mean))[0])
        csi_data_ref = np.tile(csi_data[:, idx * 30: (idx + 1) * 30], [1, rx_acnt])
        # Amp Adjust[IndoTrack]
        csi_data_adj = np.zeros((np.shape(csi_data)), dtype=complex)
        csi_data_ref_adj = np.zeros(np.shape(csi_data_ref), dtype=complex)
        alpha_sum = 0
        print('shape:',np.shape(csi_data))
        exit()
        for jj in range(30 * rx_acnt):
            amp = np.abs(csi_data[:, jj])
            alpha = np.min(amp[amp != 0])
            alpha_sum = alpha_sum + alpha
            csi_data_adj[:, jj] = np.multiply(np.abs(csi_data[:, jj]) - alpha, np.exp(1j * np.angle(csi_data[:, jj])))

        beta = 1000 * alpha_sum / (30 * rx_acnt)
        for jj in range(30 * rx_acnt):
            csi_data_ref_adj[:, jj] = np.multiply(np.abs(csi_data[:, jj]) + beta, np.exp(1j * np.angle(csi_data_ref[:, jj])))

        # Conj Mult
        conj_mult = np.multiply(csi_data_adj, np.conj(csi_data_ref_adj))
        conj_mult = np.concatenate([conj_mult[:, : int(30 * idx)], conj_mult[:, int(30 * (idx + 1)): 90]], axis=1)
        # Filter Out Static Component & High Frequency Component
        for jj in range(np.size(conj_mult, 1)):
            conj_mult[:, jj] = signal.lfilter(lu, ld, conj_mult[:, jj])
            conj_mult[:, jj] = signal.lfilter(hu, hd, conj_mult[:, jj])

        # PCA analysis
        pca = PCA(n_components=60)
        conj_mult_pca = pca.fit_transform(conj_mult.astype(float))[:, 0] #不支持复数，暂时使用这种方法，和MATLAB不同
        # pca_coef = pca.components_
        # conj_mult_pca = np.dot(conj_mult, pca_coef[:, 0])

        # % TFA With CWT or STFT
        if method == 'stft':
            time_instance = np.arange(len(conj_mult_pca))
            window_size = int(np.round(sample_rate / 4 + 1))
            if np.mod(window_size, 2) == 0:
                window_size = window_size + 1
            # f, t, freq_time_prof = signal.spectrogram(conj_mult[:,0],window=signal.get_window('hamming',256),fs=1000,nperseg=256, noverlap=255,return_onesided=False,axis=0)  # 和MATLAB的spectragram()对应
            f, t, freq_time_prof = signal.stft(conj_mult[:, 0], window=signal.get_window('hann', 256), fs=1000, nperseg=256, noverlap=255, return_onesided=False)  # 和MATLAB的tfrsp()对应
        # plt.figure(ii + 1)
        plt.pcolormesh(np.arange(np.size(freq_time_prof, 1)), np.arange(np.size(freq_time_prof, 0)),
                       circshift1D(np.abs(freq_time_prof),int(np.size(freq_time_prof, 0) / 2)))
        # Select Concerned Freq

        # Spectrum Normalization By Sum For Each Snapshot
        freq_time_prof = np.divide(np.abs(freq_time_prof), np.tile(np.sum(np.abs(freq_time_prof), 0), np.size(freq_time_prof, 0)).reshape([np.size(freq_time_prof, 0), np.size(freq_time_prof, 1)]))

        # Frequency Bin(Corresponding to FFT Results)

        # Store Doppler Velocity Spectrum
        if ii == 0:
            doppler_spectrum = np.zeros([rx_cnt, np.size(freq_time_prof, 0), np.size(freq_time_prof, 1)])
        if np.size(freq_time_prof, 1) >= np.size(doppler_spectrum, 2):
            doppler_spectrum[ii, :, :] = freq_time_prof[:, : np.size(doppler_spectrum, 2)]
        else:
            doppler_spectrum[ii, :, :] = np.concatenate([freq_time_prof, np.zeros([np.size(doppler_spectrum, 1), int(np.size(doppler_spectrum, 2) - np.size(freq_time_prof, 1))])], axis=1)

    doppler_spectrum_center = int(np.where(f == np.max(f))[0])
    return doppler_spectrum, circshift1D(f, doppler_spectrum_center)[doppler_spectrum_center - freq_bin_len: doppler_spectrum_center + freq_bin_len + 1]


def circshift3D(lst, k):
    # k是右移
    return np.concatenate([lst[:, -k:, :], lst[:, : -k, :]], axis=1)


def circshift1D(lst, k):
    # k是右移
    return np.concatenate([lst[-k:], lst[: -k]])


def circshift2D(lst, k):
    # k是右移
    return np.concatenate([lst[:,-k:], lst[:, :-k]])


# Input:(If rx_cnt=2)
# A: 2*2
# velocity_bin: 1*M
# freq_bin: 1*F
# Output: 2*M*M*F, component is ...
# 参考Widar的第三节，公式3、5
# 移动速率velocity_bin会影响某些频率f的信号功率分布
# plcr_hz是公式3的f
# VDM是公式5的assignment matrix A
def get_velocity2doppler_mapping_matrix(A, wave_length, velocity_bin, freq_bin, rx_cnt):
    if np.size(A, 0) != rx_cnt:
        print('Error Rx Count')
    F = np.size(freq_bin, 1)
    M = len(velocity_bin)
    freq_min = np.min(freq_bin)
    fre_max = np.max(freq_bin)

    VDM = np.zeros([rx_cnt, M, M, F])  # A的维度为F*M^2。rx_cnt时rx的个数，M是vectorized BVP V元素的个数，M是Link的条数，F是DFS的频率采样点数
    # For Each Link
    for ii in range(rx_cnt):
        for i in range(M):
            for j in range(M):
                plcr_hz = int(np.round(np.dot(A[ii, :], np.transpose([velocity_bin[i], velocity_bin[j]]) / wave_length))) # 公式3
                if plcr_hz > fre_max or plcr_hz < freq_min: # 若超过频率范围，则设为1e10
                    VDM[ii, i, j, :] = 1e10 * np.ones([1, np.size(VDM, 3)])
                    continue
                idx = plcr_hz - freq_min
                # ii时rx的索引，idx是vectorized BVP V的第idx个元素，i是第i条Link，j是DFS的第j个频率采样点
                VDM[ii, i, j, idx] = 1  # 当f_= f(v_k)，设为1
    return VDM


## 公式(4)，建立a_x、a_y矩阵
def get_A_matrix(torso_pos, Tx_pos, Rx_pos, rx_cnt):
    if rx_cnt > np.size(Rx_pos, 0):
        print('Error Rx Count')
    A = np.zeros([rx_cnt, 2])

    for ii in range(rx_cnt):
        dis_torso_tx = np.sqrt(np.sum(np.dot((torso_pos - Tx_pos), (torso_pos - Tx_pos))))
        dis_torso_rx = np.sqrt(np.sum(np.dot((torso_pos - Rx_pos[ii, :]), (torso_pos - Rx_pos[ii, :]))))
        A[ii, :] = np.divide((torso_pos - Tx_pos), dis_torso_tx) + np.divide((torso_pos - Rx_pos[ii, :]), dis_torso_rx)
    return A


# Input:
# P: M*M
# VDM: 2*M*M*F
# doppler_spectrum_seg_tgt: 2*F
# 目标是最小化公式7
def dvm_target_func(x,VDM, lamb, doppler_spectrum_seg_tgt, rx_cnt, norm):
    # VDM, M, lamb, doppler_spectrum_seg_tgt, rx_cnt, norm = args
    M = int(np.sqrt(np.size(x, 0)))
    P = copy.copy(np.reshape(x, [M, M]))

    # Initialize Variable
    F = np.size(doppler_spectrum_seg_tgt, 1)
    P_extent = np.expand_dims(P, 2).repeat(rx_cnt, axis=2)
    P_extent = np.expand_dims(P_extent, 3).repeat(F, axis=3)

    #  Construct Approximation Doppler Spectrum
    doppler_spectrum_seg_approximate = np.sum(np.sum(np.multiply(P_extent, VDM), 0), 0)

    # Construct Loss Function #
    # EMD Distance, 对应公式7
    floss = 0
    for ii in range(rx_cnt):
        if np.sum(doppler_spectrum_seg_tgt[ii, :] > 1e-5):
            floss = floss + np.sum(np.abs(np.dot(doppler_spectrum_seg_approximate[ii, :] - doppler_spectrum_seg_tgt[ii, :], np.triu(np.ones([F, F]), 0))))
    # Norm Loss
    if norm == 1:
        floss = floss + lamb * np.sum(x)
    elif norm == 0:
        floss = floss + lamb * np.sum(x != 0.0)
        # print(x[x!=0], np.where(x!=0)[0])
    # print(x)
    # print(floss)
    # time.sleep(0.01)
    return floss


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
def parallel_function(index, doppler_spectrum_ges, VDM, lamb, norm, M, U_bound, CastM):
    print('进程', index, '开始')
    t_start = time.time()
    # Set-up fmincon Input
    doppler_spectrum_ges_tgt = np.mean(doppler_spectrum_ges, 2)

    # Normalization Between Receivers(Compensate Path-Loss)
    for jj in range(1, np.size(doppler_spectrum_ges_tgt, 0)):
        if np.sum(doppler_spectrum_ges_tgt[jj, :] != 0):
            doppler_spectrum_ges_tgt[jj, :] = doppler_spectrum_ges_tgt[jj, :] * np.sum(doppler_spectrum_ges_tgt[0, :]) / np.sum(doppler_spectrum_ges_tgt[jj, :])

    args = (VDM, lamb, doppler_spectrum_ges_tgt, np.size(doppler_spectrum_ges_tgt, 0), norm,)
    bound = [(0, U_bound[0,0]) for i in range(M*M)]
    cons = {'type': 'ineq', 'fun': lambda x: np.sum(np.multiply(x, np.reshape(CastM, 400)))}  # 少了jac
    # BFGG要指定梯度jac
    res = minimize(fun=dvm_target_func, x0=np.zeros(M*M), args=args, bounds=bound)
    print('进程', index, '退出, 用时: ', time.time() - t_start)
    return index, np.reshape(res.x, [M, M])

def get_CastM_matrix(A, wave_length, velocity_bin, freq_bin):
    AA = A[0: 1, :]
    M = np.size(velocity_bin, 0)
    F_max = np.max(freq_bin)
    F_min = np.min(freq_bin)
    CastM = np.zeros([M, M])

    for ii in range(M):
        for jj in range(M):
            plcr_hz = int(np.round(np.dot(AA, np.transpose([velocity_bin[ii], velocity_bin[jj]]) / wave_length))) # 公式3
            # if np.max(plcr_hz) > F_max or np.min(plcr_hz) < F_min:
            #     CastM[ii, jj] = 1
            if np.sqrt(np.sum(np.multiply(plcr_hz, plcr_hz))) > F_max:
                CastM[ii, jj] = 1
    return CastM


def get_rotated_spectrum(velocity_spectrum, torso_ori):
    return 1


def dvm_main(spfx_ges, rx_cnt, rx_acnt, pos_sel, dpth_people):
    # Segment Settings
    ges_per_file = 1
    norm = 0
    lamb = 1e-7
    torso_pos = np.array(
        [[1.365, 0.455], [0.455, 0.455], [0.455, 1.365], [1.365, 1.365], [0.91, 0.91], [2.275, 1.365], [2.275, 2.275],
         [1.365, 2.275]])
    torso_ori = [-90, -45, 0, 45, 90]
    Tx_pos = [0, 0]
    Rx_pos = np.array([[0.455, -0.455], [1.365, -0.455], [2.0, 0], [-0.455, 0.455], [-0.455, 1.365], [0, 2.0]])
    seg_length = 100
    wave_length = 299792458 / 5.825e9
    V_max = 2
    V_min = -2
    V_bins = 20
    V_resolution = (V_max - V_min) / V_bins
    M = int((V_max - V_min) / V_resolution)
    velocity_bin = (np.arange(1, M + 1) - M / 2) / (M / 2) * V_max  # 有疑问
    MaxFunctionEvaluations = 100000
    freq_bin_len = 30

    # Generate Doppler Spectrum
    doppler_spectrum, freq_bin = get_doppler_spectrum(spfx_ges, rx_cnt, rx_acnt, freq_bin_len, 'stft')
    # data = scio.loadmat('BVP_data/dd.mat')
    # doppler_spectrum = data['doppler_spectrum']
    # freq_bin = data['freq_bin']

    # Cyclic Doppler Spectrum According To frequency bin
    # circ_len = int(np.size(doppler_spectrum, 1) / 2)
    # doppler_spectrum = circshift3D(doppler_spectrum, circ_len)[:, circ_len - freq_bin_len : circ_len + freq_bin_len + 1, :]
    circ_len = np.size(freq_bin, 1) - int(np.where(freq_bin[0, :] == np.max(freq_bin[0, :]))[0])
    doppler_spectrum = circshift3D(doppler_spectrum, circ_len)

    # 画频谱
    # for kk in range(6):
    #     plt.figure(kk + 1)
    #     plt.plot(doppler_spectrum[kk, :, 0])
    #     plt.show()

    # 画Spectrogram
    # for kk in range(6):
    #     plt.figure(kk+1)
    #     # plt.pcolormesh(np.arange(np.size(doppler_spectrum,2)),np.arange(np.size(doppler_spectrum, 1)), doppler_spectrum[kk, :, :])
    #     plt.pcolormesh(np.arange(np.size(doppler_spectrum,2)),np.arange(-60, 61), doppler_spectrum[kk, :, :])
    #     plt.ylabel('Frequency [Hz]')
    #     plt.xlabel('Time [sec]')
    #     plt.show()
    # exit()
    # For Each Segment Do Mapping
    doppler_spectrum_max = np.max(doppler_spectrum)
    U_bound = np.tile(doppler_spectrum_max, [M, M])
    A = get_A_matrix(torso_pos[pos_sel, :], Tx_pos, Rx_pos, rx_cnt)
    VDM = np.transpose(get_velocity2doppler_mapping_matrix(A, wave_length, velocity_bin, freq_bin, rx_cnt), [1, 2, 0, 3])
    CastM = get_CastM_matrix(A, wave_length, velocity_bin, freq_bin)

    for ges_number in range(ges_per_file):
        seg_number = int(np.floor(np.size(doppler_spectrum, 2) / seg_length))
        doppler_spectrum_ges = doppler_spectrum.copy()
        velocity_spectrum = np.zeros([M, M, seg_number])

        pool = mul.Pool(processes=10)
        result = []
        for ii in range(seg_number):
            result.append(pool.apply_async(parallel_function, args=(ii, doppler_spectrum_ges[:, :, ii * seg_length: (ii + 1) * seg_length], VDM, lamb, norm, M, U_bound, CastM)))
        pool.close()
        pool.join()
        for res in result:
            velocity_spectrum[:, :, res.get()[0]] = res.get()[1]

        scio.savemat('BVP_data/' + dpth_people + '-' + str(ges_number) + '-' + str(lamb) + '-' + str(seg_length) + str(V_bins) + '-L' + str(norm) + '.mat', {'velocity_spectrum': velocity_spectrum})


def generate_vs():
    start_index = [0, 0, 0, 0]
    total_mo = 1
    total_pos = 1
    total_ori = 1
    total_ges = 1
    start_index_met = 0
    rx_cnt = 6
    rx_acnt = 3
    dpth_pwd = './Widar3.0/BVPExtractionCode/'
    dpth_date = 'Data'
    dpth_people = 'userA'

    dpth_ges = dpth_pwd + dpth_date + '/'
    dpth_vs = dpth_pwd + 'Data/'
    exception_fid = open(dpth_vs + 'exception_log_' + dpth_date + '-' + dpth_people + '.log', 'w')
    exception_fid.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    t_start = time.time()
    for mo_sel in range(total_mo):
        for pos_sel in range(total_pos):
            for ori_sel in range(total_ori):
                for ges_sel in range(total_ges):
                    spfx_ges = dpth_people + '-' + str(mo_sel + 1) + '-' + str(pos_sel + 1) + '-' + str(ori_sel + 1) + '-' + str(ges_sel + 1)
                    if mo_sel == start_index[2] and pos_sel == start_index[1] and ori_sel == start_index[2] and ges_sel == start_index[3]:
                        start_index_met = 1
                    if start_index_met == 1:
                        print('Running: ' + spfx_ges)
                        try:
                            dvm_main(dpth_ges + spfx_ges, rx_cnt, rx_acnt, pos_sel, dpth_people)
                            print('dvm_main()完成')
                        except:
                            print('出现异常')
                            exception_fid.write(spfx_ges)
                            exception_fid.write('错误信息')
                            continue
                    else:
                        print('Skipping' + spfx_ges)
    print('运行总时长: ', time.time() - t_start)
    exception_fid.close()