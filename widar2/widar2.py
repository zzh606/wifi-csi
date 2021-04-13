import picos as pic
import numpy as np
import scipy.io as scio
import math
import copy
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import os
from scipy import interpolate
import time
from scipy import signal

plt.figure(1)
data_path = '../MATLAB_data/'
ddata_path = 'MATLAB_org_data/'
static_file = 'S01.mat'
trace_file = 'T02.mat'
cdata = scio.loadmat(data_path + trace_file)
ccsi_data = cdata['csi_data']
ctime_stamp = cdata['time_stamp']
fdata = scio.loadmat(data_path + 'location-' + trace_file)
location_truth = fdata['location']
cdata = scio.loadmat(data_path + 'device_config.mat')
rx_loc = cdata['rx_loc']  # 接收机位置
tx_loc = cdata['tx_loc']  # 发射机位置
c = np.squeeze(cdata['c'])
carrier_frequency = np.squeeze(cdata['carrier_frequency'])
yb = np.squeeze(cdata['yb'])  # y轴取值范围
xb = np.squeeze(cdata['xb'])  # x轴取值范围


def sage_const_build(T, F, A, L, N, FI, TI, AS, TR,AR, DR, UR):
    sage_const_temp = {
        'T': T,
        'F': F,
        'A': A,
        'L': L,
        'N': N,
        'frequency_interval': FI,
        'time_interval': TI,
        'antenna_spatio': AS,
        'tof_range': TR,
        'aoa_range': AR,
        'doppler_range': DR,
        'update_ratio': UR
    }
    return sage_const_temp


def sage_generate_steering_vector(sage_const):
    T = sage_const['T']
    F = sage_const['F']
    A = sage_const['A']
    frequency_interval = sage_const['frequency_interval']
    time_interval = sage_const['time_interval']
    antenna_spatio = sage_const['antenna_spatio']
    doppler_range = sage_const['doppler_range']
    aoa_range = sage_const['aoa_range']
    tof_range = sage_const['tof_range']
    tof_candidates = np.zeros([F, np.size(tof_range, 1)], dtype = complex)
    aoa_candidates = np.zeros([A, np.size(aoa_range, 1)], dtype = complex)
    doppler_candidates = np.zeros([T, np.size(doppler_range, 1)], dtype = complex)

    temp1 = -1j * 2 * np.pi * frequency_interval * np.arange(0, F).reshape([F, 1]) * tof_range
    for ii in range(0, len(temp1)):
        for jj in range(0, len(temp1[0])):
            tof_candidates[ii, jj] = (np.exp(temp1[ii,jj]))  # 复数指数用cmath库
    sage_const.update({'tof_candidates': tof_candidates})

    cos_temp = np.zeros([1, np.size(aoa_range, 1)])
    for ii in range(np.size(aoa_range, 1)):
        cos_temp[0, ii] = np.cos(aoa_range[0, ii])
    temp2 = -1j * 2 * np.pi * antenna_spatio * np.arange(0, A).reshape([A, 1]) * cos_temp
    for ii in range(0, len(temp2)):
        for jj in range(0, len(temp2[0])):
            aoa_candidates[ii, jj] = (np.exp(temp2[ii, jj]))  # 复数指数用cmath库,或np.exp([...])
    sage_const.update({'aoa_candidates': aoa_candidates})

    temp3 = 1j * 2 * np.pi * time_interval * np.arange(0, T).reshape([T, 1]) * doppler_range
    for ii in range(0, len(temp3)):
        for jj in range(len(temp3[0])):
            doppler_candidates[ii, jj] = (np.exp(temp3[ii, jj]))  # 复数指数用cmath库
    sage_const.update({'doppler_candidates': doppler_candidates})
    return sage_const


def csi_sanitization(csi_data, M, N):
    # M = 3 = A, N = 30 = F
    freq_delta = 2 * 312.5e3

    csi_phase = np.zeros(M*N)
    for ii in range(1, M+1):
        if ii == 1:
            csi_phase[(ii-1)*N: ii*N] = np.unwrap(np.angle(csi_data[(ii-1)*N: ii*N]))
        else:
            csi_diff = np.angle(np.multiply(csi_data[(ii-1)*N: ii*N], np.conj(csi_data[(ii-2)*N: (ii-1)*N])))
            csi_phase[(ii-1)*N: ii*N] = np.unwrap(csi_phase[(ii-2)*N: (ii-1)*N]+csi_diff)

    ai = 2 * np.pi * freq_delta * np.tile(np.arange(0, N), M).reshape([1, N*M])
    bi = np.ones((1, len(csi_phase)))
    ci = csi_phase
    A = np.dot(ai, np.transpose(ai))[0]
    B = np.dot(ai, np.transpose(bi))[0]
    C = np.dot(bi, np.transpose(bi))[0]
    D = np.dot(ai, np.transpose(ci))[0]
    E = np.dot(bi, np.transpose(ci))[0]
    rho_opt = (B * E - C * D) / (A * C - np.square(B))
    beta_opt = (B * D - A * E) / (A * C - np.square(B))

    csi_phase_2 = csi_phase + 2 * np.pi * freq_delta * np.tile(np.arange(0, N), M).reshape([1, N*M]) * rho_opt + beta_opt
    result = np.multiply(np.abs(csi_data), np.exp(1j * csi_phase_2))
    return result


def sage_signal(latent_parameter, latent_index, sage_const):
    T = sage_const['T']
    F = sage_const['F']
    A = sage_const['A']

    tof_matrix_temp = np.zeros([F, T, A], dtype=complex)
    t3 = sage_const['tof_candidates'][:, latent_index[0]]
    for kk in range(np.size(tof_matrix_temp, 2)):
        for jj in range(np.size(tof_matrix_temp, 1)):
            tof_matrix_temp[:, jj, kk] = t3
    tof_matrix = np.transpose(tof_matrix_temp, [1, 0, 2])

    aoa_matrix_temp = np.zeros([A, T, F], dtype=complex)
    t1 = sage_const['aoa_candidates'][:, latent_index[1]]
    for kk in range(np.size(aoa_matrix_temp, 2)):
        for jj in range(np.size(aoa_matrix_temp, 1)):
            aoa_matrix_temp[:, jj, kk] = t1
    aoa_matrix = np.transpose(aoa_matrix_temp, [1, 2, 0])

    doppler_matrix = np.zeros([T, F, A], dtype=complex)
    t2 = sage_const['doppler_candidates'][:, latent_index[2]]
    for kk in range(np.size(doppler_matrix, 2)):
        for jj in range(np.size(doppler_matrix, 1)):
            doppler_matrix[:, jj, kk] = t2

    latent_signal = np.multiply(np.multiply(np.multiply(latent_parameter[3], tof_matrix), aoa_matrix), doppler_matrix)
    return latent_signal


def sage_expectation(csi_signal, latent_signal, expect_index, update_ratio):
    noise_signal = csi_signal - np.sum(latent_signal, 3)
    expect_signal = latent_signal[:, :, :, expect_index] + update_ratio * noise_signal
    return expect_signal


def sage_maximization(latent_signal, latent_parameter, latent_index, sage_const):
    T = np.size(latent_signal, 0)
    F = np.size(latent_signal, 1)
    A = np.size(latent_signal, 2)
    doppler_range = sage_const['doppler_range']
    aoa_range = sage_const['aoa_range']
    tof_range = sage_const['tof_range']
    latent_parameter_temp = copy.copy(latent_parameter)
    latent_index_temp = copy.copy(latent_index)

    aoa_matrix = np.transpose(np.tile(np.transpose(np.tile(sage_const['aoa_candidates'][:, latent_index_temp[1]], T).reshape(T, A)), F).reshape(3, T, F), [1, 2, 0])
    doppler_matrix = np.tile(np.transpose(np.tile(sage_const['doppler_candidates'][:, latent_index_temp[2]], F).reshape(F, T)), A).reshape(T, F, A)

    coeff_matrix = np.multiply(np.multiply(latent_signal, np.conj(aoa_matrix)), np.conj(doppler_matrix))
    coeff_vector = np.reshape(np.sum(np.sum(coeff_matrix, 0), 1), [F, 1], order='F')
    coeff_vector = np.tile(coeff_vector, [1, len(tof_range)])
    object_vector = np.abs(np.sum(np.multiply(coeff_vector, np.conj(sage_const['tof_candidates'])), 0))
    latent_index_temp[0] = int(np.where(object_vector == np.max(object_vector))[0])
    latent_parameter_temp[0] = tof_range[0, latent_index_temp[0]]

    tof_matrix = np.transpose(np.tile(np.transpose(np.tile(sage_const['tof_candidates'][:, latent_index_temp[0]], T).reshape(T, F)), A).reshape(F, T, A), [1, 0, 2])

    coeff_matrix = np.multiply(np.multiply(latent_signal, np.conj(doppler_matrix)), np.conj(tof_matrix))
    coeff_vector = np.reshape(np.sum(np.sum(coeff_matrix, 0), 0), [A, 1], order='F')
    coeff_vector = np.tile(coeff_vector, [1, len(aoa_range)])
    object_vector = np.abs(np.sum(np.multiply(coeff_vector, np.conj(sage_const['aoa_candidates'])), 0))
    latent_index_temp[1] = int(np.where(object_vector == np.max(object_vector))[0][0])
    latent_parameter_temp[1] = aoa_range[0, latent_index_temp[1]]

    aoa_matrix = np.transpose(np.tile(np.transpose(np.tile(sage_const['aoa_candidates'][:, latent_index_temp[1]], T).reshape(T, A)), F).reshape(3, T, F), [1, 2, 0])

    coeff_matrix = np.multiply(np.multiply(latent_signal, np.conj(aoa_matrix)), np.conj(tof_matrix))
    coeff_vector = np.reshape(np.sum(np.sum(coeff_matrix, 1), 1), [T, 1], order='F')
    coeff_vector = np.tile(coeff_vector, [1, len(doppler_range)])
    object_vector = np.abs(np.sum(np.multiply(coeff_vector, np.conj(sage_const['doppler_candidates'])), 0))
    latent_index_temp[2] = int(np.where(object_vector == np.max(object_vector))[0])
    latent_parameter_temp[2] = doppler_range[0, latent_index_temp[2]]

    doppler_matrix = np.tile(np.transpose(np.tile(sage_const['doppler_candidates'][:, latent_index_temp[2]], F).reshape(F, T)), A).reshape(T, F, A)

    coeff_matrix = np.multiply(np.multiply(np.multiply(np.conj(aoa_matrix), np.conj(tof_matrix)), np.conj(doppler_matrix)), latent_signal)
    latent_parameter_temp[3] = np.sum(coeff_matrix) / (T * F * A)
    return latent_parameter_temp, latent_index_temp


def sage_sfg(csi_signal, initial_parameter, initial_index, sage_const, threshold, qualified_num, bigger_num):
    T = sage_const['T']
    L = sage_const['L']
    F = sage_const['F']
    A = sage_const['A']
    N = sage_const['N']
    update_ratio = sage_const['update_ratio']
    latent_signal = np.zeros([T, F, A, L], dtype=complex)

    for ii in range(L):
        if initial_parameter[3, ii] != 0:
            latent_signal[:, :, :, ii] = sage_signal(initial_parameter[:, ii], initial_index[:, ii], sage_const)  #不优化

    final_parameter = copy.copy(initial_parameter)
    temp_parameter = copy.copy(initial_parameter)
    temp_index = copy.copy(initial_index)
    final_index = copy.copy(initial_index)
    parameter_diff_temp = [0, 0, 0, 0]
    for ii in range(N):
        # 一个循环0.01s
        for jj in range(L):
            t3 = time.time()
            temp_signal = sage_expectation(csi_signal, latent_signal, jj, update_ratio) # 正确
            t4 = time.time()
            # print('sage_expectation:', t4 - t3)
            temp_parameter[:, jj], temp_index[:, jj] = sage_maximization(temp_signal, final_parameter[:, jj], final_index[:, jj], sage_const) #正确
            t3 = time.time()
            # print('sage_maximization:', t3 - t4)
            latent_signal[:, :, :, jj] = sage_signal(temp_parameter[:, jj], temp_index[:, jj], sage_const)

        parameter_diff = np.sqrt(np.sum(np.abs(temp_parameter - final_parameter) ** 2, 1))
        parameter_diff_temp = copy.copy(parameter_diff)
        final_parameter = copy.copy(temp_parameter)
        final_index = copy.copy(temp_index)

        # 若下一次循环>K的指标比前一个误差更大，则跳出循环
        if np.sum(threshold >= parameter_diff) >= qualified_num or np.sum(parameter_diff_temp < parameter_diff) >= bigger_num:
            break
        # if parameter_diff[0] < 1e-9 and parameter_diff[1] < 1 / 180 * np.pi and parameter_diff[2] < 1 and parameter_diff[3] < 1e-9:
        #     break

    residue_error = csi_signal - np.sum(latent_signal, 3)
    residue_error = np.mean(np.abs(residue_error)) / np.mean(np.abs(csi_signal))
    return final_parameter, residue_error


def sage_main(csi_data, G, sage_const, exact, qualified_num, bigger_num):
    T = sage_const['T']
    L = sage_const['L']
    F = sage_const['F']
    A = sage_const['A']
    G = np.min([G, np.floor(np.size(csi_data, 0)/T)]).astype(int)  # 分段长度
    GI = np.arange(0, (G-1)*T+1, T)
    estimated_parameter = np.zeros([4, L, G], dtype=complex)
    initial_parameter = np.zeros([4, L], dtype=complex)
    initial_index = np.zeros([3, L], dtype=int)
    initial_index[0, :] = int(round((0 - sage_const['tof_range'][0, 0]) / (sage_const['tof_range'][0, 1] - sage_const['tof_range'][0, 0])))
    initial_index[1, :] = int(round((0 - sage_const['aoa_range'][0, 0]) / (sage_const['aoa_range'][0, 1] - sage_const['aoa_range'][0, 0])))
    initial_index[2, :] = int(round((0 - sage_const['doppler_range'][0, 0]) / (sage_const['doppler_range'][0, 1] - sage_const['doppler_range'][0, 0])))
    residue_errors = np.zeros([1, G], dtype=complex)

    print('SAGE Main...')
    for ii in range(G):
        print(ii + 1)
        csi_signal = np.transpose(csi_data[GI[ii] + np.arange(0, T), :])
        csi_signal = np.transpose(np.reshape(csi_signal, [F, A, T], order='F'), [2, 0, 1])
        estimated_parameter[:, :, ii], residue_errors[:, ii] = sage_sfg(csi_signal, initial_parameter, initial_index, sage_const, exact, qualified_num, bigger_num)

    return estimated_parameter


def sage_optimal_initialization(G, L): # G=6,L=2
    prob = pic.Problem()
    variables = prob.add_variable('np.variables', G*G*L*L, vtype='binary')
    IJ = []
    IJK = []
    PQ = []

    for i in range(G): #G*L
        for j in range(L):
            IJ.append((i, j))

    for i in range(G): #G*L
        for j in range(G):
            for k in range(L):
                IJK.append((i, j, k))

    for j in range(G):
        for l in range(L):
            PQ.append((j, l))

    # M[i,j,k,l] = M[G*L*L*ii + L*L*jj + L*kk + ll]
    # constraints = [constraints np.variables(ii,ii,jj,jj) == 1]
    prob.add_list_of_constraints([variables[G*L*L*ii + L*L*ii + L*jj + jj] == 1 for (ii, jj) in IJ])

    for ii in range(G):
        for jj in range(L):
            # constraints = [constraints np.squeeze(np.variables(ii,:,jj,:)) == np.squeeze(np.variables(:,ii,:,jj))];
            prob.add_list_of_constraints([variables[G*L*L*ii + L*L*kk + L*jj + ll] == variables[G*L*L*kk + L*L*ii + L*ll + jj] for (kk, ll) in IJ])

    # constraints = [constraints np.sum(np.sum(np.sum(np.variables, 2), 3), 4) == G * L]
    prob.add_list_of_constraints([pic.sum(variables[G*L*L*ii: G*L*L*(ii+1)]) == G*L for ii in range(G)])

    # constraints = [constraints np.sum(np.variables,4) == 1]
    prob.add_list_of_constraints([pic.sum(variables[G*L*L*ii + L*L*jj + L*kk: G*L*L*ii + L*L*jj + L*(kk+1)]) == 1 for (ii, jj, kk) in IJK])

    for ii in range(G):
        for jj in range(G):
            if ii == jj:
                continue
            for mm in range(L):
                for nn in range(L):
                    # constraints = [constraints np.variables(ii,jj,mm,nn) + np.squeeze(np.variables(jj,:,nn,:)) <= 1 + np.squeeze(np.variables(ii,:,mm,:))];
                    prob.add_list_of_constraints([variables[G*L*L*ii + L*L*jj + L*mm + nn] + variables[G*L*L*jj + L*L*pp + L*nn + qq] <= \
                                                  1 + variables[G*L*L*ii + L*L*pp + L*mm + qq] for (pp, qq) in PQ])

    return prob, variables


def sage_optimal_matching(estimated_parameter, optimize_variables, optimize_problem):
    L = np.size(estimated_parameter, 1)
    G = np.size(estimated_parameter, 2)
    estimated_parameter_temp = copy.copy(estimated_parameter)

    cost_matrix = np.ones([G, G, L, L]) * 10000
    temp_parameter = np.zeros([np.size(estimated_parameter, 0), L], dtype=complex)
    for ii in range(G):
        for kk in range(L):
            cost_matrix[ii, ii, kk, kk] = 0
            for qq in range(L):
                    temp_parameter[:, qq] = copy.copy(estimated_parameter[:, kk, ii])
            for jj in range(ii+1, G):
                cost_matrix[ii, jj, kk, :] = np.sqrt(np.sum(np.square(np.abs(temp_parameter - estimated_parameter[:, :, jj])), 0))
                cost_matrix[jj, ii, :, kk] = np.sqrt(np.sum(np.square(np.abs(temp_parameter - estimated_parameter[:, :, jj])), 0))

    cost_matrix_pic = pic.new_param('cost_matrix', np.reshape(cost_matrix, [G*G*L*L, 1]))
    optimize_problem.set_objective('min', cost_matrix_pic | optimize_variables)
    optimize_problem.solve(verbose=0)
    # print(optimize_problem)

    edges = optimize_variables.value
    edges = [0 if x<=1e-5 else 1 for x in edges]
    estimated_index = np.zeros([L, G], dtype=int)
    estimated_index[:, 0] = np.arange(0, L)
    for ii in range(1, G):
        for jj in range(L):
            estimated_index[jj, ii] = np.flatnonzero(edges[L*L*ii + L*jj: L*L*ii + L*(jj+1)])  # 找到非0索引
        estimated_parameter_temp[:, :, ii] = estimated_parameter_temp[:, estimated_index[:, ii], ii]
    return estimated_parameter_temp, estimated_index


def hungary(task_matrix):
    b = task_matrix.copy()
    # 行和列减0
    for i in range(len(b)):
        row_min = min(b[i])
        for j in range(len(b[i])):
            b[i][j] -= row_min
    for i in range(len(b[0])):
        col_min = min(b[:, i])
        for j in range(len(b)):
            b[j][i] -= col_min
    line_count = 0
    # 线数目小于矩阵长度时，进行循环
    while (line_count < len(b)):
        line_count = 0
        row_zero_count = []
        col_zero_count = []
        for i in range(len(b)):
            row_zero_count.append(np.sum(b[i] == 0))
        for i in range(len(b[0])):
            col_zero_count.append((np.sum(b[:, i] == 0)))
        # 划线的顺序（分行或列）
        line_order = []
        row_or_col = []
        for i in range(len(b[0]), 0, -1):
            while (i in row_zero_count):
                line_order.append(row_zero_count.index(i))
                row_or_col.append(0)
                row_zero_count[row_zero_count.index(i)] = 0
            while (i in col_zero_count):
                line_order.append(col_zero_count.index(i))
                row_or_col.append(1)
                col_zero_count[col_zero_count.index(i)] = 0
        # 画线覆盖0，并得到行减最小值，列加最小值后的矩阵
        count_of_row = []
        count_of_rol = []
        row_and_col = [i for i in range(len(b))]
        for i in range(len(line_order)):
            if row_or_col[i] == 0:
                count_of_row.append(line_order[i])
            else:
                count_of_rol.append(line_order[i])
            c = np.delete(b, count_of_row, axis=0)
            c = np.delete(c, count_of_rol, axis=1)
            line_count = len(count_of_row) + len(count_of_rol)
            # 线数目等于矩阵长度时，跳出
            if line_count == len(b):
                break
            # 判断是否画线覆盖所有0，若覆盖，进行加减操作
            if 0 not in c:
                row_sub = list(set(row_and_col) - set(count_of_row))
                min_value = np.min(c)
                for i in row_sub:
                    b[i] = b[i] - min_value
                for i in count_of_rol:
                    b[:, i] = b[:, i] + min_value
                break
    row_ind, col_ind = linear_sum_assignment(b)
    min_cost = task_matrix[row_ind, col_ind].sum()
    best_solution = list(task_matrix[row_ind, col_ind])
    return best_solution, col_ind, min_cost


def sage_greedy_matching(estimated_parameter, reference_parameter):
    G = np.size(reference_parameter, 2)
    L = np.size(estimated_parameter, 1)
    cost_matrix = np.zeros([L, L])

    for ii in range(G):
        for jj in range(L):
            mat = np.transpose([reference_parameter[:, jj, ii] for q in range(L)])
            cost_matrix[jj, :] = cost_matrix[jj, :] + np.sqrt(np.sum(np.square(np.abs(mat - estimated_parameter)), 0))

    estimated_path, estimated_index, estimated_cost = hungary(cost_matrix)# 匈牙利算法
    # estimated_path = estimated_parameter[:, estimated_index]
    return estimated_index


def sage_path_mapping(estimated_parameter, parameter_weight, N, S):
    K = np.size(estimated_parameter, 0)
    G = np.size(estimated_parameter, 2)
    L = np.size(estimated_parameter, 1)
    normalized_parameter = copy.copy(estimated_parameter)
    normalized_parameter[3, :, :] = np.abs(normalized_parameter[3, :, :])
    group_parameter = np.zeros([np.size(normalized_parameter, 0), L])

    for ii in range(0, G):
        normalized_parameter[3, :, ii] = normalized_parameter[3, :, ii] / np.sum(normalized_parameter[3, :, ii])

    matrix_weight = np.zeros([K, L, G], dtype=complex)
    for kk in range(np.size(matrix_weight, 2)):
        for jj in range(np.size(matrix_weight, 1)):
            matrix_weight[:, jj, kk] = copy.copy(parameter_weight)

    normalized_parameter = np.multiply(normalized_parameter, matrix_weight)
    M = int(np.floor((G - 1) / (N - 1)))
    optimal_problem, optimal_variables = sage_optimal_initialization(int((N + 1) / 2), int(np.size(estimated_parameter, 1)))  # 固定

    for ii in range(M):
        if ii == 0:
            col_index = ii * (N - 1) + np.arange(0, N, 2)
            k, estimated_index = sage_optimal_matching(normalized_parameter[:, :, col_index], optimal_variables, optimal_problem)
        else:
            col_index = ii * (N - 1) + np.arange(2, N, 2)
            temp_parameter = np.zeros([np.size(normalized_parameter, 0), L, len(col_index)+1], dtype=complex)
            temp_parameter[:, :, 0] = group_parameter
            temp_parameter[:, :, 1:] = normalized_parameter[:, :, col_index]
            k, estimated_index = sage_optimal_matching(temp_parameter, optimal_variables, optimal_problem)

        for jj in range(len(col_index)):
            normalized_parameter[:, :, col_index[jj]] = normalized_parameter[:, estimated_index[:, jj], col_index[jj]]
            estimated_parameter[:, :, col_index[jj]] = estimated_parameter[:, estimated_index[:, jj], col_index[jj]]

        col_index = ii * (N - 1) + np.arange(1, N, 2)
        for jj in range(len(col_index)):
            kk = col_index[jj]
            span_index = np.arange(np.max([0, kk - (2 * S - 1)]), min([col_index[len(col_index) - 1], kk + (2 * S - 1)]))
            estimated_index = sage_greedy_matching(normalized_parameter[:, :, kk], normalized_parameter[:, :, span_index])
            normalized_parameter[:, :, kk] = normalized_parameter[:, estimated_index, kk]

        col_index = ii * (N-1) + np.arange(N)
        group_parameter = np.median(normalized_parameter[:, :, col_index], 2)

        for ii in range(M*(N-1), G):
            span_index = np.arange(ii-S+1, ii)
            estimated_index = sage_greedy_matching(normalized_parameter[:, :, ii], normalized_parameter[:, :, span_index])
            normalized_parameter[:, :, ii] = normalized_parameter[:, estimated_index, ii]
            estimated_parameter[:, :, ii] = estimated_parameter[:, estimated_index, ii]

    return estimated_parameter


def sage_path_mapping_filter(estimated_parameter, parameter_weight, N, S):
    G = np.size(estimated_parameter, 2)
    L = np.size(estimated_parameter, 1)
    estimated_parameter_temp = copy.copy(estimated_parameter)

    for ii in range(0, G):
        si = np.argsort(-np.abs(estimated_parameter[3, :, ii]))
        estimated_parameter_temp[:, :, ii] = estimated_parameter_temp[:, si, ii]

    signal_power = np.mean(np.squeeze(np.abs(estimated_parameter_temp[3, :, :])), 1)
    power_thres = signal_power[0] * 0.7
    di = signal_power > power_thres

    if np.sum(di) == 1:
        di[1] = 1

    estimated_parameter_temp = estimated_parameter_temp[:, di, :]
    estimated_parameter_temp = sage_path_mapping(estimated_parameter_temp, parameter_weight, N, S)
    return estimated_parameter_temp


def hampel(X, k=3, nsigma=3):
    length = X.shape[0] - 1
    iLo = np.array([i - k for i in range(0, length + 1)])
    iHi = np.array([i + k for i in range(0, length + 1)])
    iLo[iLo < 0] = 0
    iHi[iHi > length] = length
    xmad = []
    xmedian = []
    for i in range(length + 1):
        w = X[iLo[i]:iHi[i] + 1]
        medj = np.median(w)
        mad = np.median(np.abs(w - medj))
        xmad.append(mad)
        xmedian.append(medj)
    xmad = np.array(xmad)
    xmedian = np.array(xmedian)
    scale = 1.4826  # 缩放
    xsigma = scale * xmad
    xi = ~(np.abs(X - xmedian) <= nsigma * xsigma)  # 找出离群点（即超过nsigma个标准差）

    # 将离群点替换为中为数值
    xf = X.copy()
    xf[xi] = xmedian[xi]
    return xf


def sage_tof_filter(z, u, dt, comp): # z:tof, u:doppler, dt不变
    xt = np.zeros(np.size(z))
    xt1 = np.zeros(np.size(z))
    pt = np.zeros(np.size(z))
    pt1 = np.zeros(np.size(z))
    R = np.var(z[0: int(np.floor(2 / dt))])
    Q = np.var(u[0: int(np.floor(2 / dt))])
    xt[0] = copy.copy(z[0])  # tof
    xt1[0] = copy.copy(z[0]) # tof
    pt[0] = copy.copy(R)
    B = copy.copy(dt)

    for ii in range (1, np.size(z)):
        xt1[ii] = xt[ii - 1] + B * u[ii - 1]
        pt1[ii] = pt[ii - 1] + Q
        K = pt1[ii] / (pt1[ii] + R)
        xt[ii] = xt1[ii] + K * (z[ii] - xt1[ii])
        pt[ii] = pt1[ii] - K * pt1[ii]

    x = copy.copy(xt)
    p = copy.copy(pt)
    for ii in range(np.size(z) - 2, -1, -1):
        L = pt[ii] / pt1[ii + 1]
        x[ii] = xt[ii] + L * (x[ii + 1] - xt1[ii + 1])
        p[ii] = pt[ii] + L * (p[ii + 1] - pt1[ii + 1]) * L

    return x, p, z[len(z)-comp-1:], u[len(u)-comp-1:]


def sage_tof_filter2(z, u, dt, xt0, pt0, comp): # z:tof, u:doppler, dt不变
    if np.array(xt0).any() == 0:
        x, p, xt, pt = sage_tof_filter(z, u, dt, comp)
        return x, p, xt, pt
    else:
        xt = np.zeros(np.size(z) + np.size(xt0), dtype=complex)
        xt1 = np.zeros(np.size(z) + np.size(xt0), dtype=complex)
        pt = np.zeros(np.size(z) + np.size(pt0), dtype=complex)
        pt1 = np.zeros(np.size(z) + np.size(pt0), dtype=complex)
        z_temp = list(xt0)
        z_temp.extend(z)
        u_temp = list(pt0)
        u_temp.extend(u)
        R = np.var(z_temp[0: int(np.floor(2 / dt)) + 1]) #dt=sample_rate/T, 2/dt=2T/sample_rate
        Q = np.var(u_temp[0: int(np.floor(2 / dt)) + 1])
        xt[0: len(xt0)] = copy.copy(xt0)  # tof 初始条件
        pt[0: len(pt0)] = copy.copy(R)
        B = copy.copy(dt) # 不变

        for ii in range (1, np.size(z_temp)):
            xt1[ii] = xt[ii - 1] + B * u_temp[ii - 1]
            pt1[ii] = pt[ii - 1] + Q
            K = pt1[ii] / (pt1[ii] + R)
            xt[ii] = xt1[ii] + K * (z_temp[ii] - xt1[ii])
            pt[ii] = pt1[ii] - K * pt1[ii]
        x = copy.copy(xt)
        p = copy.copy(pt)
        for ii in range(np.size(z_temp) - 2, -1, -1):
            L = pt[ii] / pt1[ii + 1]
            x[ii] = xt[ii] + L * (x[ii + 1] - xt1[ii + 1])
            p[ii] = pt[ii] + L * (p[ii + 1] - pt1[ii + 1]) * L

        return x[comp+1:], p[comp+1:], z[len(z)-comp-1:], u[len(u)-comp-1:]


def sage_localization(signal_range, aoa, rx_loc, xb, yb):
    target_loc = np.zeros(([2, np.size(signal_range, 0)]))
    mxb = np.mean(xb)
    myb = np.mean(yb)

    for ii in range(np.size(signal_range, 0)):
        current_loc = np.zeros([2, 1])
        for jj in range(np.size(rx_loc, 1)):
            if np.sum([1 if x == rx_loc[0, jj] else 0 for x in xb]) < 0:
                s = np.sign((mxb - rx_loc[0, jj]) * np.cos(aoa[ii, jj]))
                current_loc[0, jj] = (signal_range[ii, jj] ** 2 * np.cos(aoa[ii, jj]) ** 2 + 2 * s * signal_range[ii, jj] * rx_loc[0, jj] * np.cos(aoa[ii, jj]) + rx_loc[0, jj] ** 2 - (np.sin(aoa[ii, jj]) * rx_loc[0, jj] - np.cos(aoa[ii, jj]) * rx_loc[1, jj]) ** 2) \
                    / 2 / (np.cos(aoa[ii, jj]) ** 2 * rx_loc[0, jj] + np.sin(aoa[ii, jj]) * np.cos(aoa[ii, jj]) * rx_loc[1, jj] + signal_range[ii, jj] * s * np.cos(aoa[ii, jj]))
                current_loc[1, jj] = np.tan(aoa[ii, jj]) * (current_loc[0, jj] - rx_loc[0, jj]) + rx_loc[1, jj]
            else:
                s = np.sign((myb - rx_loc[1, jj]) * np.sin(aoa[ii, jj]))
                current_loc[1, jj] = (signal_range[ii, jj] ** 2 * np.sin(aoa[ii, jj]) ** 2 + 2 * s * signal_range[ii, jj] * rx_loc[1, jj] * np.sin(aoa[ii, jj]) + rx_loc[1, jj] ** 2 - (np.cos(aoa[ii, jj]) * rx_loc[1, jj] - np.sin(aoa[ii, jj]) * rx_loc[0, jj]) ** 2) \
                    / 2 / (np.sin(aoa[ii, jj]) ** 2 * rx_loc[1, jj] + np.sin(aoa[ii, jj]) * np.cos(aoa[ii, jj]) * rx_loc[0, jj] + signal_range[ii, jj] * s * np.sin(aoa[ii, jj]))
                current_loc[0, jj] = 1 / np.tan(aoa[ii, jj]) * (current_loc[1, jj] - rx_loc[1, jj]) + rx_loc[0, jj]
        target_loc[:, ii] = current_loc.flatten()
    return target_loc


def dynamic_pre_buffer_build():
    dynamic_pre_buffer = {
        'pre_time_stamp': np.transpose([[]]),
        'pre_csi_mult': np.transpose([[] for i in range(90)]),
        'pre_rfl_path': [[] for i in range(4)],
        'pre_signal': [],
        'pre_aoa': [],
    }
    return dynamic_pre_buffer


def plan_A(orient):
    T = 100
    F = 30
    A = 3
    L = 5
    N = 100
    FI = 2 * 312.5e3
    sample_rate = 1000
    TI = 1 / sample_rate
    AS = 0.5
    TR = np.arange(-100, 401, 1).reshape([1, 501]) * 1e-9
    AR = np.arange(0, 181, 1).reshape([1, 181]) / 180 * np.pi
    DR = np.arange(-80, 81, 1).reshape([1, 161])
    UR = 1

    sage_const = sage_const_build(T, F, A, L, N, FI, TI, AS, TR, AR, DR, UR)
    sage_const = sage_generate_steering_vector(sage_const)

    MN = round(sample_rate / T) + 1
    MS = 3
    parameter_weight = np.array([1e9, 180 / np.pi, 1, 1]).reshape([1, 4]) * np.array(
        [1 / 200, 1 / 90, 1 / 80, 10]).reshape([1, 4])

    cdata = scio.loadmat(data_path + trace_file)
    ccsi_data = cdata['csi_data']
    ctime_stamp = cdata['time_stamp']
    ############################################################################################################################
    cdata = scio.loadmat(data_path + trace_file)
    ground_truth = cdata['ground_truth']
    cdata = scio.loadmat(data_path + 'location-' + trace_file)
    location_truth = cdata['location']
    ddata = scio.loadmat(data_path + 'PMM-' + trace_file)
    estimated_path_truth = ddata['estimated_path']
    rfl_matrix_truth = np.sum(np.abs(estimated_path_truth[3, :, :]), 1)
    rfl_index_truth = int(np.where(rfl_matrix_truth == np.max(rfl_matrix_truth))[0])
    # rfl_index_truth = rfl_matrix_truth.index(np.max(rfl_matrix_truth))
    rfl_path_truth = np.squeeze(estimated_path_truth[:, rfl_index_truth, :])
    motion_index_truth = np.arange(10, np.size(rfl_path_truth, 1) - 10)
    window = 5
    aoa1 = hampel(rfl_path_truth[1, motion_index_truth], window, np.pi / 180)  # 滤波器
    aoa1 = signal.savgol_filter(aoa1, window, polyorder=2)  # 代替smooth # 滤波器
    tof1 = hampel(rfl_path_truth[0, motion_index_truth], window, 2e-9)  # 滤波器
    doppler1 = hampel(rfl_path_truth[2, motion_index_truth], window, 6)  # 滤波器
    doppler1 = signal.savgol_filter(doppler1, window, polyorder=2)  # 代替smooth # 滤波器
    x = np.arange(0, len(tof1))
    plt.subplot(2, 3, 2)
    plt.plot(x, tof1)
    plt.subplot(2, 3, 3)
    plt.plot(x, aoa1)
    plt.subplot(2, 3, 4)
    plt.plot(x, doppler1)
    signal_range1, pp, xt, pt = sage_tof_filter(tof1 * c, -doppler1 * c / carrier_frequency, T / sample_rate, 2)
    plt.subplot(2, 3, 5)
    plt.plot(signal_range1)
    plt.subplot(2, 3, 6)
    plt.plot(ground_truth[0, :], ground_truth[1, :], 'r--')
    plt.plot(location_truth[0, :], location_truth[1, :], 'g--')
    ############################################################################################################################

    step = 1000
    count = 0
    csi_data_cut = 50
    csi_data_len = np.size(ccsi_data, 0)
    pre_time_stamp = np.transpose([[]])
    pre_time_stamp_len = 100
    pre_csi_mult = np.transpose([[] for i in range(np.size(ccsi_data, 1))])
    pre_csi_mult_len = pre_time_stamp_len
    pre_rfl_path = [[] for i in range(4)]
    pre_rfl_path_len = 10
    window = 5  # =5
    para_cut = 2
    comp = pre_rfl_path_len - para_cut
    xt0 = [0]
    pt0 = [0]
    pre_signal = []
    pre_aoa = []
    comp_len = 0
    pre_signal_len = 10
    signal_range_cut = 2
    x = np.arange(0, csi_data_len)
    plt.figure(1)

    while count * step <= csi_data_len:
        csi_data = ccsi_data[step * count: step * (count + 1)]  # 分段读取CSI

        csi_amplitude = np.mean(np.abs(csi_data), axis=0)
        csi_variance = np.sqrt([np.var(np.abs(csi_data[:, i])) for i in range(np.size(csi_data, 1))])
        csi_ratio = np.divide(csi_amplitude, csi_variance)
        ant_ratio = np.mean(np.reshape(csi_ratio, [F, A]), axis=0)
        midx = int(np.where(ant_ratio == np.max(ant_ratio))[0])
        # midx = ant_ratio.index(np.max(ant_ratio))
        csi_ref = np.zeros([np.size(csi_data, 0), F * A], dtype=complex)
        for qq in range(A):
            csi_ref[:, F * qq: F * (qq + 1)] = copy.copy(csi_data[:, midx * F + np.arange(F)])

        alpha_all = 0
        for jj in range(np.size(csi_data, 1)):
            alpha = min(np.abs(csi_data[:, jj]))
            alpha_all = alpha_all + alpha
            csi_data[:, jj] = np.multiply(np.abs(csi_data[:, jj]) - alpha, np.exp(1j * np.angle(csi_data[:, jj])))

        beta = alpha_all / np.size(csi_data, 1) * 1000
        for jj in range(np.size(csi_ref, 1)):
            csi_ref[:, jj] = np.multiply(np.abs(csi_ref[:, jj]) + beta, np.exp(1j * np.angle(csi_ref[:, jj])))
        csi_mult = np.multiply(csi_data, np.conj(csi_ref))
        ####################################################################################################################################
        plt.subplot(2, 3, 1)
        hlfrt = sample_rate / 2
        upper_order = 6
        upper_stop = 80
        lower_order = 3
        lower_stop = 2
        b, a = signal.butter(5, [lower_stop / hlfrt, upper_stop / hlfrt], 'bandpass')
        csi_mult_ex = np.concatenate((pre_csi_mult, csi_mult))
        csi_filter = np.zeros(np.shape(csi_mult_ex), dtype=complex)
        for kk in range(np.size(csi_mult_ex, 1)):
            csi_filter[:, kk] = signal.filtfilt(b, a, csi_mult_ex[:, kk])  # 直接采取前后截断
        if count != 0:
            csi_filter_final = csi_filter[pre_csi_mult_len - csi_data_cut: -csi_data_cut]
            plt.plot(x[
                     count * step - pre_csi_mult_len + csi_data_cut: count * step - pre_csi_mult_len + csi_data_cut + len(
                         csi_filter_final)], csi_filter_final[:, 1])
        else:
            csi_filter_final = csi_filter[0: -csi_data_cut]
            plt.plot(x[0: len(csi_filter_final)], csi_filter_final[:, 1])

        pre_csi_mult = csi_mult[-pre_csi_mult_len - 1: -1]
        plt.show()
        ####################################################################################################################################
        time_stamp = ctime_stamp[step * count: step * (count + 1)]  # 临时使用
        time_stamp_ex = np.concatenate((pre_time_stamp, time_stamp))
        pre_time_stamp = time_stamp[-pre_time_stamp_len:]
        time_stamp_diff = list(np.diff(time_stamp_ex, axis=0))
        time_wrap_index = [time_stamp_diff.index(ii) for ii in time_stamp_diff if ii < 0]  # 暂时不做修改
        if count != 0:
            time_stamp_ex = np.transpose((time_stamp_ex - time_stamp_ex[0, 0]) / 1e6).flatten()[
                            pre_time_stamp_len - csi_data_cut: -csi_data_cut]
        else:
            time_stamp_ex = np.transpose((time_stamp_ex - time_stamp_ex[0, 0]) / 1e6).flatten()[0: -csi_data_cut]
        interp_stamp = np.arange(np.floor(time_stamp_ex[-1] * sample_rate) / sample_rate)
        csi_interp = np.zeros([len(time_stamp_ex), np.size(csi_filter_final, 1)], dtype=complex)
        xnew = np.arange(len(time_stamp_ex))
        for jj in range(np.size(csi_filter_final, 1)):
            k = interpolate.interp1d(time_stamp_ex, csi_filter_final[:, jj], kind='linear')  # 插值
            csi_interp[:, jj] = k(time_stamp_ex[xnew])  # ？？？？？
        ###############################################################################################################################
        t1 = time.time()
        estimated_parameter = sage_main(csi_interp, 1000, sage_const, [1e-9, 1 / 180 * np.pi, 1, 1e-9], 3, 1)
        print('SAGE用时：', time.time() - t1)
        print('开始匹配路径')
        t1 = time.time()
        estimated_path = sage_path_mapping_filter(estimated_parameter, parameter_weight, MN, MS)
        print('用时：', time.time() - t1)
        ####################################################################################################################################
        rfl_matrix = np.sum(np.abs(estimated_path[3, :, :]), 1)  # 转换速度慢
        rfl_index = int(np.where(rfl_matrix == np.max(rfl_matrix))[0])
        # rfl_index = rfl_matrix.index(np.max(rfl_matrix))  # =0，可只运行一次
        rfl_path = np.squeeze(estimated_path[:, rfl_index, :])
        rfl_path_ex = np.concatenate((pre_rfl_path, rfl_path), axis=1)
        motion_index = np.arange(np.size(rfl_path_ex, 1))  ##??????

        if count != 0:
            tof = hampel(rfl_path_ex[0, motion_index], window, 2e-9)[comp: -para_cut]
            aoa = hampel(rfl_path_ex[1, motion_index], window, np.pi / 180)[comp: -para_cut]
            aoa = signal.savgol_filter(aoa, window, polyorder=2).squeeze()  # 代替smooth
            doppler = hampel(rfl_path_ex[2, motion_index], window, 6)[comp: -para_cut]
            doppler = signal.savgol_filter(doppler, window, polyorder=2).squeeze()
            signal_range, pp, xt0, pt0 = sage_tof_filter2(tof * c, -doppler * c / carrier_frequency, T / sample_rate,
                                                          xt0, pt0, 5)

            # power = rfl_path_ex[3, motion_index]
            plt.subplot(2, 3, 2)
            plt.plot(x[count * int(step / T) - para_cut - 1: count * int(step / T) - para_cut - 1 + len(tof)], tof)
            plt.subplot(2, 3, 3)
            plt.plot(x[count * int(step / T) - para_cut - 1: count * int(step / T) - para_cut - 1 + len(aoa)], aoa)
            plt.subplot(2, 3, 4)
            plt.plot(x[count * int(step / T) - para_cut - 1: count * int(step / T) - para_cut - 1 + len(doppler)],
                     doppler)
            plt.subplot(2, 3, 5)
            plt.plot(x[count * int(step / T) - para_cut - 1: count * int(step / T) - para_cut - 1 + len(signal_range)],
                     signal_range)
        else:
            tof = hampel(rfl_path_ex[0, motion_index], window, 2e-9)[: -para_cut]
            aoa = hampel(rfl_path_ex[1, motion_index], window, np.pi / 180)[: -para_cut]
            aoa = signal.savgol_filter(aoa, window, polyorder=2).squeeze()  # 代替smooth
            doppler = hampel(rfl_path_ex[2, motion_index], window, 6)[: -para_cut]
            doppler = signal.savgol_filter(doppler, window, polyorder=2).squeeze()
            signal_range, pp, xt0, pt0 = sage_tof_filter2(tof * c, -doppler * c / carrier_frequency, T / sample_rate,
                                                          xt0, pt0, 5)

            plt.subplot(2, 3, 2)
            plt.plot(x[: len(tof)], tof)
            plt.subplot(2, 3, 3)
            plt.plot(x[: len(tof)], aoa)
            plt.subplot(2, 3, 4)
            plt.plot(x[: len(tof)], doppler)
            plt.subplot(2, 3, 5)
            plt.plot(x[: len(signal_range)], signal_range)
        pre_rfl_path = rfl_path[:, -pre_rfl_path_len:]
        ##########################################################################################################################
        # 卡尔曼滤波器
        signal_range[signal_range <= 0.3] = 0.3
        signal_range = signal.savgol_filter(signal_range, window, polyorder=2)
        signal_range = signal_range + np.linalg.norm(rx_loc - tx_loc)
        aoa = orient - aoa

        signal_range_ex = np.concatenate((pre_signal, signal_range))
        aoa_ex = np.concatenate((pre_aoa, np.squeeze(aoa)))
        location = sage_localization(signal_range_ex.reshape([np.size(signal_range_ex, 0), 1]),
                                     aoa_ex.reshape([np.size(aoa_ex, 0), 1]), rx_loc, xb, yb)[:,
                   comp_len: -signal_range_cut]
        pre_signal = signal_range_ex[-pre_signal_len - 1: -1]
        pre_aoa = aoa_ex[-pre_signal_len - 1: -1]
        comp_len = pre_signal_len - signal_range_cut

        for ii in range(np.size(location, 0)):
            location[ii, :] = signal.savgol_filter(location[ii, :], window, polyorder=2)  # 滤波器

        plt.subplot(2, 3, 6)
        # plt.plot(ground_truth[0, :], ground_truth[1, :], 'r--')
        plt.plot(location[0, :], location[1, :], 'o--')

        plt.show()
        plt.pause(2)
        count = count + 1


def plan_B(orient):
    T = 100
    F = 30
    A = 3
    L = 5
    N = 10
    FI = 2 * 312.5e3
    sample_rate = 1000
    TI = 1 / sample_rate
    AS = 0.5
    TR = np.arange(-100, 401, 1).reshape([1, 501]) * 1e-9
    AR = np.arange(0, 181, 1).reshape([1, 181]) / 180 * np.pi
    DR = np.arange(-80, 81, 1).reshape([1, 161])
    UR = 1

    sage_const = sage_const_build(T, F, A, L, N, FI, TI, AS, TR, AR, DR, UR)
    sage_const = sage_generate_steering_vector(sage_const)

    MN = round(sample_rate / T) + 1
    MS = 3
    parameter_weight = np.array([1e9, 180 / np.pi, 1, 1]).reshape([1, 4]) * np.array(
        [1 / 200, 1 / 90, 1 / 80, 10]).reshape([1, 4])
    window = 5

    cdata = scio.loadmat(data_path + trace_file)
    ccsi_data = cdata['csi_data']
    ctime_stamp = cdata['time_stamp']
    ground_truth = cdata['ground_truth']
    x = np.arange(0, np.size(ccsi_data, 0))

    count = 0
    step = 1000
    plt.plot(ground_truth[0, :], ground_truth[1, :], 'r--')
    while count <= np.size(ccsi_data, 0):
        csi_data = ccsi_data[count: count + step]
        time_stamp = ctime_stamp[count: count + step]

        csi_amplitude = np.mean(np.abs(csi_data), axis=0)
        csi_variance = np.sqrt([np.var(np.abs(csi_data[:, i])) for i in range(np.size(csi_data, 1))])
        csi_ratio = np.divide(csi_amplitude, csi_variance)
        ant_ratio = list(np.mean(np.reshape(csi_ratio, [F, A]), axis=0))
        midx = int(np.where(ant_ratio == np.max(ant_ratio))[0])
        # midx = ant_ratio.index(np.max(ant_ratio))
        csi_ref = np.zeros([np.size(csi_data, 0), F * A], dtype=complex)
        for ii in range(A):
            csi_ref[:, F * ii: F * (ii + 1)] = copy.copy(csi_data[:, midx * F + np.arange(F)])

        alpha_all = 0
        for jj in range(np.size(csi_data, 1)):
            alpha = min(np.abs(csi_data[:, jj]))
            alpha_all = alpha_all + alpha
            csi_data[:, jj] = np.multiply(np.abs(csi_data[:, jj]) - alpha, np.exp(1j * np.angle(csi_data[:, jj])))

        beta = alpha_all / np.size(csi_data, 1) * 1000
        for jj in range(np.size(csi_ref, 1)):
            csi_ref[:, jj] = np.multiply(np.abs(csi_ref[:, jj]) + beta, np.exp(1j * np.angle(csi_ref[:, jj])))
        csi_mult = np.multiply(csi_data, np.conj(csi_ref))
        scio.savemat(data_path + 'csi_mult-' + trace_file, {'csi_mult': csi_mult})

        hlfrt = sample_rate / 2
        upper_order = 6
        upper_stop = 80
        lower_order = 3
        lower_stop = 2
        b, a = signal.butter(5, [lower_stop / hlfrt, upper_stop / hlfrt], 'bandpass')
        csi_filter = np.zeros(np.shape(csi_mult), dtype=complex)
        for kk in range(np.size(csi_mult, 1)):
            csi_filter[:, kk] = signal.filtfilt(b, a, csi_mult[:, kk])


        # time_stamp_diff = list(np.diff(time_stamp, axis=0))  # 没有用到
        # time_wrap_index = [time_stamp_diff.index(ii) for ii in time_stamp_diff if ii < 0]  # 没有用到
        time_stamp = np.transpose((time_stamp - time_stamp[0, 0]) / 1e6).flatten()
        interp_stamp = np.arange(np.floor(time_stamp[-1] * sample_rate) / sample_rate)
        csi_interp = np.zeros([len(time_stamp), np.size(csi_filter, 1)], dtype=complex)
        xnew = np.arange(len(time_stamp))
        for jj in range(np.size(csi_filter, 1)):
            k = interpolate.interp1d(time_stamp, csi_filter[:, jj], kind='linear')  # 插值
            csi_interp[:, jj] = k(time_stamp[xnew])  # ？？？？？

        estimated_parameter = sage_main(csi_interp, 1000, sage_const, [1e-9, 1 / 180 * np.pi, 1, 1e-9], 4, 4)
        estimated_path = sage_path_mapping_filter(estimated_parameter, parameter_weight, MN, MS)

        rfl_matrix = np.sum(np.abs(estimated_path[3, :, :]), 1)
        rfl_index = int(np.where(rfl_matrix == np.max(rfl_matrix))[0])
        # rfl_index = rfl_matrix.index(np.max(rfl_matrix))
        rfl_path = np.squeeze(estimated_path[:, rfl_index, :])
        motion_index = np.arange(np.size(rfl_path, 1))
        tof = hampel(rfl_path[0, motion_index], window, 2e-9)
        aoa = hampel(rfl_path[1, motion_index], window, np.pi / 180)
        aoa = signal.savgol_filter(aoa, window, polyorder=2)  # 代替smooth
        doppler = hampel(rfl_path[2, motion_index], window, 6)
        doppler = signal.savgol_filter(doppler, window, polyorder=2).squeeze()
        power = rfl_path[3, motion_index]

        # 卡尔曼滤波器
        signal_range, pp, _, _ = sage_tof_filter(tof * c, -doppler * c / carrier_frequency, T / sample_rate, 0)
        # signal_range[signal_range <= 0.3] = 0.3
        for ii in range(np.size(signal_range, 0)):
            if signal_range[ii] <= 0.3:
                signal_range[ii] = 0.3
        signal_range = signal.savgol_filter(signal_range, window, polyorder=2)
        signal_range = signal_range + np.linalg.norm(rx_loc - tx_loc)
        aoa = orient - aoa

        location = sage_localization(signal_range.reshape([len(signal_range), 1]), aoa.reshape([len(signal_range), 1]),
                                     rx_loc.reshape([len(rx_loc), 1]), xb, yb)
        for ii in range(np.size(location, 0)):
            location[ii, :] = signal.savgol_filter(location[ii, :], window, polyorder=2)
        scio.savemat(data_path + 'location-' + trace_file, {'location': location})


        plt.plot(location[0, :], location[1, :], '*-')

        plt.show()
        plt.pause(1)
        count = count + int(step / 2)


def static_analize():
    if os.path.exists(data_path + 'device_orient.mat'):
        print('存在device_orient.mat')
        cdata = scio.loadmat(data_path + 'device_orient.mat')
        orient = cdata['orient']
    else:
        print('不存在device_orient.mat')
        print('第一次执行')
        T = 100
        F = 30
        A = 3
        L = 10
        N = 100
        FI = 2 * 312.5e3
        sample_rate = 1000
        TI = 1 / sample_rate
        AS = 0.5
        TR = np.arange(-100, 101, 1).reshape([1, 201]) * 1e-9
        AR = np.arange(0, 181, 1).reshape([1, 181]) / 180 * np.pi
        DR = np.arange(-20, 21, 1).reshape([1, 41])
        UR = 1

        sage_const = sage_const_build(T, F, A, L, N, FI, TI, AS, TR, AR, DR, UR)
        sage_const = sage_generate_steering_vector(sage_const)

        MN = round(sample_rate / T) + 1
        MS = 3
        parameter_weight = np.array([1e9, 180 / np.pi, 1, 1]).reshape([1, 4]) * np.array(
            [1 / 200, 1 / 90, 1 / 80, 10]).reshape([1, 4])

        data = scio.loadmat(data_path + static_file)
        csi_data = data['csi_data']

        plt.plot(np.unwrap(np.angle(csi_data[:, 0])))
        plt.plot(np.unwrap(np.angle(csi_data[:, 1])))
        plt.plot(np.unwrap(np.angle(csi_data[:, 2])))
        plt.plot(np.unwrap(np.angle(csi_data[:, 3])))
        plt.show()

        for jj in range(np.size(csi_data, 0)):
            csi_data[jj, :] = csi_sanitization(csi_data[jj, :], A, F)   # CSI cleaning

        estimated_parameter = sage_main(csi_data, 1000, sage_const, [1e-9, 1 / 180 * np.pi, 1, 1e-9], 4, 4)  # 正确
        # scio.savemat(data_path + 'PPM-' + static_file, {'estimated_parameter': estimated_parameter})

        # Path matching
        print('开始匹配路径')
        estimated_path = sage_path_mapping_filter(estimated_parameter, parameter_weight, MN, MS)
        # scio.savemat(data_path + 'PMM-' + static_file, {'estimated_path': estimated_path})

        #  Identify LoS path
        ss = np.sum(np.abs(estimated_path[3, :, :]), 1)
        los_index = int(np.where(ss == np.max(ss))[0])
        # los_index = ss.index(np.max(ss))
        los_path = np.squeeze(estimated_path[:, los_index, :])
        los_aoa = np.median(los_path[1, :])

        # Calculate device orientation
        rx_rmat = [[np.cos(los_aoa), np.sin(los_aoa)], [-np.sin(los_aoa), np.cos(los_aoa)]]
        rx_rvec = rx_loc / np.linalg.norm(rx_loc)
        rx_dvec = np.dot(np.transpose(rx_rvec), rx_rmat)[0]
        orient = math.atan2(rx_dvec[1], rx_dvec[0])
        # scio.savemat(data_path + 'device_orient.mat', {'orient': orient})
    return orient


def dynamic_analize(csi_data, time_stamp, count, step, orient, pre_buffer):
    T = 100
    F = 30
    A = 3
    L = 5
    N = 100
    FI = 2 * 312.5e3
    sample_rate = 1000
    TI = 1 / sample_rate
    AS = 0.5
    TR = np.arange(-100, 401, 1).reshape([1, 501]) * 1e-9
    AR = np.arange(0, 181, 1).reshape([1, 181]) / 180 * np.pi
    DR = np.arange(-80, 81, 1).reshape([1, 161])
    UR = 1

    sage_const = sage_const_build(T, F, A, L, N, FI, TI, AS, TR, AR, DR, UR)
    sage_const = sage_generate_steering_vector(sage_const)

    MN = 5
    MS = 2
    parameter_weight = np.array([1e9, 180 / np.pi, 1, 1]).reshape([1, 4]) * np.array(
        [1 / 200, 1 / 90, 1 / 80, 10]).reshape([1, 4])

    csi_data_cut = 50
    pre_time_stamp = pre_buffer['pre_time_stamp']
    pre_time_stamp_len = 100
    pre_csi_mult = pre_buffer['pre_csi_mult']
    pre_csi_mult_len = pre_time_stamp_len
    pre_rfl_path = pre_buffer['pre_rfl_path']
    pre_rfl_path_len = 10
    window = 5  # =5
    para_cut = 2
    comp = pre_rfl_path_len - para_cut
    xt0 = [0]
    pt0 = [0]
    pre_signal = pre_buffer['pre_signal']
    pre_aoa = pre_buffer['pre_aoa']
    comp_len = 0
    pre_signal_len = 10
    signal_range_cut = 2
    x = np.arange(0, 15000)
    plt.figure(1)

    csi_amplitude = np.mean(np.abs(csi_data), axis=0)
    csi_variance = np.sqrt([np.var(np.abs(csi_data[:, i])) for i in range(np.size(csi_data, 1))])
    csi_ratio = np.divide(csi_amplitude, csi_variance)
    ant_ratio = np.mean(np.reshape(csi_ratio, [F, A]), axis=0)
    midx = int(np.where(ant_ratio == np.max(ant_ratio))[0])

    # midx = ant_ratio.index(np.max(ant_ratio))
    csi_ref = np.zeros([np.size(csi_data, 0), F * A], dtype=complex)
    for qq in range(A):
        csi_ref[:, F * qq: F * (qq + 1)] = copy.copy(csi_data[:, midx * F + np.arange(F)])

    alpha_all = 0
    for jj in range(np.size(csi_data, 1)):
        alpha = min(np.abs(csi_data[:, jj]))
        alpha_all = alpha_all + alpha
        csi_data[:, jj] = np.multiply(np.abs(csi_data[:, jj]) - alpha, np.exp(1j * np.angle(csi_data[:, jj])))

    beta = alpha_all / np.size(csi_data, 1) * 1000
    for jj in range(np.size(csi_ref, 1)):
        csi_ref[:, jj] = np.multiply(np.abs(csi_ref[:, jj]) + beta, np.exp(1j * np.angle(csi_ref[:, jj])))
    csi_mult = np.multiply(csi_data, np.conj(csi_ref))
    ####################################################################################################################################
    plt.subplot(2, 3, 1)
    hlfrt = sample_rate / 2
    upper_order = 6
    upper_stop = 80
    lower_order = 3
    lower_stop = 2
    b, a = signal.butter(5, [lower_stop / hlfrt, upper_stop / hlfrt], 'bandpass')
    csi_mult_ex = np.concatenate((pre_csi_mult, csi_mult))
    csi_filter = np.zeros(np.shape(csi_mult_ex), dtype=complex)
    for kk in range(np.size(csi_mult_ex, 1)):
        csi_filter[:, kk] = signal.filtfilt(b, a, csi_mult_ex[:, kk])  # 直接采取前后截断

    if count != 0:
        csi_filter_final = csi_filter[pre_csi_mult_len - csi_data_cut: -csi_data_cut]
        plt.plot(x[
                 count * step - pre_csi_mult_len + csi_data_cut: count * step - pre_csi_mult_len + csi_data_cut + len(
                     csi_filter_final)], csi_filter_final[:, 1])
    else:
        csi_filter_final = csi_filter[0: -csi_data_cut]
        plt.plot(x[0: len(csi_filter_final)], csi_filter_final[:, 1])

    pre_csi_mult = csi_mult[-pre_csi_mult_len - 1: -1]
    ####################################################################################################################################
    time_stamp_ex = np.concatenate((pre_time_stamp, time_stamp))
    pre_time_stamp = time_stamp[-pre_time_stamp_len:]

    if count != 0:
        time_stamp_ex = np.transpose((time_stamp_ex - time_stamp_ex[0, 0]) / 1e6).flatten()[
                        pre_time_stamp_len - csi_data_cut: -csi_data_cut]
    else:
        time_stamp_ex = np.transpose((time_stamp_ex - time_stamp_ex[0, 0]) / 1e6).flatten()[0: -csi_data_cut]

    interp_stamp = np.arange(np.floor(time_stamp_ex[-1] * sample_rate) / sample_rate)
    csi_interp = np.zeros([len(time_stamp_ex), np.size(csi_filter_final, 1)], dtype=complex)
    xnew = np.arange(len(time_stamp_ex))
    for jj in range(np.size(csi_filter_final, 1)):
        try:
            k = interpolate.interp1d(time_stamp_ex, csi_filter_final[:, jj], kind='linear')  # 插值
        except:
            k = interpolate.interp1d(time_stamp_ex[: len(csi_filter_final[:, jj])], csi_filter_final[:, jj], kind='linear')  # 插值
        csi_interp[:, jj] = k(time_stamp_ex[xnew])  # ？？？？？
    ###############################################################################################################################
    t1 = time.time()
    estimated_parameter = sage_main(csi_interp, 1000, sage_const, [1e-9, 1 / 180 * np.pi, 1, 1e-9], 4, 4)
    print('SAGE用时：', time.time() - t1)
    print('开始匹配路径')
    t1 = time.time()
    estimated_path = sage_path_mapping_filter(estimated_parameter, parameter_weight, MN, MS)
    print('用时：', time.time() - t1)
    ####################################################################################################################################
    rfl_matrix = np.sum(np.abs(estimated_path[3, :, :]), 1)  # 转换速度慢
    rfl_index = int(np.where(rfl_matrix == np.max(rfl_matrix))[0])
    # rfl_index = rfl_matrix.index(np.max(rfl_matrix))  # =0，可只运行一次
    rfl_path = np.squeeze(estimated_path[:, rfl_index, :])
    rfl_path_ex = np.concatenate((pre_rfl_path, rfl_path), axis=1)
    motion_index = np.arange(np.size(rfl_path_ex, 1))  ##??????

    if count != 0:
        tof = hampel(rfl_path_ex[0, motion_index], window, 2e-9)[comp: -para_cut]
        aoa = hampel(rfl_path_ex[1, motion_index], window, np.pi / 180)[comp: -para_cut]
        print(np.shape(aoa))
        aoa = signal.savgol_filter(aoa, window, polyorder=2).squeeze()  # 代替smooth
        doppler = hampel(rfl_path_ex[2, motion_index], window, 6)[comp: -para_cut]
        doppler = signal.savgol_filter(doppler, window, polyorder=2).squeeze()
        signal_range, pp, xt0, pt0 = sage_tof_filter2(tof * c, -doppler * c / carrier_frequency, T / sample_rate,
                                                      xt0, pt0, 5)

        # power = rfl_path_ex[3, motion_index]
        plt.subplot(2, 3, 2)
        plt.plot(x[count * int(step / T) - para_cut - 1: count * int(step / T) - para_cut - 1 + len(tof)], tof)
        plt.subplot(2, 3, 3)
        plt.plot(x[count * int(step / T) - para_cut - 1: count * int(step / T) - para_cut - 1 + len(aoa)], aoa)
        plt.subplot(2, 3, 4)
        plt.plot(x[count * int(step / T) - para_cut - 1: count * int(step / T) - para_cut - 1 + len(doppler)],
                 doppler)
        plt.subplot(2, 3, 5)
        plt.plot(x[count * int(step / T) - para_cut - 1: count * int(step / T) - para_cut - 1 + len(signal_range)],
                 signal_range)
    else:
        tof = hampel(rfl_path_ex[0, motion_index], window, 2e-9)[: -para_cut]
        aoa = hampel(rfl_path_ex[1, motion_index], window, np.pi / 180)[: -para_cut]
        aoa = signal.savgol_filter(aoa, window, polyorder=2).squeeze()  # 代替smooth
        doppler = hampel(rfl_path_ex[2, motion_index], window, 6)[: -para_cut]
        doppler = signal.savgol_filter(doppler, window, polyorder=2).squeeze()
        signal_range, pp, xt0, pt0 = sage_tof_filter2(tof * c, -doppler * c / carrier_frequency, T / sample_rate,
                                                      xt0, pt0, 5)

        plt.subplot(2, 3, 2)
        plt.plot(x[: len(tof)], tof)
        plt.subplot(2, 3, 3)
        plt.plot(x[: len(tof)], aoa)
        plt.subplot(2, 3, 4)
        plt.plot(x[: len(tof)], doppler)
        plt.subplot(2, 3, 5)
        plt.plot(x[: len(signal_range)], signal_range)

    pre_rfl_path = rfl_path[:, -pre_rfl_path_len:]
    ##########################################################################################################################
    # 卡尔曼滤波器
    signal_range[signal_range <= 0.3] = 0.3
    signal_range = signal.savgol_filter(signal_range, window, polyorder=2)
    signal_range = signal_range + np.linalg.norm(rx_loc - tx_loc)
    aoa = orient - aoa

    if count != 0:
        comp_len = pre_signal_len - signal_range_cut
    else:
        comp_len = 0
    signal_range_ex = np.concatenate((pre_signal, signal_range))
    aoa_ex = np.concatenate((pre_aoa, np.squeeze(aoa)))
    location = sage_localization(signal_range_ex.reshape([np.size(signal_range_ex, 0), 1]),
                                 aoa_ex.reshape([np.size(aoa_ex, 0), 1]), rx_loc, xb, yb)[:,
               comp_len: -signal_range_cut]
    pre_signal = signal_range_ex[-pre_signal_len - 1: -1]
    pre_aoa = aoa_ex[-pre_signal_len - 1: -1]

    for ii in range(np.size(location, 0)):
        location[ii, :] = signal.savgol_filter(location[ii, :], window, polyorder=2)  # 滤波器

    plt.subplot(2, 3, 6)
    # plt.plot(ground_truth[0, :], ground_truth[1, :], 'r--')
    plt.plot(location[0, :], location[1, :], 'o--')
    plt.draw()
    # plt.show()
    plt.pause(0.1)

    pre_buffer_return = {
        'pre_time_stamp': pre_time_stamp,
        'pre_csi_mult': pre_csi_mult,
        'pre_rfl_path': pre_rfl_path,
        'pre_signal': pre_signal,
        'pre_aoa': pre_aoa
    }
    return pre_buffer_return


def dynamic_analize_test(csi_data, time_stamp, count, step, orient, pre_buffer):
    # SAGE parameters
    # F: 子载波数，T：包数(but not practical rate of transmitting package)，TR：ToF范围，A（S）：天线数（传感器数）
    # FI: 频率整数(differential between adjacent freq)，TI：时间整数，AS：天线空间，
    # AR：AOA范围，DR：多普勒范围，UR：更新率
    # L：传播路径数，N: iteration number in sage_cfg
    # L、N在本算法中无用处
    # m = (i, j, k)，其中i = 0, 1, ..., T，j = 0, 1, ..., F，k = 0, 1, ..., A
    # m is the hyper - domain for CSI measurement H(i, j, k)

    T = 100  # data_len / T = number of circles
    F = 30
    A = 3
    L = 5
    N = 100
    FI = 2 * 312.5e3
    sample_rate = 200  # modify
    TI = 1 / sample_rate
    AS = 0.5
    TR = np.arange(-100, 401, 1).reshape([1, 501]) * 1e-9
    AR = np.arange(0, 181, 1).reshape([1, 181]) / 180 * np.pi
    DR = np.arange(-80, 81, 1).reshape([1, 161])
    UR = 1

    sage_const = sage_const_build(T, F, A, L, N, FI, TI, AS, TR, AR, DR, UR)
    sage_const = sage_generate_steering_vector(sage_const)

    MN = 5
    MS = 2
    parameter_weight = np.array([1e9, 180 / np.pi, 1, 1]).reshape([1, 4]) * np.array(
        [1 / 200, 1 / 90, 1 / 80, 10]).reshape([1, 4])

    csi_data_cut = 50
    pre_time_stamp = pre_buffer['pre_time_stamp']
    pre_time_stamp_len = 100
    pre_csi_mult = pre_buffer['pre_csi_mult']
    pre_csi_mult_len = pre_time_stamp_len
    pre_rfl_path = pre_buffer['pre_rfl_path']
    pre_rfl_path_len = 10
    window = 5  # =5
    para_cut = 2
    comp = pre_rfl_path_len - para_cut
    xt0 = [0]
    pt0 = [0]
    pre_signal = pre_buffer['pre_signal']
    pre_aoa = pre_buffer['pre_aoa']
    comp_len = 0
    pre_signal_len = 10
    signal_range_cut = 2
    x = np.arange(0, 15000)
    plt.figure(1)

    csi_amplitude = np.mean(np.abs(csi_data), axis=0)
    csi_variance = np.sqrt([np.var(np.abs(csi_data[:, i])) for i in range(np.size(csi_data, 1))])
    csi_ratio = np.divide(csi_amplitude, csi_variance)
    ant_ratio = np.mean(np.reshape(csi_ratio, [F, A]), axis=0)
    midx = int(np.where(ant_ratio == np.max(ant_ratio))[0])

    # midx = ant_ratio.index(np.max(ant_ratio))
    csi_ref = np.zeros([np.size(csi_data, 0), F * A], dtype=complex)
    for qq in range(A):
        csi_ref[:, F * qq: F * (qq + 1)] = copy.copy(csi_data[:, midx * F + np.arange(F)])

    alpha_all = 0
    for jj in range(np.size(csi_data, 1)):
        alpha = min(np.abs(csi_data[:, jj]))
        alpha_all = alpha_all + alpha
        csi_data[:, jj] = np.multiply(np.abs(csi_data[:, jj]) - alpha, np.exp(1j * np.angle(csi_data[:, jj])))

    beta = alpha_all / np.size(csi_data, 1) * 1000
    for jj in range(np.size(csi_ref, 1)):
        csi_ref[:, jj] = np.multiply(np.abs(csi_ref[:, jj]) + beta, np.exp(1j * np.angle(csi_ref[:, jj])))
    csi_mult = np.multiply(csi_data, np.conj(csi_ref))
    ####################################################################################################################################
    plt.subplot(2, 3, 1)
    hlfrt = sample_rate / 2
    upper_order = 6
    upper_stop = 80
    lower_order = 3
    lower_stop = 2
    b, a = signal.butter(5, [lower_stop / hlfrt, upper_stop / hlfrt], 'bandpass')
    csi_mult_ex = np.concatenate((pre_csi_mult, csi_mult))
    csi_filter = np.zeros(np.shape(csi_mult_ex), dtype=complex)
    for kk in range(np.size(csi_mult_ex, 1)):
        csi_filter[:, kk] = signal.filtfilt(b, a, csi_mult_ex[:, kk])  # 直接采取前后截断

    if count != 0:
        csi_filter_final = csi_filter[pre_csi_mult_len - csi_data_cut: -csi_data_cut]
        plt.plot(x[
                 count * step - pre_csi_mult_len + csi_data_cut: count * step - pre_csi_mult_len + csi_data_cut + len(
                     csi_filter_final)], csi_filter_final[:, 1])
    else:
        csi_filter_final = csi_filter[0: -csi_data_cut]
        plt.plot(x[0: len(csi_filter_final)], csi_filter_final[:, 1])

    pre_csi_mult = csi_mult[-pre_csi_mult_len - 1: -1]
    ####################################################################################################################################
    time_stamp_ex = np.concatenate((pre_time_stamp, time_stamp))
    pre_time_stamp = time_stamp[-pre_time_stamp_len:]

    if count != 0:
        time_stamp_ex = np.transpose((time_stamp_ex - time_stamp_ex[0, 0]) / 1e6).flatten()[
                        pre_time_stamp_len - csi_data_cut: -csi_data_cut]
    else:
        time_stamp_ex = np.transpose((time_stamp_ex - time_stamp_ex[0, 0]) / 1e6).flatten()[0: -csi_data_cut]

    interp_stamp = np.arange(np.floor(time_stamp_ex[-1] * sample_rate) / sample_rate)
    csi_interp = np.zeros([len(time_stamp_ex), np.size(csi_filter_final, 1)], dtype=complex)
    xnew = np.arange(len(time_stamp_ex))
    for jj in range(np.size(csi_filter_final, 1)):
        try:
            k = interpolate.interp1d(time_stamp_ex, csi_filter_final[:, jj], kind='linear')  # 插值
        except:
            k = interpolate.interp1d(time_stamp_ex[: len(csi_filter_final[:, jj])], csi_filter_final[:, jj], kind='linear')  # 插值
        csi_interp[:, jj] = k(time_stamp_ex[xnew])  # ？？？？？
    ###############################################################################################################################
    t1 = time.time()
    estimated_parameter = sage_main(csi_interp, 1000, sage_const, [1e-9, 1 / 180 * np.pi, 1, 1e-9], 4, 4)
    print('SAGE用时：', time.time() - t1)
    print('开始匹配路径')
    t1 = time.time()
    estimated_path = sage_path_mapping_filter(estimated_parameter, parameter_weight, MN, MS)
    print('用时：', time.time() - t1)
    ####################################################################################################################################
    rfl_matrix = np.sum(np.abs(estimated_path[3, :, :]), 1)  # 转换速度慢
    rfl_index = int(np.where(rfl_matrix == np.max(rfl_matrix))[0])
    # rfl_index = rfl_matrix.index(np.max(rfl_matrix))  # =0，可只运行一次
    rfl_path = np.squeeze(estimated_path[:, rfl_index, :])
    rfl_path_ex = np.concatenate((pre_rfl_path, rfl_path), axis=1)
    motion_index = np.arange(np.size(rfl_path_ex, 1))  ##??????

    if count != 0:
        tof = hampel(rfl_path_ex[0, motion_index], window, 2e-9)[comp: -para_cut]
        aoa = hampel(rfl_path_ex[1, motion_index], window, np.pi / 180)[comp: -para_cut]
        aoa = signal.savgol_filter(aoa, window, polyorder=2).squeeze()  # 代替smooth
        doppler = hampel(rfl_path_ex[2, motion_index], window, 6)[comp: -para_cut]
        doppler = signal.savgol_filter(doppler, window, polyorder=2).squeeze()
        signal_range, pp, xt0, pt0 = sage_tof_filter2(tof * c, -doppler * c / carrier_frequency, T / sample_rate,
                                                      xt0, pt0, 5)

        # power = rfl_path_ex[3, motion_index]
        plt.subplot(2, 3, 2)
        plt.plot(x[count * int(step / T) - para_cut - 1: count * int(step / T) - para_cut - 1 + len(tof)], tof)
        plt.subplot(2, 3, 3)
        plt.plot(x[count * int(step / T) - para_cut - 1: count * int(step / T) - para_cut - 1 + len(aoa)], aoa)
        plt.subplot(2, 3, 4)
        plt.plot(x[count * int(step / T) - para_cut - 1: count * int(step / T) - para_cut - 1 + len(doppler)],
                 doppler)
        plt.subplot(2, 3, 5)
        plt.plot(x[count * int(step / T) - para_cut - 1: count * int(step / T) - para_cut - 1 + len(signal_range)],
                 signal_range)
    else:
        tof = hampel(rfl_path_ex[0, motion_index], window, 2e-9)[: -para_cut]
        aoa = hampel(rfl_path_ex[1, motion_index], window, np.pi / 180)[: -para_cut]
        aoa = signal.savgol_filter(aoa, window, polyorder=2).squeeze()  # 代替smooth
        doppler = hampel(rfl_path_ex[2, motion_index], window, 6)[: -para_cut]
        doppler = signal.savgol_filter(doppler, window, polyorder=2).squeeze()
        signal_range, pp, xt0, pt0 = sage_tof_filter2(tof * c, -doppler * c / carrier_frequency, T / sample_rate,
                                                      xt0, pt0, 5)

        plt.subplot(2, 3, 2)
        plt.plot(x[: len(tof)], tof)
        plt.subplot(2, 3, 3)
        plt.plot(x[: len(tof)], aoa)
        plt.subplot(2, 3, 4)
        plt.plot(x[: len(tof)], doppler)
        plt.subplot(2, 3, 5)
        plt.plot(x[: len(signal_range)], signal_range)

    pre_rfl_path = rfl_path[:, -pre_rfl_path_len:]
    ##########################################################################################################################
    # 卡尔曼滤波器
    signal_range[signal_range <= 0.3] = 0.3
    signal_range = signal.savgol_filter(signal_range, window, polyorder=2)
    signal_range = signal_range + np.linalg.norm(rx_loc - tx_loc)
    aoa = orient - aoa

    if count != 0:
        comp_len = pre_signal_len - signal_range_cut
    else:
        comp_len = 0
    signal_range_ex = np.concatenate((pre_signal, signal_range))
    aoa_ex = np.concatenate((pre_aoa, np.squeeze(aoa)))
    location = sage_localization(signal_range_ex.reshape([np.size(signal_range_ex, 0), 1]),
                                 aoa_ex.reshape([np.size(aoa_ex, 0), 1]), rx_loc, xb, yb)[:,
               comp_len: -signal_range_cut]
    pre_signal = signal_range_ex[-pre_signal_len - 1: -1]
    pre_aoa = aoa_ex[-pre_signal_len - 1: -1]

    for ii in range(np.size(location, 0)):
        location[ii, :] = signal.savgol_filter(location[ii, :], window, polyorder=2)  # 滤波器

    plt.subplot(2, 3, 6)
    # plt.plot(ground_truth[0, :], ground_truth[1, :], 'r--')
    plt.plot(location[0, :], location[1, :], 'o--')
    plt.draw()
    # plt.show()
    plt.pause(0.1)

    pre_buffer_return = {
        'pre_time_stamp': pre_time_stamp,
        'pre_csi_mult': pre_csi_mult,
        'pre_rfl_path': pre_rfl_path,
        'pre_signal': pre_signal,
        'pre_aoa': pre_aoa
    }
    return pre_buffer_return


if __name__ == '__main__':
    orient = static_analize()

    count = 0
    step = 1000
    pre_buffer = dynamic_pre_buffer_build()
    plt.subplot(2, 3, 6)
    plt.plot(location_truth[0, :], location_truth[1, :], 'g--')
    plt.draw()
    plt.pause(0.1)

    while True:
        csi_data = ccsi_data[count*step: (count+1)*step]
        time_stamp = ctime_stamp[count: count + step]
        pre_buffer = dynamic_analize(csi_data, time_stamp, count, step, orient, pre_buffer)
        count = count + 1
