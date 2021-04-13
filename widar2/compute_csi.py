import picos as pic
import numpy as np
import scipy.io as scio
from scipy import signal, interpolate
import math
import copy
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import os
# from scipy.interpolate import spline
# from pykalman import KalmanFilter  # 卡尔曼平滑

# plt.figure(1)

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

    temp1 = -1j * 2 * np.pi * frequency_interval * np.arange(0, F).reshape([F,1]) * tof_range
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
    # M = 3
    # N = 30
    freq_delta = 2 * 312.5e3

    csi_phase = np.zeros(M*N)
    for ii in range(1, M+1):
        if ii == 1:
            csi_phase[(ii-1)*N: ii*N] = np.unwrap(np.angle(csi_data[(ii-1)*N: ii*N]))
        else:
            csi_diff = np.angle(np.multiply(csi_data[(ii-1)*N: ii*N], np.conj(csi_data[(ii-2)*N: (ii-1)*N])))
            csi_phase[(ii-1)*N: ii*N] = np.unwrap(csi_phase[(ii-2)*N: (ii-1)*N]+csi_diff)

    ai = 2 * np.pi * freq_delta * np.tile(np.arange(0, N),M).reshape([1, N*M])
    bi = np.ones((1, len(csi_phase)))
    ci = csi_phase
    A = np.dot(ai, np.transpose(ai))[0]
    B = np.dot(ai, np.transpose(bi))[0]
    C = np.dot(bi, np.transpose(bi))[0]
    D = np.dot(ai, np.transpose(ci))[0]
    E = np.dot(bi, np.transpose(ci))[0]
    rho_opt = (B * E - C * D) / (A * C - np.square(B))
    beta_opt = (B * D - A * E) / (A * C - np.square(B))

    csi_phase_2 = csi_phase + 2 * np.pi * freq_delta * np.tile(np.arange(0, N),M).reshape([1, N*M]) * rho_opt + beta_opt
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

    aoa_matrix_temp = np.zeros([A, T, F], dtype=complex)
    t1 = sage_const['aoa_candidates'][:, latent_index_temp[1]]
    for kk in range(np.size(aoa_matrix_temp, 2)):
        for jj in range(np.size(aoa_matrix_temp, 1)):
            aoa_matrix_temp[:, jj, kk] = t1
    aoa_matrix = np.transpose(aoa_matrix_temp, [1, 2, 0])

    doppler_matrix = np.zeros([T, F, A], dtype=complex)
    t2 = sage_const['doppler_candidates'][:, latent_index_temp[2]]
    for kk in range(np.size(doppler_matrix, 2)):
        for jj in range(np.size(doppler_matrix, 1)):
            doppler_matrix[:, jj, kk] = t2

    coeff_matrix = np.multiply(np.multiply(latent_signal, np.conj(aoa_matrix)), np.conj(doppler_matrix))
    coeff_vector = np.reshape(np.sum(np.sum(coeff_matrix, 0), 1), [F, 1], order='F')
    coeff_vector = np.tile(coeff_vector, [1, len(tof_range)])
    object_vector = list(np.abs(np.sum(np.multiply(coeff_vector, np.conj(sage_const['tof_candidates'])), 0)))
    latent_index_temp[0] = object_vector.index(np.max(object_vector))
    latent_parameter_temp[0] = tof_range[0, latent_index_temp[0]]

    tof_matrix_temp = np.zeros([F, T, A], dtype=complex)
    t3 = sage_const['tof_candidates'][:, latent_index_temp[0]]
    for kk in range(np.size(tof_matrix_temp, 2)):
        for jj in range(np.size(tof_matrix_temp, 1)):
            tof_matrix_temp[:, jj, kk] = t3
    tof_matrix = np.transpose(tof_matrix_temp, [1, 0, 2])

    coeff_matrix = np.multiply(np.multiply(latent_signal, np.conj(doppler_matrix)), np.conj(tof_matrix))
    coeff_vector = np.reshape(np.sum(np.sum(coeff_matrix, 0), 0), [A, 1], order='F')
    coeff_vector = np.tile(coeff_vector, [1, len(aoa_range)])
    object_vector = list(np.abs(np.sum(np.multiply(coeff_vector, np.conj(sage_const['aoa_candidates'])), 0)))
    latent_index_temp[1] = object_vector.index(np.max(object_vector))
    latent_parameter_temp[1] = aoa_range[0, latent_index_temp[1]]

    t1 = sage_const['aoa_candidates'][:, latent_index_temp[1]]
    for kk in range(np.size(aoa_matrix_temp, 2)):
        for jj in range(np.size(aoa_matrix_temp, 1)):
            aoa_matrix_temp[:, jj, kk] = t1
    aoa_matrix = np.transpose(aoa_matrix_temp, [1, 2, 0])

    coeff_matrix = np.multiply(np.multiply(latent_signal, np.conj(aoa_matrix)), np.conj(tof_matrix))
    coeff_vector = np.reshape(np.sum(np.sum(coeff_matrix, 1), 1), [T, 1], order='F')
    coeff_vector = np.tile(coeff_vector, [1, len(doppler_range)])
    object_vector = list(np.abs(np.sum(np.multiply(coeff_vector, np.conj(sage_const['doppler_candidates'])), 0)))
    latent_index_temp[2] = object_vector.index(np.max(object_vector))
    latent_parameter_temp[2] = doppler_range[0, latent_index_temp[2]]

    t2 = sage_const['doppler_candidates'][:, latent_index_temp[2]]
    for kk in range(np.size(doppler_matrix, 2)):
        for jj in range(np.size(doppler_matrix, 1)):
            doppler_matrix[:, jj, kk] = t2

    coeff_matrix = np.multiply(np.multiply(np.multiply(np.conj(aoa_matrix), np.conj(tof_matrix)), np.conj(doppler_matrix)), latent_signal)
    latent_parameter_temp[3] = np.sum(coeff_matrix) / (T * F * A)
    return latent_parameter_temp, latent_index_temp


def sage_sfg(csi_signal, initial_parameter, initial_index, sage_const):
    T = sage_const['T']
    L = sage_const['L']
    F = sage_const['F']
    A = sage_const['A']
    N = sage_const['N']
    update_ratio = sage_const['update_ratio']
    latent_signal = np.zeros([T, F, A, L], dtype=complex)

    for ii in range(L):
        if initial_parameter[3, ii] != 0:
            latent_signal[:, :, :, ii] = sage_signal(initial_parameter[:, ii], initial_index[:, ii], sage_const)

    final_parameter = copy.copy(initial_parameter)
    temp_parameter = copy.copy(initial_parameter)
    temp_index = copy.copy(initial_index)
    final_index = copy.copy(initial_index)
    for ii in range(N):
        for jj in range(L):
            temp_signal = sage_expectation(csi_signal, latent_signal, jj, update_ratio) # 正确
            temp_parameter[:, jj],temp_index[:, jj] = sage_maximization(temp_signal, final_parameter[:, jj], final_index[:, jj], sage_const) #正确
            latent_signal[:, :, :, jj] = sage_signal(temp_parameter[:, jj], temp_index[:, jj], sage_const)

        parameter_diff = np.sqrt(np.sum(np.abs(temp_parameter - final_parameter) ** 2,1))
        final_parameter = copy.copy(temp_parameter)
        final_index = copy.copy(temp_index)

        if parameter_diff[0] < 1e-9 and parameter_diff[1] < 1 / 180 * np.pi and parameter_diff[2] < 1 and parameter_diff[3] < 1e-9:
            break
    residue_error = csi_signal - np.sum(latent_signal, 3)
    residue_error = np.mean(np.abs(residue_error)) / np.mean(np.abs(csi_signal))
    return final_parameter,residue_error


def sage_main(csi_data, G, sage_const):
    T = sage_const['T']
    L = sage_const['L']
    F = sage_const['F']
    A = sage_const['A']
    G = min([G, np.floor(np.size(csi_data, 0)/T)]).astype(int)  # 分段长度
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
        estimated_parameter[:, :, ii], residue_errors[:, ii] = sage_sfg(csi_signal, initial_parameter, initial_index, sage_const)
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
                min_value = min(c)
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


def sage_tof_filter(z, u, dt):
    xt = np.zeros(np.size(z))
    xt1 = np.zeros(np.size(z))
    pt = np.zeros(np.size(z))
    pt1 = np.zeros(np.size(z))
    R = np.var(z[0: int(np.floor(2 / dt))])
    Q = np.var(u[0: int(np.floor(2 / dt))])
    xt[0] = copy.copy(z[0])
    xt1[0] = copy.copy(z[0])
    pt[0] = copy.copy(R)
    A = 1
    H = 1
    B = copy.copy(dt)

    for ii in range (1, np.size(z)):
        xt1[ii] = xt[ii - 1] + B * u[ii - 1]
        pt1[ii] = pt[ii - 1] + Q
        K = pt1[ii] / (pt1[ii] + R)
        xt[ii] = xt1[ii] + K * (z[ii] - xt1[ii])
        pt[ii] = pt1[ii] - K * pt1[ii]

    x = copy.copy(xt)
    p = copy.copy(pt)
    for ii in range(np.size(z)-2, -1, -1):
        L = pt[ii] / pt1[ii + 1]
        x[ii] = xt[ii] + L * (x[ii + 1] - xt1[ii + 1])
        p[ii] = pt[ii] + L * (p[ii + 1] - pt1[ii + 1]) * L

    return x, p


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



if __name__ == '__main__':
    data_path = '../MATLAB_data/'
    ddata_path = 'MATLAB_org_data/'
    static_file = 'S01.mat'
    trace_file = 'T06.mat'

    cdata = scio.loadmat(data_path + 'device_config.mat')
    rx_loc = cdata['rx_loc']  # 接收机位置
    tx_loc = cdata['tx_loc']  # 发射机位置
    c = np.squeeze(cdata['c'])
    carrier_frequency = np.squeeze(cdata['carrier_frequency'])
    yb = np.squeeze(cdata['yb'])  # y轴取值范围
    xb = np.squeeze(cdata['xb'])  # x轴取值范围

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
        parameter_weight =np.array([1e9, 180/np.pi, 1, 1]).reshape([1, 4]) * np.array([1/200, 1/90, 1/80, 10]).reshape([1, 4])

        data = scio.loadmat(data_path + static_file)
        csi_data = data['csi_data']

        for jj in range(np.size(csi_data, 0)):
            csi_data[jj, :] = csi_sanitization(csi_data[jj, :], A, F)

        estimated_parameter = sage_main(csi_data, 1000, sage_const) # 正确
        scio.savemat(data_path + 'PPM-' + static_file, {'estimated_parameter': estimated_parameter})

        # Path matching
        print('开始匹配路径')
        estimated_path = sage_path_mapping_filter(estimated_parameter, parameter_weight, MN, MS)
        scio.savemat(data_path + 'PMM-' + static_file, {'estimated_path': estimated_path})

        #  Identify LoS path
        ss = list(np.sum(np.abs(estimated_path[3, :, :]), 1))
        los_index = ss.index(np.max(ss))
        los_path = np.squeeze(estimated_path[:, los_index, :])
        los_aoa = np.median(los_path[1, :])

        # Calculate device orientation
        rx_rmat = [[np.cos(los_aoa), np.sin(los_aoa)], [-np.sin(los_aoa), np.cos(los_aoa)]]
        rx_rvec = rx_loc / np.linalg.norm(rx_loc)
        rx_dvec = np.dot(np.transpose(rx_rvec), rx_rmat)[0]
        orient = math.atan2(rx_dvec[1], rx_dvec[0])
        scio.savemat(data_path + 'device_orient.mat', {'orient': orient})

    print('已经执行过第一步')
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
    parameter_weight = np.array([1e9, 180 / np.pi, 1, 1]).reshape([1, 4]) * np.array([1 / 200, 1 / 90, 1 / 80, 10]).reshape([1, 4])

    cdata = scio.loadmat(data_path + trace_file)
    csi_data = cdata['csi_data']
    ground_truth = cdata['ground_truth']
    if os.path.exists(data_path + 'PPM-' + trace_file):
        print('存在' + data_path + 'PPM-' + trace_file)
        ddata = scio.loadmat(data_path + 'PPM-' + trace_file)
        estimated_parameter = ddata['estimated_parameter']
    else:
        print('不存在' + data_path + 'PPM-' + trace_file)
        csi_amplitude = np.mean(np.abs(csi_data), axis=0)
        csi_variance = np.sqrt([np.var(np.abs(csi_data[:, i])) for i in range(np.size(csi_data, 1))])
        csi_ratio = np.divide(csi_amplitude, csi_variance)
        ant_ratio = list(np.mean(np.reshape(csi_ratio, [F, A]), axis=0))
        midx = ant_ratio.index(np.max(ant_ratio))
        csi_ref = np.zeros([np.size(csi_data, 0), F * A], dtype=complex)
        for ii in range(A):
            csi_ref[:, F*ii: F*(ii+1)] = copy.copy(csi_data[:, midx * F + np.arange(F)])

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
        B, A = signal.butter(5, [lower_stop / hlfrt, upper_stop / hlfrt], 'bandpass')
        csi_filter = np.zeros(np.shape(csi_mult), dtype=complex)
        for kk in range(np.size(csi_mult, 1)):
            csi_filter[:, kk] = signal.filtfilt(B, A, csi_mult[:, kk])

        time_stamp = cdata['time_stamp']
        time_stamp_diff = list(np.diff(time_stamp, axis=0))
        time_wrap_index = [time_stamp_diff.index(ii) for ii in time_stamp_diff if ii <0]
        time_stamp = np.transpose((time_stamp - time_stamp[0, 0]) / 1e6).flatten()
        interp_stamp = np.arange(np.floor(time_stamp[-1] * sample_rate) / sample_rate)
        csi_interp = np.zeros([len(time_stamp), np.size(csi_filter, 1)], dtype=complex)
        xnew = np.arange(len(time_stamp))
        for jj in range(np.size(csi_filter, 1)):
            k = interpolate.interp1d(time_stamp, csi_filter[:, jj], kind='linear')  # 插值
            csi_interp[:, jj] = k(time_stamp[xnew])  #？？？？？

        scio.savemat(data_path + 'variables-' + trace_file, {'csi_interp': csi_interp})
        estimated_parameter = sage_main(csi_interp, 1000, sage_const)
        scio.savemat(data_path + 'PPM-' + trace_file, {'estimated_parameter': estimated_parameter})

    if os.path.exists(data_path + 'PMM-' + trace_file):
        print('存在' + data_path + 'PMM-' + trace_file)
        ddata = scio.loadmat(data_path + 'PMM-' + trace_file)
        estimated_path = ddata['estimated_path']
    else:
        print('不存在' + data_path + 'PMM-' + trace_file)  #关键是PMM！！！！
        estimated_path = sage_path_mapping_filter(estimated_parameter, parameter_weight, MN, MS)
        scio.savemat(data_path + 'PMM-' + trace_file, {'estimated_path': estimated_path})

    rfl_matrix = list(np.sum(np.abs(estimated_path[3, :, :]), 1))
    rfl_index = rfl_matrix.index(np.max(rfl_matrix))
    rfl_path = np.squeeze(estimated_path[:, rfl_index, :])
    motion_index = np.arange(10, np.size(rfl_path, 1) - 10)
    tof = hampel(rfl_path[0, motion_index], int(round(0.5 * (sample_rate / T))), 2e-9)
    aoa = hampel(rfl_path[1, motion_index], int(round(0.5 * (sample_rate / T))), np.pi/180)
    aoa = signal.savgol_filter(aoa, int(round(sample_rate / T - 1)), polyorder=2)  # 代替smooth
    doppler = hampel(rfl_path[2, motion_index], int(round(0.5 * (sample_rate / T))), 6)
    doppler = signal.savgol_filter(doppler, int(round(sample_rate / T - 1)), polyorder=2).squeeze()
    power = rfl_path[3, motion_index]

    # 卡尔曼滤波器
    signal_range,pp = sage_tof_filter(tof * c, -doppler * c / carrier_frequency, T / sample_rate)
    # signal_range[signal_range <= 0.3] = 0.3
    for ii in range(np.size(signal_range, 0)):
        if signal_range[ii] <= 0.3:
            signal_range[ii] = 0.3
    signal_range = signal.savgol_filter(signal_range, int(round(2* sample_rate / T) - 1), polyorder=2)
    signal_range = signal_range + np.linalg.norm(rx_loc - tx_loc)
    aoa = orient - aoa

    location = sage_localization(signal_range.reshape([len(signal_range), 1]), aoa.reshape([len(signal_range), 1]), rx_loc.reshape([len(rx_loc), 1]), xb, yb)
    for ii in range(np.size(location, 0)):
        location[ii, :] = signal.savgol_filter(location[ii, :], int(round(0.5 * sample_rate / T)), polyorder=2)
    scio.savemat(data_path + 'location-' + trace_file, {'location': location})

    plt.plot(ground_truth[0, :], ground_truth[1, :], 'r--')
    plt.plot(location[0, :], location[1,:], 'g--')

    plt.show()

