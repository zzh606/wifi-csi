import picos as pic
import numpy as np
import scipy.io as scio
import sys
import math
import cmath
import copy
import matplotlib.pyplot as plt
# import ecos

# plt.figure(1)


def sage_optimal_initialization(G, L): # G=6,L=2
    prob = pic.Problem()
    variables = prob.add_variable('variables', G*G*L*L, vtype='binary')
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
    # constraints = [constraints variables(ii,ii,jj,jj) == 1]
    prob.add_list_of_constraints([variables[G*L*L*ii + L*L*ii + L*jj + jj] == 1 for (ii, jj) in IJ])

    for ii in range(G):
        for jj in range(L):
            # constraints = [constraints squeeze(variables(ii,:,jj,:)) == squeeze(variables(:,ii,:,jj))];
            prob.add_list_of_constraints([variables[G*L*L*ii + L*L*kk + L*jj + ll] == variables[G*L*L*kk + L*L*ii + L*ll + jj] for (kk, ll) in IJ])

    # constraints = [constraints sum(sum(sum(variables, 2), 3), 4) == G * L]
    prob.add_list_of_constraints([pic.sum(variables[G*L*L*ii: G*L*L*(ii+1)]) == G*L for ii in range(G)])

    # constraints = [constraints sum(variables,4) == 1]
    prob.add_list_of_constraints([pic.sum(variables[G*L*L*ii + L*L*jj + L*kk: G*L*L*ii + L*L*jj + L*(kk+1)]) == 1 for (ii, jj, kk) in IJK])

    for ii in range(G):
        for jj in range(G):
            if ii == jj:
                continue
            for mm in range(L):
                for nn in range(L):
                    # constraints = [constraints variables(ii,jj,mm,nn) + squeeze(variables(jj,:,nn,:)) <= 1 + squeeze(variables(ii,:,mm,:))];
                    prob.add_list_of_constraints([variables[G*L*L*ii + L*L*jj + L*mm + nn] + variables[G*L*L*jj + L*L*pp + L*nn + qq] <= \
                                                  1 + variables[G*L*L*ii + L*L*pp + L*mm + qq] for (pp, qq) in PQ])

    return prob, variables


def sage_optimal_matching(estimated_parameter, optimize_variables, optimize_problem):
    L = np.size(estimated_parameter, 1)
    G = np.size(estimated_parameter, 2)

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
    optimize_problem.solve(verbose=0, solver='ecos')
    print(optimize_problem)

    edges = copy.copy(optimize_variables)
    edges = (edges > 0.5) + 0

    estimated_index = np.zeros([L, G])
    estimated_index[:, 0] = np.arange(0, L)

    for ii in range(1, G):
        for jj in range(L):
            estimated_index[jj, ii] = edges[L*L*ii + L*jj: L*L*ii + L*(jj+1)]
        estimated_parameter[:, :, ii] = estimated_parameter[:, estimated_index[:, ii], ii]

    return estimated_parameter, estimated_index


def sage_path_mapping(estimated_parameter, parameter_weight, N, S):
    K = np.size(estimated_parameter, 0)
    G = np.size(estimated_parameter, 2)
    L = np.size(estimated_parameter, 1)
    normalized_parameter = copy.copy(estimated_parameter)
    normalized_parameter[3, :, :] = np.abs(normalized_parameter[3, :, :])

    for ii in range(0, G):
        normalized_parameter[3, :, ii] = normalized_parameter[3, :, ii] / np.sum(normalized_parameter[3, :, ii])

    matrix_weight = np.zeros([K, L, G], dtype=complex)
    for kk in range(np.size(matrix_weight, 2)):
        for jj in range(np.size(matrix_weight, 1)):
            matrix_weight[:, jj, kk] = copy.copy(parameter_weight)

    normalized_parameter = np.multiply(normalized_parameter, matrix_weight)
    M = int(np.floor((G - 1) / (N - 1)))
    optimal_problem, optimal_variables = sage_optimal_initialization(int((N + 1) / 2), int(np.size(estimated_parameter, 1)))

    for ii in range(M):
        if ii == 0:
            col_index = ii * (N - 1) + np.arange(0, N, 2)
            estimated_index = sage_optimal_matching(normalized_parameter[:, :, col_index], optimal_variables, optimal_problem)
        else:
            col_index = ii * (N - 1) + np.arange(2, N, 2) #....
            temp_parameter = np.zeros([np.size(normalized_parameter, 0), L, len(col_index)+1])
            # temp_parameter[:, :, 0] = group_parameter
            temp_parameter[:, :, 1:] = normalized_parameter[:, :, col_index]
            estimated_index = sage_optimal_matching(temp_parameter, optimal_variables, optimal_problem)

        for jj in range(len(col_index)):
            normalized_parameter[:, :, col_index[jj]] = normalized_parameter[:, estimated_index[:, jj], col_index[jj]]
            estimated_parameter[:, :, col_index[jj]] = estimated_parameter[:, estimated_index[:, jj], col_index[jj]]

        col_index = ii * (N - 1) + np.arange(1, N, 2)
        for jj in range(len(col_index)):
            kk = col_index[jj]
            span_index = np.arange(np.max(0, kk-(2*S-1)), np.min(col_index[len(col_index)-1], kk+(2*S-1)))
            estimated_index = sage_greedy_matching(normalized_parameter[:, :, kk], normalized_parameter[:, :, span_index])
            normalized_parameter[:, :, kk] = normalized_parameter[:, estimated_index, kk]

        col_index = ii * (N-1) + np.arange(N)
        group_parameter = np.median(normalized_parameter[:, :, col_index], 2)

        for ii in range(M*(N-1), G):
            span_index = np.arange(ii-S+1, ii)
            estimated_index = sage_greedy_matching(normalized_parameter[:, :, ii], normalized_parameter[:, :, span_index])
            normalized_parameter[:, :, ii] = normalized_parameter[:, estimated_index, ii]
            estimated_parameter[:, :, ii] = estimated_parameter[:, estimated_index, ii]


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
    # plt.plot(estimated_parameter_temp[0])
    # plt.show()
    estimated_parameter_temp = sage_path_mapping(estimated_parameter_temp, parameter_weight, N, S)


if __name__ == '__main__':
    data_path = 'MATLAB_data/'
    ddata_path = 'MATLAB_org_data/'
    static_file = 'S01.mat'
    trace_file = 'T01.mat'
    parameter_weight = np.array([1e9, 180 / np.pi, 1, 1]).reshape([1, 4]) * np.array([1 / 200, 1 / 90, 1 / 80, 10]).reshape(
        [1, 4])
    sample_rate = 1000
    T = 100
    MN = round(sample_rate / T) + 1
    MS = 3

    ddata = scio.loadmat(ddata_path + 'PPM-' + static_file)
    estimated_parameter = ddata['estimated_parameter']

    x = np.arange(0,np.size(estimated_parameter,1))

    # plt.plot(estimated_parameter[0],'bo',ms=1)
    # plt.show()

    estimated_path = sage_path_mapping_filter(estimated_parameter, parameter_weight, MN, MS)

    # estimated_path = sage_path_mapping_filter(estimated_parameter, parameter_weight, MN, MS)
    # scio.savemat(data_path + 'PMM-' + static_file, {'estimated_path': estimated_path})


