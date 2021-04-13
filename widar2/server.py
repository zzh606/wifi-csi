# -*- coding:utf-8 -*-
import socket
import numpy as np

import matplotlib.pyplot as plt
import scipy.io as scio
from Bfee import Bfee
from widar2.widar2 import sage_const_build, sage_generate_steering_vector, dynamic_analize, dynamic_pre_buffer_build
from get_scale_csi import get_scale_csi

def static_step(client):
    # SAGE parameters
    # F:子载波数，T：包数，TR：ToF范围，A（S）：天线数（传感器数）
    # FI:频率整数，TI：时间整数，AS：天线空间，
    # AR：AOA范围，DR：多普勒范围，UR：更新率
    # L：传播路径数，
    # L、N在本算法中无用处
    # m=(i,j,k)，其中i=0,1,...,T，j=0,1,...,F，k=0,1,...,A
    # m is the hyper-domain for CSI measurement H(i, j, k)

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
    # Path matching parameters
    MN = round(sample_rate / T) + 1
    MS = 3
    csi_data = []
    parameter_weight = np.array([1e9, 180 / np.pi, 1, 1]).reshape([1, 4]) * np.array(
        [1 / 200, 1 / 90, 1 / 80, 10]).reshape([1, 4])

    sage_const = sage_const_build(T, F, A, L, N, FI, TI, AS, TR, AR, DR, UR)
    sage_const = sage_generate_steering_vector(sage_const)

    while True:
        ## 从Socket获得msg
        try:
            msg = client.recv(2)
            l = int.from_bytes(msg, byteorder='big', signed=False)
            msg2 = client.recv(l)
            bfee = Bfee.from_stream(msg2)
            if msg == b"":
                print("static_step结束\n")
                return -1
        except:
            client.close()
            mySocket.close()
            print("连接断开")
            setup_socket(mySocket)
            print("重新建立套接字")
            continue

        ## 解析出csi_data
        for i in range(len(bfee.all_csi)):
            csi_temp = get_scale_csi(bfee.dicts[i])
            csi_data.append(np.abs(csi_temp))


def dynamic_step(client, mySocket, orient):
    pre_buffer = dynamic_pre_buffer_build()
    count = 0
    step = 1000
    while True:
        try:
            msg = client.recv(2)
            l = int.from_bytes(msg, byteorder='big', signed=False)
            msg2 = client.recv(l)
            bfee = Bfee.from_stream(msg2)
            if msg == b"":
                print("static_step结束\n")
                return -1
        except:
            client.close()
            mySocket.close()
            print("连接断开")
            setup_socket(mySocket)
            print("重新建立套接字")
            continue

        time_stamp = bfee.all_timestamp
        csi_data = []
        for i in range(len(bfee.all_csi)):
            csi_temp = get_scale_csi(bfee.dicts[i])
            csi_data.append(np.abs(csi_temp))

        dynamic_analize(csi_data, time_stamp, count, step, orient, pre_buffer)
        count = count + 1
        client.sendall('idle')  # 若完成则接收下一次数据

def run(mySocket, MAXCSI):
    #   设置IP和端口
    host = socket.gethostname()
    ip = socket.gethostbyname(host)
    print("ip:", ip)
    port = 3333
    #   bind绑定该端口
    mySocket.bind((host, port))
    #   监听

    while True:
        print("程序开始")
        mySocket.listen(10)
        #   接收客户端连接
        print("等待连接....")
        client, address = mySocket.accept()
        print("新连接")
        print("IP is %s" % address[0])
        print("port is %d\n" % address[1])

        analyze_flag = 0
        while True:
            try:
                if analyze_flag == 0:
                    orient = static_step(client, mySocket)
                    analyze_flag = 1
                else:
                    dynamic_step(client, mySocket, orient)
            except:
                print('分析过程有错误')

            # if msg == b"":
            #     print("程序结束2\n")
            #     break
            # else:
            #     with open('csi_res.dat', 'wb+') as f:
            #         qq = '\0'
            #         kk = qq+msg.decode('utf-8', "ignore")  # 非严格格式，忽略非法字符
            #         f.write(kk.encode("utf-8", "ignore"))
            #         f.write(msg2)

            # if count % 10 == 0:
            # display(csi_data, MAXCSI)


def setup_socket(mySocket):
    #   设置IP和端口
    host = socket.gethostname()
    ip = socket.gethostbyname(host)
    port = 3333
    #   bind绑定该端口
    mySocket.bind((host, port))
    #   监听
    mySocket.listen(10)
    print("等待连接....")
    client, address = mySocket.accept()
    print("新连接")
    print("IP is %s" % address[0])
    print("port is %d\n" % address[1])


def display(csi, MAXCSI):
    plt.cla()
    # plt.legend()
    try:
        for i in range(len(csi[0])):  # 数据包数量
            c_tmp = []

            for j in csi:  # 载波数
                q = j[i][0]
                c_tmp.append(q)
            cc = np.array(c_tmp)
            lenc = len(cc)
            x = np.arange(0, lenc, 1)
            if lenc <= MAXCSI:
               plt.plot(x, cc)
            else:
               plt.plot(x[lenc-1-MAXCSI:], cc[lenc-1-MAXCSI:])
            break # 临时加入
        plt.ylim([0, 30])
        plt.show()
        plt.pause(0.01)  # 需要暂停否则无法显示
    except:
        print("display有错误")


if __name__ == '__main__':
    data_path = '../MATLAB_data/'
    cdata = scio.loadmat(data_path + 'device_config.mat')
    rx_loc = cdata['rx_loc']  # 接收机位置
    tx_loc = cdata['tx_loc']  # 发射机位置
    c = np.squeeze(cdata['c'])  # 光速
    carrier_frequency = np.squeeze(cdata['carrier_frequency'])  # 载波频率
    yb = np.squeeze(cdata['yb'])  # y轴取值范围
    xb = np.squeeze(cdata['xb'])  # x轴取值范围

    plt.figure(1)
    MAXCSI = 100
    mySocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建套接字

    run(mySocket, MAXCSI)
