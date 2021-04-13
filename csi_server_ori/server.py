# -*- coding:utf-8 -*-
import socket
import numpy as np
from Bfee import Bfee
from get_scale_csi import get_scale_csi
import matplotlib.pyplot as plt
import threading

plt.figure(1)
csi = []
MAXCSI = 1000
mySocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #   创建套接字


def sock():
    #   设置IP和端口
    global csi
    host = socket.gethostname()  # sudo vi /etc/hosts replace the ip
    ip = socket.gethostbyname(host)
    port = 3333
    print("hostname: " + host + ", host ip: " + ip + ', host port:' + str(port))
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
        print("client IP is %s" % address[0])
        print("client port is %d" % address[1])

        count = 0
        while True:
            try:
                msg = client.recv(4)
                ll = int.from_bytes(msg, byteorder='little', signed=False)

                msg2_len = 0
                msg2 = b''
                while msg2_len < ll:  # read buffer for many times
                    msg2_temp = client.recv(390000)
                    msg2 = msg2 + msg2_temp  # combine bytes
                    msg2_len = msg2_len + len(msg2_temp)
                count = count + 1
            except:
                client.close()
                mySocket.close()
                print("连接断开")
                setup_connect()
                print("重新建立套接字")
                continue

            if msg == b"":
                print("程序结束2\n")
                break
            else:
                with open('csi_res.dat', 'wb+') as f:
                    # qq = '\0'
                    # kk = qq+msg.decode('utf-8', "ignore")  # 非严格格式，忽略非法字符
                    # f.write(kk.encode("utf-8", "ignore"))
                    f.write(msg2)

            try:
                bfee = Bfee.from_stream(msg2)
                for i in range(len(bfee.all_csi)):
                    csi_temp = get_scale_csi(bfee.dicts[i])
                    csi.append(np.abs(csi_temp))
            except:
                print("有错误2")
                continue
            display()
            csi = []


def setup_connect():
    #   设置IP和端口
    host = socket.gethostname()
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


def display():
    plt.cla()
    step = int(MAXCSI / 100)
    count = 0
    # plt.legend()
    try:
        csi_tmp = []
        for csi_k in csi:  # 载波数  csi_k=30*3*2
            if count % step == 0:
                csi_tmp.append(csi_k[15, 0, 0])  # csi_k[0]=3*2
            count = count + 1
        cc = np.array(csi_tmp)  # cc=100*2
        lenc = len(cc)
        x = np.arange(0, lenc, 1)
        if lenc <= MAXCSI:
           plt.plot(x, cc)
        else:
           plt.plot(x[lenc-1-MAXCSI:], cc[lenc-1-MAXCSI:])

        plt.ylim([0, 50])
        plt.show()
        plt.pause(0.01)  # 需要暂停否则无法显示
    except:
        print("display有错误")


def main():
    sck = threading.Thread(target=sock())
    dis = threading.Thread(target=display())
    dis.start()
    sck.start()


# 多网卡情况下，根据前缀获取IP
def GetLocalIPByPrefix(prefix):
    localIP = ''
    print(socket.gethostbyname_ex(socket.gethostname()))
    for ip in socket.gethostbyname_ex(socket.gethostname())[1]:
        print(ip)
        if ip.startswith(prefix):
            localIP = ip
    return localIP


if __name__ == '__main__':
    try:
        sock()
    except:
        mySocket.close()
