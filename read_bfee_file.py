import numpy as np
from Bfee import Bfee
from get_scale_csi import get_scale_csi
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    inputfile = 'csi_res.dat'
    csi = []

    # inputfile = str(sys.argv[1])
    # bfee = Bfee.from_file(inputfile, model_name_encode="gb2312")
    bfee = Bfee.from_file_atheros(inputfile, model_name_encode="gb2312")

    for i in range(len(bfee.all_csi)):
        # csi_temp = get_scale_csi(bfee.dicts[i])
        csi_temp = bfee.all_csi[i]
        csi.append(csi_temp)
    plt.figure(1)
    x = np.arange(len(csi))
    for i in range(len(csi[0])):  # i=0~55
        c_tmp = []
        for j in csi:  # j=(56,3,1)
            q = np.abs(j[i, 0, 0])
            c_tmp.append(q)
        cc = np.array(c_tmp)
        plt.plot(x, cc)
        break
    plt.show()

