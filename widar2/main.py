import scipy.io as scio
import numpy as np
from widar3 import bvp
import matplotlib.pyplot as plt
from widar2 import widar2 as widar

if __name__ == '__main__':
    data_path = '../MATLAB_data/'
    ddata_path = 'MATLAB_org_data/'
    static_file = 'S01.mat'
    trace_file = 'T04.mat'
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

    orient = widar.static_analize()
    count = 0
    step = 1000
    pre_buffer = widar.dynamic_pre_buffer_build()
    plt.subplot(2, 3, 6)
    plt.plot(location_truth[0, :], location_truth[1, :], 'g--')
    while True:
        print(count)
        csi_data = ccsi_data[count*step: (count+1)*step]
        time_stamp = ctime_stamp[count: count + step]
        pre_buffer = widar.dynamic_analize(csi_data, time_stamp, count, step, orient, pre_buffer)
        count = count + 1