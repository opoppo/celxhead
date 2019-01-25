import numpy as np
import os
import SNN
import glob

def loadtxt(file):
    info = np.loadtxt(file)

    x=[]
    y=[]
    pol=[]

    ts1 = info[:, 0]
    ts= ts1 * 0.08     # change unit into us
    x[:] = [int(a) for a in info[:, 1]]
    y[:] = [int(a) for a in info[:, 2]]
    pol[:] = [int(a) for a in info[:, 3]]
    #print(ts)
    return ts, x, y, pol




if __name__ == '__main__':
    readpath = './'
    # for root, dirs, fs in os.walk(readpath):
    #     for f in fs:
    #         str = f.split('.')
    #         print(str[0])
    #         name = '/home/lpg/dataset/headposepng/cor/0/%s/' % str[0]
    #         os.makedirs(name)
    snn = SNN.SNN()
    ts, x, y, pol = loadtxt(readpath+'events.txt')
    snn.spiking(10000, ts, x, y, pol, 'event')


        #file = 'C:/Users/57531/Desktop/headpose_data/2019.01.05/01/Davis346redColor-2019-01-05T20-28-09+0800-00000027-0.aedat'
        # snn = SNN.SNN()
        # ts, x, y, pol = loadtxt(file)
        #
        # snn.spiking(10000, ts, x, y, pol)