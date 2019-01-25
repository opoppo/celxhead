import numpy as np
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2



class SNN():
    ts = []
    x = []
    y = []
    pol = []

    def __init__(self, height = 641, width = 769, threshold=1.1, decay=0.001, margin=1):
        self.threshold = threshold
        self.decay = decay
        self.margin = margin
        self.height = height
        self.width = width

        self.network = np.zeros((self.height, self.width), dtype=np.float64)    # MembranePotential of each pixel
        self.timenet = np.zeros((self.height, self.width), dtype=np.int32)      # Firing time of each neuron (to compute dacay value)
        # self.image = np.zeros((self.height, self.width), dtype=np.uint8)
        self.spike_counter = np.zeros((self.height, self.width), dtype=np.uint8)      # count when exceed threshold

        self.proc_counter = 0       # each stepsize: counter++

    def load_data_to_snn(self, ts, x, y, pol):
        self.ts = ts
        self.x = x
        self.y = y     # set (x0, y0) as (0, 0)
        self.pol = pol

    def spiking(self, stepsize, ts, x, y, pol, name):
        self.load_data_to_snn(ts, x, y, pol)        # load data
        self.name = name
        if len(self.ts) != 0:
            start_time = self.ts[0]
            for idx in range(0, len(self.ts)):
                if self.ts[idx] - start_time < stepsize:  # stepsize is the time interval to process the counting
                    self.neuron_update(idx)
                else:
                    self.show_image(self.proc_counter)

                    self.proc_counter += 1
                    start_time = self.ts[idx]

                    self.spike_counter = np.zeros((self.height, self.width), dtype=np.uint8)
        else:
            print('Please load Davis Data using : loadDataToSnn(ts, x, y, pol)')
            return

    def neuron_update(self, idx):

        x = self.x[idx]
        y = self.y[idx]
        escape_time = (self.ts[idx]-self.timenet[y][x])/1000.0  #change unit to us
        residual = max(self.network[y][x]-self.decay*escape_time, 0)
        self.network[y][x] = residual + 1       # spiking value = 1
        self.timenet[y][x] = self.ts[idx]

        if self.network[y][x] > self.threshold:
            self.spike_count([x, y])
            self.neuron_clear([x, y])

    def spike_count(self, position):
        self.spike_counter[position[1]][position[0]] += 1     # position[1]:x  position[0]:y

    def neuron_clear(self, position):
        for i in range((-1)*self.margin, self.margin):
            for j in range((-1)*self.margin, self.margin):
                if position[0]+i < 0 or position[0]+i > self.width or position[1]+j < 0 or position[1]+j > self.height:
                    continue
                else:
                    self.network[position[1]+j][position[0]+i] = 0.0

    def show_image(self,num_image):
        image = np.zeros((641, 769), dtype=np.uint8)
        # show SNN-filtered image
        for i in range(0, self.width):
            for j in range(0, self.height):
                grayscale = (int(255 * (1.0 / (1 + np.exp(-int(self.spike_counter[j][i]))))) - 127) * 2
                image[j][i] = grayscale
        #image = cv2.flip(image, 0)
        cv2.imshow('img', image)
        cv2.waitKey(2)
        print(num_image)
        cv2.imwrite(self.name+'/%05d.png' % num_image, image)







