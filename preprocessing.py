import numpy as np
import cv2
import os
import shutil
import torch

if __name__ == "__main__":
    count = 0

    for i in range(16):  # for each person
        # txtpath = './txt/%d' % i + '/cortxt/'
        imgoutputpath = './img_output/%d' % i + '/'
        for category in os.listdir(imgoutputpath):  # for each action category
            datasetpath = './dataset/%s' % category + '/'
            for seqtensor in os.listdir(imgoutputpath + category):  # for each action sequence
                if not os.path.exists(datasetpath):
                    os.makedirs(datasetpath)
                shutil.move(imgoutputpath + category + '/' + seqtensor,
                                datasetpath + seqtensor)
