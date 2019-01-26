import numpy as np
import cv2
import os

import torch

if __name__ == "__main__":
    count = 0

    for i in range(16):  # for each person
        # txtpath = './txt/%d' % i + '/cortxt/'
        imgpath = './img/%d' % i + '/'
        imgoutputpath = './img_output/%d' % i + '/'
        for category in os.listdir(imgpath):  # for each action category
            for seq in os.listdir(imgpath + category):  # for each action sequence
                imgseq = []
                for img in os.listdir(imgpath + category + '/' + seq):  # for each image(frame) in the sequence
                    # print(imgpath+category+'/'+seq+'/'+img)
                    img = cv2.imread(imgpath + category + '/' + seq + '/' + img)
                    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                    img = torch.from_numpy(img)
                    imgseq.append(img)
                imgtensor = torch.cat(imgseq,dim=-1)
                imgtensor.resize_(224,224,600)
                outpath = imgoutputpath + category + '/'
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
                torch.save(imgtensor, imgoutputpath + category + '/' + seq+'.pt' )
                count += 1
                print(count, '///', imgtensor.size())
