import numpy as np
import cv2
import os


def loadTxt(file):
    count = 0
    ts = []
    x = []
    y = []
    pol = []
    with open(file) as f:
        for line in f:
            items = line.split()
            ts.insert(count, float(items[0]))
            x.insert(count, int(items[1]))
            y.insert(count, int(items[2]))
            pol.insert(count, int(items[3]))
            # print(items)

            count = count + 1

    return ts, x, y, pol


if __name__ == "__main__":

    for i in range(2):
        txtpath='./txt/%d'%i+'/cortxt/'
        imgpath='./img/%d'%i+'/'
        for fpathe, dirs, fs in os.walk(txtpath):
            for f in fs:

                file = txtpath+f
                imgoutputpath = imgpath+f.split('.')[0]+'/'

                if not os.path.exists(imgoutputpath):
                    os.makedirs(imgoutputpath)

                t, x, y, pol = loadTxt(file)
                x[:] = [int(a - 1) for a in x]
                y[:] = [int(a - 1) for a in y]

                img = np.zeros((640, 768, 3), dtype=np.uint8)

                idx = 0
                start_idx = 0
                startTime = 0
                endTime = 0
                stepTime = 10000 / 0.08
                imgCount = 1

                while startTime < t[-1]:
                    endTime = startTime + stepTime
                    while t[idx] < endTime and idx < len(t) - 1:
                        idx = idx + 1

                    data_x = np.array(x[start_idx:idx]).reshape((-1, 1))
                    data_y = np.array(y[start_idx:idx]).reshape((-1, 1))
                    data = np.column_stack((data_x, data_y))
                    data_filter = data

                    for i in range(0, data_filter.shape[0]):
                        img[data_filter[i][1] - 1][data_filter[i][0] - 1][0] = 255  # channel NONE
                        img[data_filter[i][1] - 1][data_filter[i][0] - 1][1] += 1  # channel frequency
                        img[data_filter[i][1] - 1][data_filter[i][0] - 1][2] = t[i]  # channel time stamp

                    img[:][:][1] = 255 * 2 * (1 / (1 + np.exp(-img[:][:][1])) - 0.5)
                    # a=img[:][:][1]
                    # print( a[a>0] )
                    img[:][:][2] = 255 * (img[:][:][2] - startTime) / stepTime
                    # a=img[:][:][2]
                    # print( a[a>0] )

                    start_idx = idx

                    # img = cv2.flip(img, 0)
                    cv2.imshow('dvs', img)
                    cv2.waitKey(5)
                    imgFullFile = imgoutputpath + ('%05d' % imgCount) + '.png'
                    cv2.imwrite(imgFullFile, img)

                    img[:] = 0
                    startTime = t[idx]
                    imgCount = imgCount + 1
