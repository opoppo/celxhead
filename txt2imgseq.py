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

            count = count + 1

    return ts, x, y, pol


if __name__ == "__main__":

    for i in range(16):
        txtpath = './txt/%d' % i + '/cortxt/'
        imgpath = './img/%d' % i + '/'
        for fpathe, dirs, fs in os.walk(txtpath):
            for p, f in enumerate(fs):

                if p%3==0:
                    imgoutputpath = imgpath +'/updown/'+ f.split('.')[0] + '/'
                elif (p-1)%3==0:
                    imgoutputpath = imgpath + '/left/' + f.split('.')[0] + '/'
                elif (p - 2) % 3 == 0:
                    imgoutputpath = imgpath + '/right/' + f.split('.')[0] + '/'

                file = txtpath + f


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
                    data_t = np.array(t[start_idx:idx]).reshape((-1, 1))
                    data = np.column_stack((data_x, data_y, data_t))
                    data_filter = data

                    for i in range(0, data_filter.shape[0]):
                        img[int(data_filter[i][1] - 1)][int(data_filter[i][0] - 1)][1] = 255  # channel NONE
                        img[int(data_filter[i][1] - 1)][int(data_filter[i][0] - 1)][0] += 85  # channel frequency
                        img[int(data_filter[i][1] - 1)][int(data_filter[i][0] - 1)][2] = 255 * (
                                data_filter[i][2] - t[start_idx]) / (t[idx] - t[start_idx])  # channel time stamp

                    start_idx = idx
                    startTime = t[idx]
                    # print(sum(sum(img)))
                    # if sum(sum(sum(img))) > 480:
                    # img = cv2.flip(img, 0)
                    cv2.imshow('dvs', img)
                    cv2.waitKey(5)
                    imgFullFile = imgoutputpath + ('%05d' % imgCount) + '.png'
                    cv2.imwrite(imgFullFile, img)
                    imgCount = imgCount + 1

                    img[:] = 0
                    # print('.')
