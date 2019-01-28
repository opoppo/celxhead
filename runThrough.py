import os
import numpy as np
import torch
import cv2
import torchvision
import torch.utils.data as data
import torch.nn as nn
import time
import pretrainedmodels
import torch.multiprocessing as mp
import maskrcnn_benchmark

mp = mp.get_context('spawn')


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, _x):
        return _x.view(self.shape)


class OutputLayer(nn.Module):
    def __init__(self):
        super(OutputLayer, self).__init__()
        # self.fc = nn.Linear(2048, (5,5), bias=True)
        self.fc = nn.Sequential(
            nn.Linear(2048, 3, bias=True),
            # nn.ReLU(inplace=True),
            # Reshape(-1, 255),
            # nn.Linear(225, 3, bias=True),
        )
        self.cls = nn.LogSoftmax(dim=1)

    def forward(self, X):
        y = self.fc(X)
        y = self.cls(y)
        # print(y)
        return y


class InputLayer(nn.Module):
    def __init__(self):
        super(InputLayer, self).__init__()
        # self.fc = nn.Linear(2048, (5,5), bias=True)
        self.fc = nn.Sequential(
            nn.Conv2d(600, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True),
        )

    def forward(self, X):
        y = self.fc(X)
        # print(y.size())
        return y


class OutputLayerInceptionv4(nn.Module):
    def __init__(self):
        super(OutputLayerInceptionv4, self).__init__()
        # self.fc = nn.Linear(2048, (5,5), bias=True)
        self.fc = nn.Sequential(
            nn.Linear(1536, 225, bias=True),
            nn.ReLU(inplace=True),
            Reshape(-1, 255),
            nn.Linear(225, 3, bias=True),
        )

    def forward(self, X):
        y = self.fc(X)
        # print(y.size())
        return y


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_triangular_lr(iteration, stepsize, base_lr, max_lr):
    """Given the inputs, calculates the lr that should be applicable for this iteration"""
    cycle = np.floor(1 + iteration / (2 * stepsize))
    x = np.abs(iteration / stepsize - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))
    return lr


# ====================================================================================================
training = 1  # ????========================================================================================
resume = 0  # ====010:  test model   11X: train model   10X: train new   011: refresh dataset
generateNewSets = 1  # REGENERATE the datasets !!!!!!!!!!!!!!!
# ====================================================================================================


os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

if training and not resume:
    # net = torchvision.models.resnet101(pretrained=True)   #256  10s
    # net.fc = OutputLayer()
    net = pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained='imagenet')  # 136  20s
    net.last_linear = OutputLayer()
    net.layer0 = InputLayer()
    # net = pretrainedmodels.__dict__['inceptionv4'](num_classes=1000, pretrained='imagenet')  # 186  20s
    # net.last_linear = OutputLayerInceptionv4()
    net = torch.nn.DataParallel(net.cuda(), device_ids=[0, 1])

testsize = -1
valsize = -1
if generateNewSets:

    rawset = torchvision.datasets.DatasetFolder('./dataset/', torch.load, ['pt'])

    (trainset, valset, testset) = data.random_split(rawset, [int(len(rawset) * 0.70), int(len(rawset) * 0.15),
                                                             len(rawset) - int(len(rawset) * 0.70) - int(
                                                                 len(rawset) * 0.15)])
    print("datasets ", len(trainset), len(valset), len(testset))
    testsize = len(testset)
    valsize = len(valset)
else:
    pass

train_loader = data.DataLoader(
    dataset=trainset,
    batch_size=32,  # 256 for 4 GPUs
    shuffle=True,
    drop_last=False,
    # pin_memory=True,
    # num_workers=24,
    # sampler=data.SubsetRandomSampler(list(range(0, 3000, 1)))
)
val_loader = data.DataLoader(
    dataset=valset,
    batch_size=2,  # 256 for 4 GPUs
    shuffle=False,
    drop_last=True,
    # pin_memory=True,
    # num_workers=24,
    # sampler=data.SubsetRandomSampler(list(range(3000, 3500, 1)))
)
test_loader = data.DataLoader(
    dataset=testset,
    batch_size=2,  # 256 for 4 GPUs
    shuffle=False,
    drop_last=True,
    # pin_memory=True,
    # num_workers=24,
    # sampler=data.SubsetRandomSampler(list(range(3500, 4239, 1)))
)

if resume and training:
    net = torch.load('nettmp')
    print('net tmp resumed')  # ==============================================================

# Predicting or Testing============
if resume and (not training):
    net = torch.load('nettt')
    print('nettt loaded')

optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001, weight_decay=0.001)
clsloss = nn.NLLLoss(reduction='mean')
# lambda1=lambda epoch: 10**np.random.uniform(-3,-6)
lambda1 = lambda epoch: get_triangular_lr(epoch, 100, 10 ** (-3), 10 ** (-1))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda1)

# training
if training:

    EPOCH = 1
    break_flag = False
    prevvalloss, prevtrainloss = 10e30, 10e30
    ppp = 0
    waitfor = 5  # rounds to wait for further improvement before quit training=================================

    totaltime, losslist = [], []
    net.train()

    for epoch in range(EPOCH):

        scheduler.step()
        time_start = time.time()
        epochTloss = 0
        if break_flag is True:
            break

        for step, (x, targets) in enumerate(train_loader):
            if break_flag is True:
                break
            x = x.type(torch.cuda.FloatTensor)
            targets = targets.type(torch.cuda.LongTensor)

            out = net(x)
            del x

            # print(bboxes_out.size(),bboxes.size())
            loss = clsloss(out.mul_((1 - out.exp()).pow_(2)), targets)  # Focal loss
            # loss = clsloss(out, targets)
            # epochTloss += loss.item()
            epochTloss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        time_end = time.time()
        totaltime.append(time_end - time_start)
        # losslist.append((epoch, epochTloss))
        lr = get_lr(optimizer)
        print("EPOCH", epoch, "  loss_total: %.4f" % epochTloss, "  epoch_time: %.2f" % (time_end - time_start),
              "s   estimated_time: %.2f" % ((EPOCH - epoch - 1) * sum(totaltime) / ((epoch + 1) * 60)), "min with lr=%e"
              % lr)

        if (epoch + 1) % 1 == 0:

            net.eval()
            epochvalloss = 0
            correct = 0
            accuracy = -1

            for step, (x, targets) in enumerate(val_loader):
                x = x.type(torch.cuda.FloatTensor)
                targets = targets.type(torch.cuda.LongTensor)
                out = net(x)
                hit = torch.sum(targets == torch.argmax(out, dim=1))
                correct += hit
                del x

                loss = clsloss(out.mul_((1 - out.exp()).pow_(2)), targets)  # Focal loss
                # epochvalloss += loss.item()
                epochvalloss = loss.item()
            accuracy = correct.detach().cpu().item() / valsize
            print("loss_total: %.4f" % epochvalloss, " on validation  accuracy :  %f" % accuracy)

            if epochvalloss <= prevvalloss and epochTloss <= prevtrainloss:
                torch.save(net, "nettmp")
                print("===improved model saved===")
                prevtrainloss = epochTloss
                prevvalloss = epochvalloss
                ppp = 0
            # else:
            #     ppp += 1
            #     print("===tried round ", ppp, " ===")
            #     if ppp >= waitfor:
            #         net = torch.load('nettmp')
            #         print("===dead end, rolling back to previous model===")
            #         break_flag = True

    torch.save(losslist, "losslist.pt")

#
net.eval()
result = []
epochtestloss = 0
correct = 0
accuracy = -1

for step, (x, targets) in enumerate(test_loader):
    x = x.type(torch.cuda.FloatTensor)
    targets = targets.type(torch.cuda.LongTensor)
    out = net(x)

    hit = torch.sum(targets == torch.argmax(out, dim=1))
    correct += hit

    # print(bboxes_out.size(),bboxes.size())
    # x = x.squeeze_().permute(2, 1, 0)
    # emptyImage = x.cpu().detach().numpy().copy()
    # print(emptyImage.shape,type(emptyImage))
    # emptyImage = cv2.resize(emptyImage, (200, 200), interpolation=cv2.INTER_CUBIC)

    # del x
    # for j, label in enumerate(bboxes.squeeze_().detach().cpu().numpy()):
    # box = bBox_2D(label[0], label[1], label[2], label[3], label[4])
    # # box.Scale(300 / 50, 100, 20)
    # box.bBoxCalcVertxex()
    # cv2.line(emptyImage, box.vertex1, box.vertex2, (155, 255, 255), 1, cv2.LINE_AA)
    # cv2.line(emptyImage, box.vertex2, box.vertex4, (155, 255, 255), 1, cv2.LINE_AA)
    # cv2.line(emptyImage, box.vertex3, box.vertex1, (155, 255, 255), 1, cv2.LINE_AA)
    # cv2.line(emptyImage, box.vertex4, box.vertex3, (155, 255, 255), 1, cv2.LINE_AA)

    # for j, label in enumerate(bboxes_out.squeeze_().detach().cpu().numpy()):
    #     box = bBox_2D(label[0], label[1], label[2], label[3], label[4])
    #     # box.Scale(300 / 50, 100, 20)
    #     # box.Scale(299 / 200, 0, 0)
    #     box.bBoxCalcVertxex()
    #     cv2.line(emptyImage, box.vertex1, box.vertex2, (155, 255, 55), 1, cv2.LINE_AA)
    #     cv2.line(emptyImage, box.vertex2, box.vertex4, (155, 255, 55), 1, cv2.LINE_AA)
    #     cv2.line(emptyImage, box.vertex3, box.vertex1, (155, 255, 55), 1, cv2.LINE_AA)
    #     cv2.line(emptyImage, box.vertex4, box.vertex3, (155, 255, 55), 1, cv2.LINE_AA)

    # emptyImage = cv2.flip(emptyImage, 0)
    # emptyImage = cv2.flip(emptyImage, 1)
    # outImage = cv2.resize(emptyImage, (1000, 1000), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('scan', outImage)
    # print(step)
    # cv2.imwrite('./testset/Result/%d.jpg' % step, outImage)
    # cv2.waitKey()

    loss = clsloss(out.mul_((1 - out.exp()).pow_(2)), targets)  # Focal loss
    epochtestloss = loss.item()

accuracy = correct.detach().cpu().item() / testsize
# print(correct, all, '====')
print("loss_total: %.4f" % epochtestloss, " on testset   accuracy :  %f" % accuracy)

torch.save(result, "result.pt")
torch.save(net, "nettt")
print("====final model saved====")
