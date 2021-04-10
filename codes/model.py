from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k=64):
        super(TNet, self).__init__()
        # Each layer has batchnorm and relu on it
        # conv 3 64
        self.k = k
        self.conv1 = nn.Sequential(
            nn.Conv2d(k, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # conv 64 128
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # conv 128 1024
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        # max pool

        # fc 1024 512
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        # fc 512 256
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        # fc 256 k*k (no batchnorm, no relu)
        self.fc3 = nn.Linear(256, k*k, bias=False)

        # add bias 1, k*k
        # self.bias = torch.rand((1, k*k), dtype=torch.float32, device=torch.device('cuda'), requires_grad=True)
        self.bias = torch.rand((1, k*k), dtype=torch.float32, device=torch.device('cpu'), requires_grad=True)
        # reshape
    
    def forward(self, x):
        # print('TNet')
        x = torch.unsqueeze(x, dim=3)  # [b, 3||64, n, 1]
        x = self.conv1(x)  # [b, 64, n, 1]
        # print('TNet conv1 ', x.shape)
        x = self.conv2(x)  # [b, 128, n, 1]
        # print('TNet conv2 ', x.shape)
        x = self.conv3(x)  # [b, 1024, n, 1]
        # print('TNet conv3 ', x.shape)
        # max pool
        x = torch.max(x, dim=2)[0]  # [b, 1024, 1]
        # print('TNet after max ', x.shape)
        x = x.view(-1, 1024)  # [b, 1024]
        # print('TNet after max view ', x.shape)

        x = self.fc1(x)  # [b, 512]
        # print('TNet fc1 ', x.shape)
        x = self.fc2(x)  # [b, 256]
        # print('TNet fc2 ', x.shape)
        x = self.fc3(x)  # [b, 9||64^2]
        # print('TNet fc3 ', x.shape)

        # k = torch.sqrt(x.shape[1])  # k
        x = x.view(-1, self.k * self.k)  # [b, 9||64^2]
        # print('TNet view k*k ', x.shape)
        x = torch.add(x, self.bias)

        # reshape
        x = x.view(-1, self.k, self.k)  # [b, 3, 3]
        # print('TNet final reshape n*k*k ', x.shape)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetfeat, self).__init__()
        self.feature_transform = feature_transform
        self.global_feat = global_feat
        # Use TNet to apply transformation on input and multiply the input points with the transformation
        self.trans1 = TNet(k=3)
        # conv 3 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Use TNet to apply transformation on features and multiply the input features with the transformation 
        #                                                                        (if feature_transform is true)
        if feature_transform:
            self.trans2 = TNet(k=64)
        # conv 64 128
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # conv 128 1024 (no relu)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 1024, kernel_size=1),
            nn.BatchNorm2d(1024)
        )
        # max pool

    def forward(self, x):  # x [b, 3, n]
        n_pts = x.size()[2]
        origin_points = x
        # print('PointNetfeat input x', x.shape)
        # You will need these extra outputs:
        # trans = output of applying TNet function to input
        # trans_feat = output of applying TNet function to features (if feature_transform is true)
        trans = self.trans1(x)  # [b, 3, 3]
        # print('PointNetfeat Tnet layer, k=3', trans.shape)
        x = torch.bmm(trans, x)  # [b, 3, n]
        # print('PointNetfeat bmm', x.shape)
        x = torch.unsqueeze(x, dim=3)  # [b, 3, n, 1]
        # print('PointNetfeat add dim', x.shape)
        x = self.conv1(x)  # [b, 64, n, 1]
        # print('PointNetfeat conv1', x.shape)
        x = torch.squeeze(x, dim=3)  # [b, 64, n]

        if self.feature_transform:
            # x = torch.squeeze(x, dim=3)  # [b, 64, n]
            # print('PointNetfeat feature transform remove dim', x.shape)
            trans_feat = self.trans2(x)  # [b, 64, 64]
            # print('PointNetfeat feature transform Tnet layer, k=64', trans_feat.shape)
            x = torch.bmm(trans_feat, x)  # [b, 64, n]
            # print('PointNetfeat feature transform bmm', x.shape)
        else:
            trans_feat = None

        pointfeat = x
        x = torch.unsqueeze(x, dim=3)  # [b, 64, n, 1]
        x = self.conv2(x)  # [b, 128, n, 1]
        # print('PointNetfeat conv2', x.shape)
        x = self.conv3(x)  # [b, 1024, n, 1]
        # print('PointNetfeat conv3', x.shape)
        x = torch.squeeze(x, dim=3)  # [b, 1024, n]

        # max pool
        indices = torch.max(x, dim=2)[1]
        # print('ind shape is', indices.shape)
        # print(indices[0])
        x = torch.max(x, dim=2)[0]  # [b, 1024]
        # print('PointNetfeat after max', x.shape)

        critical_points = torch.index_select(origin_points[0], 1, indices[0])
        critical_points = torch.transpose(critical_points, 0, 1)

        x = x.view(-1, 1024)

        if self.global_feat:  # This shows if we're doing classification or segmentation
            return x, trans, trans_feat, critical_points
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat, critical_points = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat, critical_points


class PointNetDenseCls(nn.Module):
    # k is the number of parts for this class of objects
    # output is k scores for each point in each point cloud in the batch
    def __init__(self, k=2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        # get global features + point features from PointNetfeat
        self.feature = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        # conv 1088 512
        self.conv1 = nn.Sequential(
            nn.Conv2d(1088, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # conv 512 256
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # conv 256 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # conv 128 k
        self.conv4 = nn.Conv2d(128, k, kernel_size=1)
        # softmax

    def forward(self, x):  # [b, 3, n] [4, 3, 2500] by default
        # You will need these extra outputs: 
        # trans = output of applying TNet function to input
        # trans_feat = output of applying TNet function to features (if feature_transform is true)
        # (you can directly get them from PointNetfeat)
        # print('PointNetDenseCls input x', x.shape)
        x, trans, trans_feat = self.feature(x)  # [b, 1088, n]
        # print('PointNetDenseCls after PointNetfeat', x.shape)
        x = torch.unsqueeze(x, dim=3)  # [b, 1088, n, 1]
        x = self.conv1(x)  # [b, 512, n, 1]
        # print('PointNetDenseCls conv1', x.shape)
        x = self.conv2(x)  # [b, 256, n, 1]
        # print('PointNetDenseCls conv2', x.shape)
        x = self.conv3(x)  # [b, 128, n, 1]
        # print('PointNetDenseCls conv3', x.shape)
        x = self.conv4(x)  # [b, 4, n, 1]
        # print('PointNetDenseCls conv4', x.shape)
        x = torch.squeeze(x, dim=3)  # [b, 4, n] k=4
        x = x.transpose(2, 1)  # [b, n, k]
        # print('PointNetDenseCls after transpose', x.shape)
        x = F.softmax(x, dim=2)
        # print('PointNetDenseCls after softmax', x.shape)
        return x, trans, trans_feat


def feature_transform_regularizer(trans):
    # compute |((trans * trans.transpose) - I)|^2
    b, k, _ = trans.shape  # b,k,n
    trans_t = torch.transpose(trans, 1, 2)
    i = torch.eye(k, device=torch.device('cuda'))  # identity matrix
    i = i.repeat(b, 1, 1)  # b,k,k
    temp = torch.tensor(torch.bmm(trans, trans_t) - i)
    loss = torch.norm(temp, dim=[1, 2])  # tensor, list
    loss = torch.mean(loss)
    return loss


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32, 3, 2500))
    trans = TNet(k=3)
    out = trans(sim_data)
    print('TNet', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = TNet(k=64)
    out = trans(sim_data_64d)
    print('TNet 64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k=5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k=3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
