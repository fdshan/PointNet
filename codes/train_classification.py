# reference: https://github.com/fxia22/pointnet.pytorch

from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset import ShapeNetDataset
from model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int,
                    default=32, help='input batch size')
parser.add_argument('--num_points', type=int,
                    default=2500, help='input batch size')
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=4)
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--nepoch', type=int, default=250,
                    help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--feature_transform', default=False,
                    action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

def blue(x): return '\033[94m' + x + '\033[0m'


opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


# dataset = ShapeNetDataset(root='shapenetcore_partanno_segmentation_benchmark_v0', classification=True, npoints=opt.num_points)
dataset = ShapeNetDataset(
    root=opt.dataset, classification=True, npoints=opt.num_points)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

testset = ShapeNetDataset(
    root=opt.dataset, classification=True, npoints=opt.num_points)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))


print(len(dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(
    k=num_classes, feature_transform=opt.feature_transform)
if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
classifier.cuda()
num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        # TODO
        # back-propagate loss = cls_loss + 0.001*feature_transform_regularizer()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat, critical_points = classifier(points)
        # print(critical_points.shape)
        np.savetxt("critical_points.txt",
                   critical_points.cpu().detach().numpy())
        # loss = F.cross_entropy(pred, target) + 0.001 * feature_transform_regularizer(trans)
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += 0.001 * feature_transform_regularizer(trans_feat)
        loss.backward()
        optimizer.step()
        pred_label = pred.data.max(1)[1]
        correct = pred_label.eq(target.data).cpu().sum()

        print('[{0},{1}] training loss: {2}'.format(epoch, i, loss.item()))
    if opt.feature_transform:
        torch.save(classifier.state_dict(
        ), '{0}/weights_with_transform/cls_model_{1}.pth'.format(opt.outf, epoch))
    else:
        torch.save(classifier.state_dict(
        ), '{0}/weights_without_transform/cls_model_{1}.pth'.format(opt.outf, epoch))
    print('{} weights saved!'.format(epoch))

print('Train Finish!')

total_correct = 0
total_testset = 0
for i, data in enumerate(testloader, 0):
    # TODO
    # calculate average classification accuracy
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()

    pred, _, _ = classifier(points)
    pred_label = pred.data.max(1)[1]
    correct = pred_label.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("Test accuracy {}".format(total_correct / float(total_testset)))
