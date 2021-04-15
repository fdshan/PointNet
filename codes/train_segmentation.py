from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset import ShapeNetDataset
from model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int,
                    default=4, help='input batch size')
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=25,
                    help='number of epochs to train for')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--class_choice', type=str,
                    default='Chair', help="class_choice")
parser.add_argument('--feature_transform', default=False,
                    action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = ShapeNetDataset(
    root=opt.dataset, classification=False, class_choice=[opt.class_choice])
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

test_dataset = ShapeNetDataset(root=opt.dataset, classification=False, class_choice=[
                               opt.class_choice], split='test', data_augmentation=False)
testdataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = dataset.num_seg_classes
print('classes', num_classes)
try:
    os.makedirs(opt.outf)
except OSError:
    pass

def blue(x): return '\033[94m' + x + '\033[0m'


classifier = PointNetDenseCls(
    k=num_classes, feature_transform=opt.feature_transform)
if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=20, gamma=0.5)  # adjust lr
classifier.cuda()

num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        # target = torch.flatten(target)  # target [k*n]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        # TODO
        # back-propagate loss = seg_loss + 0.001*feature_transform_regularizer()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)  # pred [b, n, k]

        pred = pred.view(-1, num_classes)  # ([10000, 4])
        target = torch.flatten(target) - 1

        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += 0.001 * feature_transform_regularizer(trans_feat)
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[{0},{1}] training loss: {2}'.format(epoch, i, loss.item()))

    if opt.feature_transform:
        torch.save(classifier.state_dict(
        ), '{0}/weights_with_transform/seg_model_{1}_{2}.pth'.format(opt.outf, opt.class_choice, epoch))
    else:
        torch.save(classifier.state_dict(
        ), '{0}/weights_without_transform/seg_model_{1}_{2}.pth'.format(opt.outf, opt.class_choice, epoch))

# benchmark mIOU
shape_ious = []
for i, data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(2)[1]

    pred_np = pred_choice.cpu().data.numpy()
    target_np = target.cpu().data.numpy() - 1

    for shape_idx in range(target_np.shape[0]):
        parts = range(num_classes)  # np.unique(target_np[shape_idx])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(
                pred_np[shape_idx] == part, target_np[shape_idx] == part))
            U = np.sum(np.logical_or(
                pred_np[shape_idx] == part, target_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))

print("mIOU for class {}: {}".format(opt.class_choice, np.mean(shape_ious)))
