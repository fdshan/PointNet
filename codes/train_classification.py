from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset import ShapeNetDataset
from model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--feature_transform', default=False, action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


# dataset = ShapeNetDataset(root='shapenetcore_partanno_segmentation_benchmark_v0', classification=True, npoints=opt.num_points)
dataset = ShapeNetDataset(root=opt.dataset, classification=True, npoints=opt.num_points)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

testset = ShapeNetDataset(root=opt.dataset, classification=True, npoints=opt.num_points)
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

print(len(dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)
if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
classifier.cuda()
num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    # scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        # TODO
        # back-propagate loss = cls_loss + 0.001*feature_transform_regularizer()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        # loss = F.cross_entropy(pred, target) + 0.001 * feature_transform_regularizer(trans)
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += 0.001 * feature_transform_regularizer(trans)
        loss.backward()
        optimizer.step()
        pred_label = pred.data.max(1)[1]
        correct = pred_label.eq(target.data).cpu().sum()

        #print('epoch %d,%d training loss: %f accuracy: %f' % (epoch, i, loss.item(), correct.item() / float(opt.batchSize)))
        print('epoch %d,%d training loss: %f' % (epoch, i, loss.item()))
    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))
    print('%s weights saved!' % epoch)

print('Finish!')
'''
total_correct = 0
total_testset = 0
for i, data in tqdm(enumerate(testloader, 0)):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))
'''




