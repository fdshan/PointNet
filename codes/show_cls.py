from __future__ import print_function
import argparse
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from dataset import ShapeNetDataset
from model import PointNetCls
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--num_points', type=int,
                    default=2500, help='input batch size')

opt = parser.parse_args()
print(opt)

test_dataset = ShapeNetDataset(
    root=opt.dataset, classification=True, npoints=opt.num_points)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))


classifier = PointNetCls(k=len(test_dataset.classes))
classifier.cuda()
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()

total_correct = 0
total_testset = 0
for i, data in enumerate(test_dataloader, 0):
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
