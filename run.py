import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
from torchvision.utils import save_image


from LUSO_loader import DeepSeaDataset
import csv
import numpy as np
import os
import ast
import sys
from PIL import Image
import time

score_dir = r'Score_test'
if not os.path.exists(score_dir):
    os.makedirs(score_dir)

output_dir = r'model_outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total

def iou(pred, target):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious

batch_size = 1
n_class = 2

use_gpu = torch.cuda.is_available()
num_gpu = ast.literal_eval(sys.argv[1])

with open('dataset_test.csv',newline ='') as f:
    reader = csv.reader(f)
    images = list(reader)

test_data = DeepSeaDataset(image_paths = images, phase ='val')
test_loader = DataLoader(test_data, batch_size = 1, num_workers = 8)

torch.cuda.set_device(num_gpu[0])
fcn_model = torch.load(r'fcn_folder')
fcn_model = fcn_model.cuda()
fcn_model = nn.DataParallel(fcn_model, device_ids = num_gpu)


fcn_model.eval()
total_ious = []
pixel_accs = []
times=[]
for iter, batch in enumerate(test_loader):
    if use_gpu:
        inputs = Variable(batch['X'].cuda())
    else:
        inputs = Variable(batch['X'])
        
    start_time = time.time()
    output = fcn_model(inputs)
    elapsed = time.time()-start_time
    times.append(elapsed)
    
    image = output[0]
    output = output.data.cpu().numpy()
    N, _, h, w = output.shape
    pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)
    target = batch['m'].cpu().numpy().reshape(N, h, w)
    for p, t in zip(pred, target):
        total_ious.append(iou(p, t))
        pixel_accs.append(pixel_acc(p,t))
    pred = pred.transpose((1,2,0))
    pred = np.squeeze(pred, axis=2)
    pred = pred.astype(np.bool)
    
    save_image(image, output_dir + r'/' + (images[iter][1].split('/'))[-1])
    #print(pred)
    #print(pred.shape)
    #image = Image.fromarray(pred)
    #image.save(output_dir + r'/' + (images[iter][1].split('/'))[-1])
    
times = np.array(times)
np.save(os.path.join(score_dir, 'times'),times)

total_ious = np.array(total_ious).T
ious = np.nanmean(total_ious, axis=1)
pixel_accs = np.array(pixel_accs).mean()
print("pix_acc: {}, meanIoU: {}, IoUs: {}".format(pixel_accs, np.nanmean(ious), ious))
np.save(os.path.join(score_dir, "meanIU"), ious)
np.save(os.path.join(score_dir, "meanPixel"), pixel_accs)

