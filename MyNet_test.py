import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from models.ESNet import ESNet
from data import test_dataset
import time
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='./test/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

#load the model
model = ESNet()
model.load_state_dict(torch.load('./cpts/best.pth'))
print("load")
model.cuda()
model.eval()
# test
test_datasets = ['SIP','NJU2K','NLPR','STERE','LFSD','DUT-RGBD']

costtim = []

for dataset in test_datasets:
    save_path = './pre_map/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root = dataset_path + dataset + '/depth/'
    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    cost_time = []
    for i in range(test_loader.size):
        image, gt, depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.repeat(1,3,1,1).cuda()
        start_time = time.perf_counter()
        res,s,s1,s2= model(image,depth)
        cost_time.append(time.perf_counter()-start_time)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path + name, res*255)
    cost_time.pop(0)
    print('Test Done!',save_path)
    print('Mean running time is :',np.mean(cost_time))
    print("FPS is :",test_loader.size/np.sum(cost_time))
    costtim.append(test_loader.size/np.sum(cost_time))
print(costtim)
