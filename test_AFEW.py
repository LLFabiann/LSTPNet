import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import torch
import os
import argparse

from dataloader.dataset_AFEW import train_data_loader, test_data_loader
from models.ST_Former import GenerateModel

import random
from utils import *
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DFEW', help='dataset (AFEW, DFEW, FERV39k)')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N')
    parser.add_argument('--gpu', type=str, default='0', help='assign multi-gpus by comma concat')  
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('-c', '--checkpoint', type=str, default=r'D:\DL_Study\Expression_Project\Video_FER\专用可视化权重\AFEW\best_val_acc_uar__epoch67_acc0.53543_uar0.49922_val_acc_uar1.0346500141334534.pth', help='Pytorch checkpoint file path')
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def test():
    args = parse_args()

    cudnn.deterministic = False
    cudnn.benchmark = True    

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])

    see = 3407
    set_seed(see) 
    print("Seed: ", see) 

    num_classes = args.num_classes
    model = GenerateModel(num_classes)
    model = torch.nn.DataParallel(model)

    print("Loading pretrained weights...", args.checkpoint)
    checkpoint = torch.load(args.checkpoint)
    checkpoint = checkpoint["model_state_dict"]
    model = load_pretrained_weights(model, checkpoint)
    model = model.cuda()
    
    test_dataset = test_data_loader()
    test_size = test_dataset.__len__()
    print('Test set size:', test_size)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=False
                                             )

    pre_labels = []
    gt_labels = []
    with torch.no_grad():
        bingo_cnt = 0
        model.eval()
        for _, (imgs, targets) in tqdm(enumerate(tqdm(test_loader))):
            
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs, _, _ = model(imgs)
            _, predicts = torch.max(outputs, 1)
            correct_or_not = torch.eq(predicts, targets)
            bingo_cnt += correct_or_not.sum().cpu()
            pre_labels += predicts.cpu().tolist()
            gt_labels += targets.cpu().tolist()

    acc = bingo_cnt.float() / float(test_size)
    acc = np.around(acc.numpy(), 4)
    print(f"Test accuracy: {acc:.4f}.")
    # UAR
    cm = confusion_matrix(gt_labels, pre_labels)
    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    uar = np.mean(np.diagonal(cm))
    uar = np.around(uar, 5)
    print(f"Test UAR: {uar:.4f}.")

if __name__ == "__main__":                    
    test()

