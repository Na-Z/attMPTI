"""Evaluating functions for Few-shot 3D Point Cloud Semantic Segmentation

Author: Zhao Na, 2020
"""
import os
import numpy as np
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from dataloaders.loader import MyTestDataset, batch_test_task_collate
from models.proto_learner import ProtoLearner
from models.mpti_learner import MPTILearner
from utils.cuda_util import cast_cuda
from utils.logger import init_logger


def evaluate_metric(logger, pred_labels_list, gt_labels_list, label2class_list, test_classes):
    """
    :param pred_labels_list: a list of np array, each entry with shape (n_queries*n_way, num_points).
    :param gt_labels_list: a list of np array, each entry with shape (n_queries*n_way, num_points).
    :param test_classes: a list of np array, each entry with shape (n_way,)
    :return: iou: scaler
    """
    assert len(pred_labels_list) == len(gt_labels_list) == len(label2class_list)

    logger.cprint('*****Test Classes: {0}*****'.format(test_classes))

    NUM_CLASS = len(test_classes) + 1 # add 1 to consider background class
    gt_classes = [0 for _ in range(NUM_CLASS)]
    positive_classes = [0 for _ in range(NUM_CLASS)]
    true_positive_classes = [0 for _ in range(NUM_CLASS)]

    for i, batch_gt_labels in enumerate(gt_labels_list):
        batch_pred_labels = pred_labels_list[i] #(n_queries*n_way, num_points)
        label2class = label2class_list[i] #(n_way,)

        for j in range(batch_pred_labels.shape[0]):
            for k in range(batch_pred_labels.shape[1]):
                gt = int(batch_gt_labels[j, k])
                pred = int(batch_pred_labels[j,k])

                if gt == 0: # 0 indicate background class
                    gt_index = 0
                else:
                    gt_class = label2class[gt-1] # the ground truth class in the dataset
                    gt_index = test_classes.index(gt_class) + 1
                gt_classes[gt_index] += 1

                if pred == 0:
                    pred_index = 0
                else:
                    pred_class = label2class[pred-1]
                    pred_index = test_classes.index(pred_class) + 1
                positive_classes[pred_index] += 1

                true_positive_classes[gt_index] += int(gt == pred)

    iou_list = []
    for c in range(NUM_CLASS):
        iou = true_positive_classes[c] / float(gt_classes[c] + positive_classes[c] - true_positive_classes[c])
        logger.cprint('----- [class %d]  IoU: %f -----'% (c, iou))
        iou_list.append(iou)

    mean_IoU = np.array(iou_list[1:]).mean()

    return mean_IoU


def test_few_shot(test_loader, learner, logger, test_classes):

    total_loss = 0

    predicted_label_total = []
    gt_label_total = []
    label2class_total = []

    for batch_idx, (data, sampled_classes) in enumerate(test_loader):
        query_label = data[-1]

        if torch.cuda.is_available():
            data = cast_cuda(data)

        query_pred, loss, accuracy = learner.test(data)
        total_loss += loss.detach().item()

        if (batch_idx+1) % 50 == 0:
            logger.cprint('[Eval] Iter: %d | Loss: %.4f | %s' % ( batch_idx+1, loss.detach().item(), str(datetime.now())))

        #compute metric for predictions
        predicted_label_total.append(query_pred.cpu().detach().numpy())
        gt_label_total.append(query_label.numpy())
        label2class_total.append(sampled_classes)

    mean_loss = total_loss/len(test_loader)
    mean_IoU = evaluate_metric(logger, predicted_label_total, gt_label_total, label2class_total, test_classes)
    return mean_loss, mean_IoU


def eval(args):
    logger = init_logger(args.log_dir, args)

    if args.phase == 'protoeval':
        learner = ProtoLearner(args, mode='test')
    elif args.phase == 'mptieval':
        learner = MPTILearner(args, mode='test')

    #Init dataset, dataloader
    TEST_DATASET = MyTestDataset(args.data_path, args.dataset, cvfold=args.cvfold,
                                 num_episode_per_comb=args.n_episode_test,
                                 n_way=args.n_way, k_shot=args.k_shot, n_queries=args.n_queries,
                                 num_point=args.pc_npts, pc_attribs=args.pc_attribs,  mode='test')
    TEST_CLASSES = list(TEST_DATASET.classes)
    TEST_LOADER = DataLoader(TEST_DATASET, batch_size=1, shuffle=False, collate_fn=batch_test_task_collate)

    test_loss, mean_IoU = test_few_shot(TEST_LOADER, learner, logger, TEST_CLASSES)

    logger.cprint('\n=====[TEST] Loss: %.4f | Mean IoU: %f =====\n' %(test_loss, mean_IoU))
