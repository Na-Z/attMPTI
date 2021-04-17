""" Finetune Baseline for Few-shot 3D Point Cloud Semantic Segmentation

Author: Zhao Na, 2020
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from runs.eval import evaluate_metric
from runs.pre_train import DGCNNSeg
from models.dgcnn import DGCNN
from dataloaders.loader import MyTestDataset, batch_test_task_collate, augment_pointcloud
from utils.logger import init_logger
from utils.cuda_util import cast_cuda
from utils.checkpoint_util import load_pretrain_checkpoint


class FineTuner(object):
    def __init__(self, args):

        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.n_queries = args.n_queries
        self.n_points = args.pc_npts

        # init model and optimizer
        self.model = DGCNNSeg(args, self.n_way+1)
        print(self.model)
        if torch.cuda.is_available():
            self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.segmenter.parameters(), lr=args.lr)

        # load pretrained model for point cloud encoding
        self.model = load_pretrain_checkpoint(self.model, args.pretrain_checkpoint_path)


    def train(self, support_x, support_y):
        """
        Args:
            support_x: support point clouds with shape (n_way*k_shot, in_channels, num_points)
            support_y: support masks (foreground) with shape (n_way*k_shot, num_points), each point \in {0,..., n_way}
        """
        support_logits = self.model(support_x)

        train_loss = F.cross_entropy(support_logits, support_y)

        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()

        return train_loss

    def test(self, query_x, query_y):
        """
        Args:
            query_x: query point clouds with shape (n_queries, in_channels, num_points)
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way}
        """

        self.model.eval()

        with torch.no_grad():
            query_logits = self.model(query_x)
            test_loss = F.cross_entropy(query_logits, query_y)

            pred = F.softmax(query_logits, dim=1).argmax(dim=1)
            correct = torch.eq(pred, query_y).sum().item()
            accuracy = correct / (self.n_queries*self.n_points)

        return pred, test_loss, accuracy


def support_mask_to_label(support_masks, n_way, k_shot, num_points):
    """
    Args:
        support_masks: binary (foreground/background) masks with shape (n_way, k_shot, num_points)
    """
    support_masks = support_masks.view(n_way, k_shot*num_points)
    support_labels = []
    for n in range(support_masks.shape[0]):
        support_mask = support_masks[n, :] #(k_shot*num_points)
        support_label = torch.zeros_like(support_mask)
        mask_index = torch.nonzero(support_mask).squeeze(1)
        support_label= support_label.scatter_(0, mask_index, n+1)
        support_labels.append(support_label)

    support_labels = torch.stack(support_labels, dim=0)
    support_labels = support_labels.view(n_way, k_shot, num_points)

    return support_labels.long()


def finetune(args):
    num_iters = args.n_iters

    logger = init_logger(args.log_dir, args)

    #Init datasets, dataloaders, and writer
    DATASET = MyTestDataset(args.data_path, args.dataset, cvfold=args.cvfold,
                                 num_episode_per_comb=args.n_episode_test,
                                 n_way=args.n_way, k_shot=args.k_shot, n_queries=args.n_queries,
                                 num_point=args.pc_npts, pc_attribs=args.pc_attribs, mode='test')
    CLASSES = list(DATASET.classes)
    DATA_LOADER = DataLoader(DATASET, batch_size=1, collate_fn=batch_test_task_collate)
    WRITER = SummaryWriter(log_dir=args.log_dir)

    #Init model and optimizer
    FT = FineTuner(args)

    predicted_label_total = []
    gt_label_total = []
    label2class_total = []

    global_iter = 0
    for batch_idx, (data, sampled_classes) in enumerate(DATA_LOADER):
        query_label = data[-1]
        data[1] = support_mask_to_label(data[1], args.n_way, args.k_shot, args.pc_npts)

        if torch.cuda.is_available():
            data = cast_cuda(data)

        [support_x, support_y, query_x, query_y] = data
        support_x = support_x.view(args.n_way * args.k_shot, -1, args.pc_npts)
        support_y = support_y.view(args.n_way * args.k_shot, args.pc_npts)

        # train on support set
        for i in range(num_iters):
            train_loss = FT.train(support_x, support_y)

            WRITER.add_scalar('Train/loss', train_loss, global_iter)
            logger.cprint('=====[Train] Batch_idx: %d | Iter: %d | Loss: %.4f =====' % (batch_idx, i, train_loss.item()))

            global_iter += 1

        # test on query set
        query_pred, test_loss, accuracy = FT.test(query_x, query_y)
        WRITER.add_scalar('Test/loss', test_loss, global_iter)
        WRITER.add_scalar('Test/accuracy', accuracy, global_iter)
        logger.cprint(
            '=====[Valid] Batch_idx: %d | Loss: %.4f =====' % (batch_idx, test_loss.item()))

        #compute metric for predictions
        predicted_label_total.append(query_pred.cpu().detach().numpy())
        gt_label_total.append(query_label.numpy())
        label2class_total.append(sampled_classes)

    mean_IoU = evaluate_metric(logger, predicted_label_total, gt_label_total, label2class_total, CLASSES)
    logger.cprint('\n=====[Test] Mean IoU: %f =====\n' % mean_IoU)
