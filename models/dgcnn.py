"""DGCNN as Backbone to extract point-level features
   Adapted from https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
   Author: Zhao Na, 2020
"""

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x) #(B,N,N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True) #(B,1,N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1) #(B,N,N)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B,N,k)
    return idx


def get_edge_feature(x, K=20, idx=None):
    """Construct edge feature for each point
      Args:
        x: point clouds (B, C, N)
        K: int
        idx: knn index, if not None, the shape is (B, N, K)
      Returns:
        edge feat: (B, 2C, N, K)
    """
    B, C, N = x.size()
    if idx is None:
        idx = knn(x, k=K)  # (batch_size, num_points, k)
    central_feat = x.unsqueeze(-1).expand(-1,-1,-1,K)
    idx = idx.unsqueeze(1).expand(-1, C, -1, -1).contiguous().view(B,C,N*K)
    knn_feat = torch.gather(x, dim=2, index=idx).contiguous().view(B,C,N,K)
    edge_feat = torch.cat((knn_feat-central_feat, central_feat), dim=1)
    return edge_feat


class conv2d(nn.Module):
    def __init__(self, in_feat, layer_dims, batch_norm=True, relu=True, bias=False):
        super().__init__()
        self.layer_dims = layer_dims
        layers = []
        for i in range(len(layer_dims)):
            in_dim = in_feat if i==0 else layer_dims[i-1]
            out_dim = layer_dims[i]
            layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=bias))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_dim))
            if relu:
                layers.append(nn.LeakyReLU(0.2))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class conv1d(nn.Module):
    def __init__(self, in_feat, layer_dims, batch_norm=True, relu=True, bias=False):
        super().__init__()
        self.layer_dims = layer_dims
        layers = []
        for i in range(len(layer_dims)):
            in_dim = in_feat if i==0 else layer_dims[i-1]
            out_dim = layer_dims[i]
            layers.append(nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=bias))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            if relu:
                layers.append(nn.LeakyReLU(0.2))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class DGCNN(nn.Module):
    """
    DGCNN with only stacked EdgeConv, return intermediate features if use attention
    Parameters:
      edgeconv_widths: list of layer widths of edgeconv blocks [[],[],...]
      mlp_widths: list of layer widths of mlps following Edgeconv blocks
      nfeat: number of input features
      k: number of neighbors
      conv_aggr: neighbor information aggregation method, Option:['add', 'mean', 'max', None]
    """
    def __init__(self, edgeconv_widths, mlp_widths, nfeat, k=20, return_edgeconvs=False):
        super(DGCNN, self).__init__()
        self.n_edgeconv = len(edgeconv_widths)
        self.k = k
        self.return_edgeconvs = return_edgeconvs

        self.edge_convs = nn.ModuleList()
        for i in range(self.n_edgeconv):
            if i==0:
                in_feat = nfeat*2
            else:
                in_feat = edgeconv_widths[i-1][-1]*2

            self.edge_convs.append(conv2d(in_feat, edgeconv_widths[i]))

        in_dim = 0
        for edgeconv_width in edgeconv_widths:
            in_dim += edgeconv_width[-1]
        self.conv = conv1d(in_dim, mlp_widths)

    def forward(self, x):
        edgeconv_outputs = []
        for i in range(self.n_edgeconv):
            x = get_edge_feature(x, K=self.k)
            x = self.edge_convs[i](x)
            x = x.max(dim=-1, keepdim=False)[0]
            edgeconv_outputs.append(x)

        out = torch.cat(edgeconv_outputs, dim=1)
        out = self.conv(out)

        if self.return_edgeconvs:
            return edgeconv_outputs, out
        else:
            return edgeconv_outputs[0], out
