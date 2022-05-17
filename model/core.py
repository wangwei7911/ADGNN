import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter ,LayerNorm ,InstanceNorm2d
from model.utils import Spatial_Temporal_Corr, ST_BLOCK_1
import torch

torch.set_default_dtype(torch.float32)


class DAGCN(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, recent, K, Kt, batch_size):
        super(DAGCN, self).__init__()
        self.tem_size = recent
        self.c_out = c_out
        self.n_nodes = num_nodes
        self.batch_size = batch_size
        self.STC_A = Spatial_Temporal_Corr(c_in, c_out, self.n_nodes, self.tem_size, K, Kt, self.batch_size)
        self.STC_B = Spatial_Temporal_Corr(c_in, c_out, self.n_nodes, self.tem_size, K, Kt, self.batch_size)
        self.STC_C = Spatial_Temporal_Corr(c_out, c_out, self.n_nodes, self.tem_size, K, Kt, self.batch_size)
        # self.STC_D = Spatial_Temporal_Corr(c_out, c_out, self.n_nodes, self.tem_size, K, Kt, self.batch_size)
        # self.STC_E = Spatial_Temporal_Corr(c_out, c_out, self.n_nodes, self.tem_size, K, Kt, self.batch_size)
        # self.STC_F = Spatial_Temporal_Corr(c_out, c_out, self.n_nodes, self.tem_size, K, Kt, self.batch_size)
        # self.STC_G = Spatial_Temporal_Corr(c_out, c_out, self.n_nodes, self.tem_size, K, Kt, self.batch_size)

        self.bn = BatchNorm2d(c_in, affine=False)

        self.conv1 = Conv2d(c_out, 1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.conv2 = Conv2d(1, 1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.conv3 = Conv2d(1, 1, kernel_size=(1, 2), padding=(0, 0), stride=(1, 2), bias=True)
        # self.conv3 = Conv2d(c_out, 1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        # self.conv4 = Conv2d(c_out, 1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

        self.h = Parameter(torch.zeros(num_nodes, dtype=torch.float32), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0000001)

        # self.h1 = Parameter(torch.empty((1,)), requires_grad=True)
        # self.h2 = Parameter(torch.empty((1,)), requires_grad=True)
        # nn.init.uniform_(self.h1)
        # nn.init.uniform_(self.h2)

    def forward(self, x_r, supports):

        x = self.bn(x_r)

        D_h = torch.diag_embed(self.h)
        A1 = supports + D_h


        # x_A, _, _ = self.STC_A(x, A1)
        # x_B, _, _ = self.STC_B(x, A1)
        # x_C = x_A * x_B

        x, _, _ = self.STC_A(x, A1)


        # x1 = x[:, :, :, 0:12]
        x1 = x[:, :, :, 0:24]


        # print(x1.shape)

        # x1 = self.bn(x)
        x_out = self.conv1(x1)
        x_out = self.conv2(x_out)
        # x_out = F.leaky_relu(x_out)
        x_out = self.conv3(x_out).squeeze()


        return x_out, _, _


""" Original Edition for Adaptive Dynamic GCN """
class ActivateGCN(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, recent, K, Kt):
        super(ActivateGCN, self).__init__()
        # tem_size = week + day + recent
        tem_size = recent
        self.nodes = num_nodes
        self.block1 = ST_BLOCK_1(c_in, c_out, num_nodes, tem_size, K, Kt)
        self.block2 = ST_BLOCK_1(c_out, c_out, num_nodes, tem_size, K, Kt)
        self.bn = BatchNorm2d(c_in, affine=False)
        self.conv1 = Conv2d(c_out, 1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.conv2 = Conv2d(c_out, 1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.conv3 = Conv2d(c_out, 1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.conv4 = Conv2d(c_out, 1, kernel_size=(1, 2), padding=(0, 0), stride=(1, 2), bias=True)

        self.h = Parameter(torch.zeros(num_nodes, dtype=torch.float32), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=1e-14)

    def forward(self, x_r, supports):
        x_r = torch.unsqueeze(x_r[:, 0, :, :], 1)

        x_r = self.bn(x_r)
        x = torch.cat((x_r,), -1)

        D_h = torch.diag_embed(self.h)

        A = supports + D_h



        A1 = F.dropout(A, 0.5, self.training)

        x, _, _ = self.block1(x, A1)
        # A1 = torch.nn.AdaptiveAvgPool2d((self.nodes, self.nodes), )(A1.unsqueeze(0).unsqueeze(0))[0, 0]
        x, d_adj, t_adj = self.block2(x, A1)

        x1 = x[:, :, :, 0:12]
        x2 = x[:, :, :, 12:24]

        x1 = self.conv1(x1).squeeze()
        x2 = self.conv2(x2).squeeze()

        x = x1 + x2
        return x, d_adj, A1


# class ActivateGCN(nn.Module):
#     def __init__(self, c_in, c_out, num_nodes, recent, K, Kt):
#         super(ActivateGCN, self).__init__()
#         # tem_size = week + day + recent
#         tem_size = recent
#         self.nodes = num_nodes
#         self.block1 = ST_BLOCK_1(c_in, c_out, num_nodes, tem_size, K, Kt)
#         self.block2 = ST_BLOCK_1(c_out, c_out, num_nodes, tem_size, K, Kt)
#         self.bn = BatchNorm2d(c_in, affine=False)
#         self.conv1 = Conv2d(c_out, 1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
#         self.conv2 = Conv2d(c_out, 1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
#         self.conv3 = Conv2d(c_out, 1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
#         self.conv4 = Conv2d(c_out, 1, kernel_size=(1, 2), padding=(0, 0), stride=(1, 2), bias=True)
#
#         self.h = Parameter(torch.zeros(1, num_nodes, dtype=torch.float32), requires_grad=True)
#         nn.init.uniform_(self.h, a=0, b=1e-6)
#
#     def forward(self, x_r, supports):
#         x_r = torch.unsqueeze(x_r[:, 0, :, :], 1)
#         x_r = self.bn(x_r)
#         x = torch.cat((x_r,), -1)
#
#
#         """ Adjacency Matrix Here"""
#         print(x_r.shape)                    # b,c,n,l
#         A_1 = torch.matmul(supports, self.h)
#         A_1 = torch.matmul(A_1, )
#
#
#
#
#         A1 = F.dropout(supports, 0.5, self.training)
#
#         x, _, _         = self.block1(x, A1)
#         x, d_adj, t_adj = self.block2(x, A1)
#
#         x1 = x[:, :, :, 0:12]
#         x2 = x[:, :, :, 12:24]
#
#         x1 = self.conv1(x1).squeeze()
#         x2 = self.conv2(x2).squeeze()
#
#         x = x1 + x2
#         return x, d_adj, A1