# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/1/23
# License: MIT License
import math

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from skorch.classifier import NeuralNetClassifier
from skorch.dataset import ValidSplit
from skorch.callbacks import LRScheduler, EpochScoring, Checkpoint, EarlyStopping


def compute_out_size(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
):
    return int(
        (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )


def compute_same_pad1d(input_size, kernel_size, stride=1, dilation=1):
    all_padding = (stride - 1) * input_size - stride + dilation * (kernel_size - 1) + 1
    return (all_padding // 2, all_padding - all_padding // 2)


def compute_same_pad2d(input_size, kernel_size, stride=(1, 1), dilation=(1, 1)):
    ud = compute_same_pad1d(
        input_size[0], kernel_size[0], stride=stride[0], dilation=dilation[0]
    )
    lr = compute_same_pad1d(
        input_size[1], kernel_size[1], stride=stride[1], dilation=dilation[1]
    )
    return [*lr, *ud]


class MaxNormConstraintConv2d(nn.Conv2d):
    def __init__(self, *args, max_norm_value=1, norm_axis=2, **kwargs):
        self.max_norm_value = max_norm_value
        self.norm_axis = norm_axis
        super().__init__(*args, **kwargs)

    def forward(self, input):
        self.weight.data = self._max_norm(self.weight.data)
        return super().forward(input)

    def _max_norm(self, w):
        with torch.no_grad():
            # similar behavior as keras MaxNorm constraint
            norms = torch.sqrt(
                torch.sum(torch.square(w), dim=self.norm_axis, keepdim=True)
            )
            desired = torch.clamp(norms, 0, self.max_norm_value)
            # instead of desired/(eps+norm), without changing norm in range
            w *= desired / norms
        return w


class MaxNormConstraintLinear(nn.Linear):
    def __init__(self, *args, max_norm_value=1, norm_axis=0, **kwargs):
        self.max_norm_value = max_norm_value
        self.norm_axis = norm_axis
        super().__init__(*args, **kwargs)

    def forward(self, input):
        self.weight.data = self._max_norm(self.weight.data)
        return super().forward(input)

    def _max_norm(self, w):
        with torch.no_grad():
            # similar behavior as keras MaxNorm constraint
            norms = torch.sqrt(
                torch.sum(torch.square(w), dim=self.norm_axis, keepdim=True)
            )
            desired = torch.clamp(norms, 0, self.max_norm_value)
            # instead of desired/(eps+norm), without changing norm in range
            w *= desired / norms
        return w


def _adabn_pre_forward_hook(self, inputs):
    old_training_state = self.training
    self.eval()
    # global AdaBN
    with torch.no_grad():
        if not hasattr(self, "num_samples_tracked"):
            self.num_samples_tracked = 0
            self.running_mean.data.zero_()
            self.running_var.data.zero_()
            self.running_var.data.fill_(1)
        k = len(inputs[0])
        self.num_samples_tracked += k

        module_name = self.__class__.__name__
        if "BatchNorm1d" in module_name:
            mean = torch.mean(inputs[0], dim=(0, 2))
            var = torch.var(input[0], dim=(0, 2))
        elif "BatchNorm2d" in module_name:
            mean = torch.mean(inputs[0], dim=(0, 2, 3))
            var = torch.var(inputs[0], dim=(0, 2, 3))
        elif "BatchNorm3d" in module_name:
            mean = torch.mean(inputs[0], dim=(0, 2, 3, 4))
            var = torch.var(inputs[0], dim=(0, 2, 3, 4))

        # see https://www.sciencedirect.com/science/article/abs/pii/S003132031830092X
        d = mean - self.running_mean.data
        self.running_mean.data.add_(d * k / self.num_samples_tracked)
        self.running_var.data.add_(
            (var - self.running_var.data) * k / self.num_samples_tracked
            + torch.square(d)
            * k
            * (self.num_samples_tracked - k)
            / (self.num_samples_tracked**2)
        )

    if old_training_state:
        self.train()


def _global_adabn_pre_forward_hook(self, inputs):
    old_training_state = self.training
    self.eval()
    # global AdaBN
    with torch.no_grad():
        module_name = self.__class__.__name__
        if "BatchNorm1d" in module_name:
            mean = torch.mean(inputs[0], dim=(0, 2))
            var = torch.var(inputs[0], dim=(0, 2))
        elif "BatchNorm2d" in module_name:
            mean = torch.mean(inputs[0], dim=(0, 2, 3))
            var = torch.var(inputs[0], dim=(0, 2, 3))
        elif "BatchNorm3d" in module_name:
            mean = torch.mean(inputs[0], dim=(0, 2, 3, 4))
            var = torch.var(inputs[0], dim=(0, 2, 3, 4))

        self.running_mean.data.zero_()
        self.running_mean.data.add_(mean)
        self.running_var.data.zero_()
        self.running_var.data.add_(var)
    if old_training_state:
        self.train()


def adaptive_batch_norm(model, use_global=False):
    # register pre_forward_hook
    handles = []
    hook = _global_adabn_pre_forward_hook if use_global else _adabn_pre_forward_hook
    for module in model.modules():
        if "BatchNorm" in module.__class__.__name__:
            handles.append(module.register_forward_pre_hook(hook))
    return model, handles


def _glorot_weight_zero_bias(model):
    """Initalize parameters of all modules by initializing weights with
    glorot uniform/xavier initialization, and setting biases to zero.
    Weights from batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
    """
    for module in model.modules():
        if hasattr(module, "weight"):
            if not ("BatchNorm" in module.__class__.__name__):
                nn.init.xavier_uniform_(module.weight, gain=1)
            else:
                nn.init.constant_(module.weight, 1)
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


def _narrow_normal_weight_zero_bias(model):
    """Initalize parameters of all modules by initializing weights with
    narrow normal distribution N(0, 0.01).
    Weights from batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
    """
    for module in model.modules():
        if hasattr(module, "weight"):
            if not ("BatchNorm" in module.__class__.__name__):
                nn.init.normal_(module.weight, mean=0.0, std=1e-2)
            else:
                nn.init.constant_(module.weight, 1)
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class NeuralNetClassifierNoLog(NeuralNetClassifier):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        return super(NeuralNetClassifier, self).get_loss(
            y_pred, y_true, *args, **kwargs
        )

    def fit(self, X, y, **fit_params):
        net = super(NeuralNetClassifier, self).fit(X, y, **fit_params)
        callbacks = OrderedDict(net.callbacks)
        if "checkpoint" in callbacks:
            net.load_params(checkpoint=callbacks["checkpoint"])
        return net


class SkorchNet:
    def __init__(self, module):
        self.module = module

    def __call__(self, *args, **kwargs):
        model = self.module(*args, **kwargs)
        net = NeuralNetClassifierNoLog(
            model.double(),
            criterion=nn.CrossEntropyLoss,
            optimizer=optim.Adam,
            optimizer__weight_decay=0,
            batch_size=128,
            lr=1e-2,
            max_epochs=300,
            device="cpu",
            train_split=ValidSplit(0.2, stratified=True),
            iterator_train__shuffle=True,
            callbacks=[
                (
                    "train_acc",
                    EpochScoring(
                        "accuracy",
                        name="train_acc",
                        on_train=True,
                        lower_is_better=False,
                    ),
                ),
                ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=300 - 1)),
                ("estoper", EarlyStopping(patience=50)),
                (
                    "checkpoint",
                    Checkpoint(dirname="checkpoints/{:s}".format(str(id(model)))),
                ),
            ],
            verbose=True,
        )
        return net


def np_to_th(X, requires_grad=False, dtype=None, pin_memory=False, **tensor_kwargs):
    """
    Convenience function to transform numpy array to `torch.Tensor`.

    Converts `X` to ndarray using asarray if necessary.

    Parameters
    ----------
    X: ndarray or list or number
        Input arrays
    requires_grad: bool
        passed on to Variable constructor
    dtype: numpy dtype, optional
    var_kwargs:
        passed on to Variable constructor

    Returns
    -------
    var: `torch.Tensor`
    """
    if not hasattr(X, "__len__"):
        X = [X]
    X = np.asarray(X)
    if dtype is not None:
        X = X.astype(dtype)
    X_tensor = torch.tensor(X, requires_grad=requires_grad, **tensor_kwargs)
    if pin_memory:
        X_tensor = X_tensor.pin_memory()
    return X_tensor


class AvgPool2dWithConv(nn.Module):
    """
    Compute average pooling using a convolution, to have the dilation parameter.

    Parameters
    ----------
    kernel_size: (int,int)
        Size of the pooling region.
    stride: (int,int)
        Stride of the pooling operation.
    dilation: int or (int,int)
        Dilation applied to the pooling filter.
    padding: int or (int,int)
        Padding applied before the pooling operation.
    """

    def __init__(self, kernel_size, stride, dilation=1, padding=0):
        super(AvgPool2dWithConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        # don't name them "weights" to
        # make sure these are not accidentally used by some procedure
        # that initializes parameters or something
        self._pool_weights = None

    def forward(self, x):
        # Create weights for the convolution on demand:
        # size or type of x changed...
        in_channels = x.size()[1]
        weight_shape = (
            in_channels,
            1,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        if self._pool_weights is None or (
            (tuple(self._pool_weights.size()) != tuple(weight_shape))
            or (self._pool_weights.is_cuda != x.is_cuda)
            or (self._pool_weights.data.type() != x.data.type())
        ):
            n_pool = np.prod(self.kernel_size)
            weights = np_to_th(np.ones(weight_shape, dtype=np.float32) / float(n_pool))
            weights = weights.type_as(x)
            if x.is_cuda:
                weights = weights.cuda()
            self._pool_weights = weights

        pooled = F.conv2d(
            x,
            self._pool_weights,
            bias=None,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            groups=in_channels,
        )
        return pooled


class Expression(nn.Module):
    """Compute given expression on forward pass.

    Parameters
    ----------
    expression_fn : callable
        Should accept variable number of objects of type
        `torch.autograd.Variable` to compute its output.
    """

    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if hasattr(self.expression_fn, "func") and hasattr(
            self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
            )
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return self.__class__.__name__ + "(expression=%s) " % expression_str


class Ensure4d(nn.Module):
    def forward(self, x):
        while len(x.shape) < 4:
            x = x.unsqueeze(-1)
        return x


def identity(x):
    return x


def transpose_time_to_spat(x):
    """Swap time and spatial dimensions.

    Returns
    -------
    x: torch.Tensor
        tensor in which last and first dimensions are swapped
    """
    return x.permute(0, 3, 2, 1)


def squeeze_final_output(x):
    """Removes empty dimension at end and potentially removes empty time
     dimension. It does  not just use squeeze as we never want to remove
     first dimension.

    Returns
    -------
    x: torch.Tensor
        squeezed tensor
    """

    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


#************************************************************#
#  稀疏编码器（Sparse Encoder）
#  本模块通过 shared_mask 控制参数的稀疏性，其中一部分权重由外部共享权重代替，
#************************************************************#

class SparseEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, shared_ratio=0.3, alpha=1.0, epsilon=20):
        """
        稀疏编码器（SparseEncoder）
        ----------------------------------------
        Args:
            input_dim    : 输入特征维度
            output_dim   : 输出特征维度
            shared_ratio : 权重共享比例（取值 0~1，越大表示共享越多）
            alpha     : 共享权重的缩放倍数
            epsilon      : Erdős-Rényi 随机稀疏度控制参数（越大越稀疏）
        ----------------------------------------
        用途：
            - 支持部分权重与外部共享
            - 通过 mask_scores 选择共享/独立权重
            - mask_scores 可学习，初值基于 Erdős-Rényi 随机稀疏图
        """
        super(SparseEncoder, self).__init__()
        # 保存基本超参数
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.shared_ratio = shared_ratio
        self.alpha = alpha
        #self.beta = beta
        self.epsilon = epsilon
        # ----------------------------
        # 1. 定义可训练权重与偏置
        # ----------------------------
        # self.weight: 当前层的本地权重 [output_dim, input_dim]
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        # self.bias: 偏置向量 [output_dim]
        self.bias = nn.Parameter(torch.Tensor(output_dim))

        # ----------------------------
        # 2. Kaiming 初始化权重和偏置
        # ----------------------------
        # Kaiming 初始化适用于 ReLU / LeakyReLU 激活，保证方差稳定
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        #nn.init.xavier_uniform_(self.weight, gain=1)
        
        # 根据 Kaiming 初始化计算偏置范围
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

        # ----------------------------
        # 3. 初始化 mask_scores
        # ----------------------------
        # mask_scores 是可学习的参数，控制哪些连接被选择为共享
        # 初值使用 Erdős-Rényi 随机稀疏图
        init_mask_scores,self.prob = self._create_erdos_renyi_mask_scores()
        self.mask_scores = nn.Parameter(init_mask_scores)

    def _create_erdos_renyi_mask_scores(self):
        """
        基于 Erdős-Rényi 随机图生成初始稀疏 mask_scores
        -------------------------------------------------
        公式：
            p = epsilon * (in + out) / (in * out)
            - epsilon 越大 -> 稀疏程度越高
            - p 表示“保留连接”的概率
        -------------------------------------------------
        返回:
            mask_scores: [output_dim, input_dim] 的 0/1 张量
                  - 1 表示连接保留
                  - 0 表示连接剪枝
        """
        device = self.weight.device  # 确保 mask_scores 与权重在同一设备上
        # 根据公式计算连接保留概率
        prob = (self.epsilon * (self.input_dim + self.output_dim)) / (self.input_dim * self.output_dim)
        # 生成随机 0/1 mask_scores：随机数 >= prob 表示保留该连接
        mask_scores = (torch.rand(self.output_dim, self.input_dim, device=device) < prob).float()
        return mask_scores,prob

    def get_mask_scores(self):
        """
        根据 mask_scores 选择共享连接
        -------------------------------------------------
        作用：
            - 从 mask_scores 中选出 top-k 最大的连接
            - 生成两个 mask_scores:
                shared_mask      : 表示使用共享权重的连接
                independent_mask : 表示使用本地权重的连接
        返回:
            shared_mask: [output_dim, input_dim] 布尔张量
            independent_mask: [output_dim, input_dim] 布尔张量
        """
        num_elements = self.mask_scores.numel()  # 总连接数 = output_dim * input_dim
        # 情况 1: 完全不共享
        if self.shared_ratio <= 0:
            shared_mask = torch.zeros_like(self.mask_scores, dtype=torch.bool)
        # 情况 2: 全部共享
        elif self.shared_ratio >= 1:
            shared_mask = torch.ones_like(self.mask_scores, dtype=torch.bool)
        # 情况 3: 部分共享 -> 取前 k 大 mask_scores
        else:
            # 计算需要共享的连接数量 k
            k = int(self.shared_ratio * num_elements)
            # 取 mask_scores 最大的 k 个索引
            _, indices = torch.topk(self.mask_scores.view(-1), k)
            # 初始化全 False 的 mask_scores
            shared_mask = torch.zeros(num_elements, dtype=torch.bool, device=self.mask_scores.device)
            # 将 top-k 的位置设为 True
            shared_mask[indices] = True
            # 恢复为原始二维形状
            shared_mask = shared_mask.view_as(self.mask_scores)
        # 非共享的连接使用本地权重
        independent_mask = ~shared_mask
        return shared_mask, independent_mask

    def forward(self, x, shared_weight):
        """
        前向传播
        -------------------------------------------------
        计算步骤：
            1. 根据 mask_scores 选择共享权重和独立权重
            2. 生成组合权重矩阵 combined_weight
            3. 对输入 x 执行线性变换
        Args:
            x            : 输入特征，形状 [batch_size, input_dim]
            shared_weight: 外部提供的共享权重，形状 [output_dim, input_dim]
        Returns:
            output: 形状 [batch_size, output_dim]
        """
        # 获取共享连接的布尔 mask_scores
        shared_mask, _ = self.get_mask_scores()
        # 确保 mask 和共享权重在同一设备
        shared_mask = shared_mask.to(shared_weight.device)
        print(f"shared_mask{shared_mask.shape}")
        print(f"shared_weight{shared_weight.shape}")
        print(f"self.weight{self.weight.shape}")
        # 根据 mask 选择使用共享权重或本地权重
        combined_weight = torch.where(
            shared_mask,                    # 如果共享
            shared_weight * self.alpha, # 使用共享权重并乘以缩放因子
            self.weight          # 否则使用本地权重
        )
        if self.training:
            dropout_mask = (torch.rand_like(combined_weight) > (1-self.prob)).float()  
            combined_weight = combined_weight * dropout_mask / self.prob
        # 线性变换 y = xW^T + b
        return F.linear(x, combined_weight, self.bias)