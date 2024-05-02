# Copyright (c) MONA Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import warnings

import torch
from monai.networks.layers.utils import get_act_layer
from monai.utils import LossReduction
from monai.utils.enums import StrEnum
from torch.nn.modules.loss import _Loss


class AdversarialCriterions(StrEnum):
    BCE = "bce"
    HINGE = "hinge"
    LEAST_SQUARE = "least_squares"


class PatchAdversarialLoss(_Loss):
    """
    Calculates an adversarial loss on a Patch Discriminator or a Multi-scale Patch Discriminator.
    Warning: due to the possibility of using different criterions, the output of the discrimination
    mustn't be passed to a final activation layer. That is taken care of internally within the loss.

    Args:
        reduction: {``"none"``, ``"mean"``, ``"sum"``} Specifies the reduction to apply to the output.
        Defaults to ``"mean"``.
        - ``"none"``: no reduction will be applied.
        - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
        - ``"sum"``: the output will be summed.
        criterion: which criterion (hinge, least_squares or bce) you want to use on the discriminators outputs.
        Depending on the criterion, a different activation layer will be used. Make sure you don't run the outputs
        through an activation layer prior to calling the loss.
        no_activation_leastsq: if True, the activation layer in the case of least-squares is removed.
    """

    def __init__(
        self,
        reduction: LossReduction | str = LossReduction.MEAN,
        criterion: str = AdversarialCriterions.LEAST_SQUARE.value,
        no_activation_leastsq: bool = False,
    ) -> None:
        super().__init__(reduction=LossReduction(reduction).value)

        if criterion.lower() not in [m.value for m in AdversarialCriterions]:
            raise ValueError(
                "Unrecognised criterion entered for Adversarial Loss. Must be one in: %s"
                % ", ".join([m.value for m in AdversarialCriterions])
            )

        # Depending on the criterion, a different activation layer is used.
        self.real_label = 1.0
        self.fake_label = 0.0
        if criterion == AdversarialCriterions.BCE.value:
            self.activation = get_act_layer("SIGMOID")
            self.loss_fct = torch.nn.BCELoss(reduction=reduction)
        elif criterion == AdversarialCriterions.HINGE.value:
            self.activation = get_act_layer("TANH")
            self.fake_label = -1.0
        elif criterion == AdversarialCriterions.LEAST_SQUARE.value:
            if no_activation_leastsq:
                self.activation = None
            else:
                self.activation = get_act_layer(name=("LEAKYRELU", {"negative_slope": 0.05}))
            self.loss_fct = torch.nn.MSELoss(reduction=reduction)

        self.criterion = criterion
        self.reduction = reduction

    def get_target_tensor(self, input: torch.FloatTensor, target_is_real: bool) -> torch.Tensor:
        """
        Gets the ground truth tensor for the discriminator depending on whether the input is real or fake.

        Args:
            input: input tensor from the discriminator (output of discriminator, or output of one of the multi-scale
            discriminator). This is used to match the shape.
            target_is_real: whether the input is real or wannabe-real (1s) or fake (0s).
        Returns:
        """
        filling_label = self.real_label if target_is_real else self.fake_label
        label_tensor = torch.tensor(1).fill_(filling_label).type(input.type()).to(input[0].device)
        label_tensor.requires_grad_(False)
        return label_tensor.expand_as(input)

    def get_zero_tensor(self, input: torch.FloatTensor) -> torch.Tensor:
        """
        Gets a zero tensor.

        Args:
            input: tensor which shape you want the zeros tensor to correspond to.
        Returns:
        """

        zero_label_tensor = torch.tensor(0).type(input[0].type()).to(input[0].device)
        zero_label_tensor.requires_grad_(False)
        return zero_label_tensor.expand_as(input)

    def forward(
        self, input: torch.FloatTensor | list, target_is_real: bool, for_discriminator: bool
    ) -> torch.Tensor | list[torch.Tensor]:
        """

        Args:
            input: output of Multi-Scale Patch Discriminator or Patch Discriminator; being a list of
            tensors or a tensor; they shouldn't have gone through an activation layer.
            target_is_real: whereas the input corresponds to discriminator output for real or fake images
            for_discriminator: whereas this is being calculated for discriminator or generator loss. In the last
            case, target_is_real is set to True, as the generator wants the input to be dimmed as real.
        Returns: if reduction is None, returns a list with the loss tensors of each discriminator if multi-scale
        discriminator is active, or the loss tensor if there is just one discriminator. Otherwise, it returns the
        summed or mean loss over the tensor and discriminator/s.

        """

        if not for_discriminator and not target_is_real:
            target_is_real = True  # With generator, we always want this to be true!
            warnings.warn(
                "Variable target_is_real has been set to False, but for_discriminator is set"
                "to False. To optimise a generator, target_is_real must be set to True."
            )

        if type(input) is not list:
            input = [input]
        target_ = []
        for _, disc_out in enumerate(input):
            if self.criterion != AdversarialCriterions.HINGE.value:
                target_.append(self.get_target_tensor(disc_out, target_is_real))
            else:
                target_.append(self.get_zero_tensor(disc_out))

        # Loss calculation
        loss = []
        for disc_ind, disc_out in enumerate(input):
            if self.activation is not None:
                disc_out = self.activation(disc_out)
            if self.criterion == AdversarialCriterions.HINGE.value and not target_is_real:
                loss_ = self.forward_single(-disc_out, target_[disc_ind])
            else:
                loss_ = self.forward_single(disc_out, target_[disc_ind])
            loss.append(loss_)

        if loss is not None:
            if self.reduction == LossReduction.MEAN.value:
                loss = torch.mean(torch.stack(loss))
            elif self.reduction == LossReduction.SUM.value:
                loss = torch.sum(torch.stack(loss))

        return loss

    def forward_single(self, input: torch.FloatTensor, target: torch.FloatTensor) -> torch.Tensor | None:
        if (
            self.criterion == AdversarialCriterions.BCE.value
            or self.criterion == AdversarialCriterions.LEAST_SQUARE.value
        ):
            return self.loss_fct(input, target)
        elif self.criterion == AdversarialCriterions.HINGE.value:
            minval = torch.min(input - 1, self.get_zero_tensor(input))
            return -torch.mean(minval)
        else:
            return None
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor


class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative losses for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each losses element in the batch.
    """

    def __init__(self,
                 apply_nonlin=None, alpha=None, gamma=2,
                 balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):

        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous() # batch, pixel_num, class_num
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long() # pixel_num, class

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        print(f'one_hot_key = {one_hot_key}')
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()
        gamma = self.gamma
        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)

        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss

class Multiclass_FocalLoss(nn.Module):

    '''
    Multi-class Focal loss implementation
    '''

    def __init__(self, gamma=2):
        super(Multiclass_FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        """
        input: [N, C] -> raw probability
        target: [N, ] -> 0 , ... , C-1 --> C class index
        """
        # [1] calculate class by weight
        # input is probability
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = ((1-pt)**self.gamma) * logpt

        # [2] what is nll_loss
        # negative log likelihood loss
        loss = F.nll_loss(logpt,
                          target.type(torch.LongTensor).to(logpt.device),)
        return loss




def dice_coeff(input: Tensor,
               target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    input = input.flatten(start_dim=0, end_dim=1)
    target = target.flatten(start_dim=0, end_dim=1)
    return dice_coeff(input, target, reduce_batch_first,  # False
                      epsilon)


def dice_loss(input: Tensor,
              target: Tensor,
              multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    # good model have high dice_coefficient
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    dice_coefficient = fn(input, target, reduce_batch_first=True)
    dice_loss = 1 - dice_coefficient
    return dice_loss

