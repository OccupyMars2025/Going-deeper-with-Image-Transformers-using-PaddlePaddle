# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


# class LabelSmoothingCrossEntropy(nn.Module):
class LabelSmoothingCrossEntropy(nn.Layer):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        # logprobs = F.log_softmax(x, dim=-1)
        logprobs = F.log_softmax(x, axis=-1)
        # nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        target_list = paddle.cast(target, dtype='int64').tolist()
        image_id_list = list(range(len(target)))
        nll_loss = -logprobs[image_id_list, target_list]

        # nll_loss = nll_loss.squeeze(1)
        # smooth_loss = -logprobs.mean(dim=-1)
        smooth_loss = paddle.mean(-logprobs, axis=-1)

        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# class SoftTargetCrossEntropy(nn.Module):
class SoftTargetCrossEntropy(nn.Layer):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        # loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        loss = paddle.sum(-target * F.log_softmax(x, axis=-1), axis=-1)
        return loss.mean()
