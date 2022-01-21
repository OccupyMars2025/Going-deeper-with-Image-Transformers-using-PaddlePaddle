""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""
import paddle


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: paddle.Tensor, target: paddle.Tensor, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    # batch_size = target.size(0)
    batch_size = target.shape[0]
    # _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
    _, pred = paddle.topk(output, k=maxk, axis=1)

    pred = pred.t()
    # correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    correct = pred.equal(target.reshape([1, -1]).expand_as(pred))

    # return [correct[:k].reshape([-1]).float().sum(0) * 100. / batch_size for k in topk]
    return [correct[:k].reshape([-1]).float().sum() / batch_size for k in topk]

