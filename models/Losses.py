import os
project_index = os.getcwd().find('skinCancer')
root = os.getcwd()[0:project_index] + 'skinCancer'
import sys
sys.path.append(root)
from config.BasicConfig import opt

import abc
from torch.nn.functional import cross_entropy, mse_loss
import torch






class BasicLoss(abc.ABC):
    def __init__(self, if_store=True, location=opt.loss_storer_location):
        self.if_store = if_store
        self.location = location

    @abc.abstractmethod
    def __call__(self, y_hat, y_true):
        pass

class CrossEntropyLossWithPenalty(BasicLoss):
    """
    带有L1正则化的Loss，
    y_hat为([p1, 1-p1], [p2, 1-p2], ..., [pm, 1-pm]), [l1, l2, ..., lm]
    y_true为([c1, c2, ..., cm], [l1, l2, ..., lm])
    其中m为batch_size
    最终得到的batch_loss = cross_entropy()
    """
    def __init__(self, alpha = opt.loss_alpha):
        super(CrossEntropyLossWithPenalty, self).__init__()
        self.alpha = alpha


    def __call__(self, y_hat, y_true):
        class_true, reg_true = y_true[:,0], y_true[:, 1]
        class_hat, reg_hat = y_hat[:, 0:2], y_hat[:, -1]
        res = cross_entropy(class_hat, class_true) + self.alpha * mse_loss(reg_hat, reg_true.float())
        return res




if __name__ == '__main__':
    cr = CrossEntropyLossWithPenalty()
    # input = torch.randn(3, 5, requires_grad=True)
    # target = [torch.as_tensor([0, 4, 0], dtype=torch.int64), torch.as_tensor([0.5, 2, 3])]
    # res = cr(input, target)
    # print(res)
    y_hat = torch.rand([16, 3]).long()
    y_true = torch.rand([16, 2]).long()
    a = cr(y_hat, y_true)
    print(a)