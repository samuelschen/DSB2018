import torch.nn as nn

class BinaryCrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, inputs, targets):
        return self.bce_loss(inputs, targets)

# DICE = 2 * Sum(PiGi) / (Sum(Pi) + Sum(Gi))
# Refer https://github.com/pytorch/pytorch/issues/1249 for Laplace/Additive smooth
class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets):
        smooth = 1.
        num = targets.size(0) # number of batches
        m1 = inputs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
        score = (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        dice = score.sum() / num
        # three kinds of loss formulas: (1) 1 - dice (2) -dice (3) -torch.log(dice)
        return 1. - dice


# Jaccard/IoU = Sum(PiGi) / (Sum(Pi) + Sum(Gi) - Sum(PiGi)) 
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets):
        smooth = 1.
        num = targets.size(0) # number of batches
        m1 = inputs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
        score = (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) - intersection.sum(1) + smooth)
        iou = score.sum() / num
        # three kinds of loss formulas: (1) 1 - iou (2) -iou (3) -torch.log(iou)
        return 1. - iou

def criterion(preds, labels):
# (1) BCE Loss
#     return BinaryCrossEntropyLoss2d().forward(preds, labels)
# (2) BCE Loss + DICE Loss
#     return BinaryCrossEntropyLoss2d().forward(preds, labels) + \
#            SoftDiceLoss().forward(preds, labels) 
# (3) BCE Loss + Jaccard/IoU Loss
    return BinaryCrossEntropyLoss2d().forward(preds, labels) + \
           IoULoss().forward(preds, labels)