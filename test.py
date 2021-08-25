from util.infoNCE import SupervisedInfoNCELoss
import torch

def test_loss():
    x1 = torch.randn((3, 768))
    x2 = torch.randn((3, 768))
    x3 = torch.randn((3, 768))

    criterion = SupervisedInfoNCELoss(taf=1e3)
    print(criterion(x1, x2, x3))


if __name__ == '__main__':
    test_loss()