from util.infoNCE import SupervisedInfoNCELoss
import torch

def test_loss():
    x1 = torch.randn((3, 256))
    x2 = torch.randn((3, 256))
    x3 = torch.randn((3, 256))

    criterion = SupervisedInfoNCELoss(taf=0.01)
    print(criterion(x1, x2, x3))


if __name__ == '__main__':
    test_loss()