import math
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
import torchvision
from nets import CNN, resnet18, resnet50, CNNCifar10, MLP, AlexNet, vgg8_bn
from torch.utils.data import Dataset, TensorDataset

loss_fn = F.l1_loss


def estimate_gradient_objective(args, victim_model, clone_model, x, epsilon=1e-7, m=5, num_classes=10):
    clone_model.eval()
    victim_model.eval()
    with torch.no_grad():
        # Sample unit noise vector
        N, C, S = x.size(0), x.size(1), x.size(2)
        dim = S ** 2 * C
        u = np.random.randn(N * m * dim).reshape(-1, m, dim)  # generate random points from normal distribution
        d = np.sqrt(np.sum(u ** 2, axis=2)).reshape(-1, m, 1)  # map to a uniform distribution on a unit sphere
        u = torch.Tensor(u / d).view(-1, m, C, S, S)
        u = torch.cat((u, torch.zeros(N, 1, C, S, S)), dim=1)  # Shape N, m + 1, S^2

        u = u.view(-1, m + 1, C, S, S)

        evaluation_points = (x.view(-1, 1, C, S, S).cpu() + epsilon * u).view(-1, C, S, S)
        evaluation_points = torch.tanh(evaluation_points)  # Apply args.G_activation function

        # Compute the approximation sequentially to allow large values of m
        pred_victim = []
        pred_clone = []
        max_number_points = 32 * 156  # Hardcoded value to split the large evaluation_points tensor to fit in GPU

        for i in (range(N * m // max_number_points + 1)):
            pts = evaluation_points[i * max_number_points: (i + 1) * max_number_points]
            pts = pts.cuda()

            pred_victim_pts = victim_model(pts).detach()
            pred_clone_pts = clone_model(pts)

            pred_victim.append(pred_victim_pts)
            pred_clone.append(pred_clone_pts)

        pred_victim = torch.cat(pred_victim, dim=0).cuda()
        pred_clone = torch.cat(pred_clone, dim=0).cuda()

        u = u.cuda()

        pred_victim = F.log_softmax(pred_victim, dim=1).detach()
        pred_victim -= pred_victim.mean(dim=1).view(-1, 1).detach()

        # Compute loss
        loss_values = - loss_fn(pred_clone, pred_victim, reduction='none')
        loss_values = loss_values.mean(dim=1)
        loss_values = loss_values.view(-1, m + 1)

        # Compute difference following each direction
        differences = loss_values[:, :-1] - loss_values[:, -1].view(-1, 1)
        differences = differences.view(-1, m, 1, 1, 1)

        # Formula for Forward Finite Differences
        gradient_estimates = 1 / epsilon * differences * u[:, :-1]
        if args.forward_differences:
            gradient_estimates *= dim

        gradient_estimates = gradient_estimates.mean(dim=1).view(-1, C, S, S) / (num_classes * N)

        clone_model.train()
        loss_G = loss_values[:, -1].mean()
        return gradient_estimates.detach(), loss_G


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()

    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()


def get_model(dataset, net):
    if dataset == "cifar10":
        if net == "cnn":
            model = CNNCifar10().cuda()
        elif net == "vgg":
            model = vgg8_bn(num_classes=10).cuda()
        elif net == "resnet18":
            model = resnet18(num_classes=10).cuda()
    elif dataset == "cifar100":
        if net == "vgg":
            model = vgg8_bn(num_classes=100).cuda()
        elif net == "resnet50":
            model = resnet50(num_classes=100).cuda()
    elif dataset == "imagenet":
        if net == "vgg":
            model = vgg8_bn(num_classes=12).cuda()
        elif net == "resnet18":
            model = resnet50(num_classes=12).cuda()
    elif "mnist" in dataset:
        if net == "mlp":
            model = MLP().cuda()
        elif net == "lenet":
            model = CNN().cuda()
        elif net == "alexnet":
            model = AlexNet().cuda()
    elif dataset == "svhn":
        if net == "alexnet":
            model = CNNCifar10().cuda()
        elif net == "vgg":
            model = vgg8_bn(num_classes=10).cuda()
        elif net == "resnet18":
            model = resnet18(num_classes=10).cuda()
    return model


# def get_teacher_model(dataset, net):
#     if "mnist" in dataset:
#         if net == "mlp":
#             model = MLP().cuda()
#         elif net == "lenet":
#             model = CNN().cuda()
#         elif net == "alexnet":
#             model = AlexNet().cuda()
#     elif dataset == "svhn":
#         if net == "alexnet":
#             model = CNNCifar10().cuda()
#         elif net == "vgg":
#             model = vgg8_bn(num_classes=10).cuda()
#         elif net == "resnet18":
#             model = resnet18(num_classes=10).cuda()
#     elif dataset == "cifar10":
#         model = CNNCifar10().cuda()
#     elif dataset == "cifar100":
#         model = resnet50(num_classes=100).cuda()
#     elif dataset == "imagenet":
#         model = resnet18(num_classes=12).cuda()
#     return model
#
#
# def get_student_model(dataset):
#     if "mnist" in dataset:
#         model = CNN().cuda()
#     elif dataset == "svhn":
#         model = CNNCifar10().cuda()
#     elif dataset == "cifar10":
#         model = CNNCifar10().cuda()
#     elif dataset == "cifar100":
#         model = resnet50(num_classes=100).cuda()
#     elif dataset == "imagenet":
#         model = resnet18(num_classes=12).cuda()
#     return model


def cal_prob(black_net, data):
    with torch.no_grad():
        outputs = black_net(data.detach())
        score = F.softmax(outputs, dim=1)  # score-based
    score = score.detach().cpu().numpy()
    score = torch.from_numpy(score).cuda().float()
    return score


def cal_label(black_net, data):
    with torch.no_grad():
        outputs = black_net(data.detach())
        _, label = torch.max(outputs.data, 1)
    label = label.detach().cpu().numpy()
    label = torch.from_numpy(label).cuda().long()
    return label


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            total += data.shape[0]
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= total
    acc = 100. * correct / total
    return acc, test_loss


def print_log(strs, log):
    print(strs)
    log.write(strs)


def get_dataset(dataset):
    data_dir = '/home/zhangjie29/dataset'
    if dataset == "mnist":
        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=transforms.Compose(
                                           [transforms.ToTensor()]))
        test_dataset = datasets.MNIST(data_dir, train=False,download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                      ]))
    elif dataset == "fmnist":
        train_dataset = datasets.FashionMNIST(data_dir, train=True,download=True,
                                              transform=transforms.Compose(
                                                  [transforms.ToTensor()]))
        test_dataset = datasets.FashionMNIST(data_dir, train=False,download=True,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                             ]))
    elif dataset == "svhn":
        train_dataset = datasets.SVHN(data_dir, split="train",download=True,
                                      transform=transforms.Compose(
                                          [transforms.ToTensor()]))
        test_dataset = datasets.SVHN(data_dir, split="test",download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                     ]))
    elif dataset == "cifar10":
        train_dataset = datasets.CIFAR10(data_dir, train=True,download=True,
                                         transform=transforms.Compose(
                                             [
                                                 transforms.RandomCrop(32, padding=4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                             ]))
        test_dataset = datasets.CIFAR10(data_dir, train=False,download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                        ]))
    elif dataset == "cifar100":
        train_dataset = datasets.CIFAR100(data_dir, train=True,download=True,
                                          transform=transforms.Compose(
                                              [
                                                  transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                              ]))
        test_dataset = datasets.CIFAR100(data_dir, train=False,download=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                         ]))
    elif dataset == "imagenet":
        # =====================
        X_train, y_train = np.load("imagenet/X_train.npy"), np.load("imagenet/y_train.npy")
        X_test, y_test = np.load("imagenet/X_test.npy"), np.load("imagenet/y_test.npy")
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
        # =====================

    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256,
                                               shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256,
                                              shuffle=False, num_workers=4)

    return train_loader, test_loader


class ScoreLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(ScoreLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, target):
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # [N, C, HW]
            logits = logits.transpose(1, 2)  # [N, HW, C]
            logits = logits.contiguous().view(-1, logits.size(2))  # [NHW, C]
        target = target.view(-1, 1)  # [NHW，1]

        score = F.log_softmax(logits, 1)  # score-based
        score = score.gather(1, target)  # [NHW, 1]
        loss = -1 * score

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)


def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


class MultiTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [t(x) for t in self.transform]

    def __repr__(self):
        return str(self.transform)


def pack_images(images, col=None, channel_last=False, padding=1):
    # N, C, H, W
    if isinstance(images, (list, tuple)):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0, 3, 1, 2)  # make it channel first
    assert len(images.shape) == 4
    assert isinstance(images, np.ndarray)

    N, C, H, W = images.shape
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))

    pack = np.zeros((C, H * row + padding * (row - 1), W * col + padding * (col - 1)), dtype=images.dtype)
    for idx, img in enumerate(images):
        h = (idx // col) * (H + padding)
        w = (idx % col) * (W + padding)
        pack[:, h:h + H, w:w + W] = img
    return pack


def save_image_batch(imgs, output, col=None, size=None, pack=True):
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy() * 255).astype('uint8')
    base_dir = os.path.dirname(output)
    if base_dir != '':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        imgs = pack_images(imgs, col=col).transpose(1, 2, 0).squeeze()
        imgs = Image.fromarray(imgs)
        if size is not None:
            if isinstance(size, (list, tuple)):
                imgs = imgs.resize(size)
            else:
                w, h = imgs.size
                max_side = max(h, w)
                scale = float(size) / float(max_side)
                _w, _h = int(w * scale), int(h * scale)
                imgs = imgs.resize([_w, _h])
        imgs.save(output)
    else:
        output_filename = output.strip('.png')
        for idx, img in enumerate(imgs):
            if img.shape[0] == 1:
                img = Image.fromarray(img[0])
            else:
                img = Image.fromarray(img.transpose(1, 2, 0))
            img.save(output_filename + '-%d.png' % (idx))


def _collect_all_images(root, postfix=['png', 'jpg', 'jpeg', 'JPEG'], batch_size=250):
    images = []
    if isinstance(postfix, str):
        postfix = [postfix]
    for dirpath, dirnames, files in os.walk(root):  
        files.sort()
        files = files[-batch_size * 400:]  
        for pos in postfix:
            for f in files:
                if f.endswith(pos):
                    images.append(os.path.join(dirpath, f))
    return images


class UnlabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, batch_size=250):
        self.root = os.path.abspath(root)
        self.images = _collect_all_images(self.root, batch_size=batch_size)  # [ os.path.join(self.root, f) for f in os.listdir( root ) ]
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return 'Unlabeled data:\n\troot: %s\n\tdata mount: %d\n\ttransforms: %s' % (
            self.root, len(self), self.transform)


class ImagePool(object):
    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._idx = 0

    def add(self, imgs, targets=None):
        save_image_batch(imgs, os.path.join(self.root, "%d.png" % (self._idx)), pack=False)
        self._idx += 1

    def get_dataset(self, transform=None, batch_size=250):
        return UnlabeledImageDataset(self.root, transform=transform, batch_size=batch_size)
