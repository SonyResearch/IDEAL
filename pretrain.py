from __future__ import print_function
import argparse  
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nets import resnet34, CNN, CNNCifar10, resnet18, resnet50, MLP, AlexNet, vgg8_bn
from utils import test, get_dataset
import warnings,wandb,os

warnings.filterwarnings('ignore')


def train(model, train_loader, optimizer, epoch):
    model.train()

    for idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


def adjust_learning_rate(lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 40))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_model(dataset, net):
    if "mnist" in dataset:
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
            model = CNNCifar10().cuda()
        elif net == "resnet18":
            model = resnet18(num_classes=10).cuda()
    elif dataset == "cifar10":
        if net == "cnn":
            model = CNNCifar10().cuda()
        elif net =="res18":
            model = resnet18(num_classes=10).cuda()
    elif dataset == "cifar100":
        model = resnet50(num_classes=100).cuda()
    elif dataset == "imagenet":
        model = resnet18(num_classes=12).cuda()
    return model


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dataset', type=str, default="cifar10",
                        help='dataset')
    parser.add_argument('--net', type=str, default="cifar10",
                        help='dataset')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--exp_name', type=str, default='pretrain_resnet32',
                        help='the name of this experiment')
    args = parser.parse_args()
    wandb.init(config=args, project="IDEAL", group="pretrain", name=args.exp_name)

    train_loader, test_loader = get_dataset(args.dataset)
    model = get_model(args.dataset, args.net)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    bst_acc = -1
    save_dir = "pretrained/{}_{}".format(args.dataset, args.net)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for epoch in range(1, args.epochs + 1):
        # adjust_learning_rate(args.lr, optimizer, epoch)
        train(model, train_loader, optimizer, epoch)
        acc, loss = test(model, test_loader)
        if acc > bst_acc:
            bst_acc = acc
            torch.save(model.state_dict(), '{}/{}_{}.pkl'.format(save_dir, args.dataset, args.net))
        wandb.log({'accuracy': acc})
        bst_acc = max(bst_acc, acc)
        print("Epoch:{},\t test_acc:{}, best_acc:{}".format(epoch, acc, bst_acc))


if __name__ == '__main__':
    main()

"""
CUDA_VISIBLE_DEVICES=0 python pretrain.py --dataset=cifar10 --net=cnn  --exp_name=pretrain_cifar10_cnn

CUDA_VISIBLE_DEVICES=1 python pretrain.py --dataset=mnist --net=lenet  --epochs=100 --exp_name=pretrain_mnist_lenet
"""
