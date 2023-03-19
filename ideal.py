#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse
import warnings
from collections import Counter

import numpy as np
import torch, wandb, os
import torch.nn.functional as F
import torch.optim as optim
from kornia import augmentation
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from nets import Generator_2
from utils import ScoreLoss, ImagePool, MultiTransform, reset_model, get_dataset, cal_label, setup_seed, \
    print_log, test, save_checkpoint, get_model

warnings.filterwarnings('ignore')


class Synthesizer():
    def __init__(self, generator, nz, num_classes, img_size,
                 iterations, lr_g,
                 sample_batch_size, save_dir, dataset):
        super(Synthesizer, self).__init__()
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.nz = nz
        self.score_loss = ScoreLoss()
        self.num_classes = num_classes
        self.sample_batch_size = sample_batch_size
        self.save_dir = save_dir
        self.data_pool = ImagePool(root=self.save_dir)
        self.data_iter = None
        self.dataset = dataset

        self.generator = generator.cuda().train()

        self.aug = MultiTransform([
            # global view
            transforms.Compose([
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                augmentation.RandomHorizontalFlip(),
            ]),
            # local view
            transforms.Compose([
                augmentation.RandomResizedCrop(size=[self.img_size[-2], self.img_size[-1]], scale=[0.25, 1.0]),
                augmentation.RandomHorizontalFlip(),
            ]),
        ])
        # =======================
        if not ("cifar" in dataset):
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
        else:
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])

    def get_data(self):
        datasets = self.data_pool.get_dataset(transform=self.transform,
                                              batch_size=self.sample_batch_size)  # 获取程序运行到现在所有的图片
        self.data_loader = torch.utils.data.DataLoader(
            datasets, batch_size=self.sample_batch_size, shuffle=True,
            num_workers=4, pin_memory=True, )
        return self.data_loader

    def gen_data(self, student, teacher, epoch):
        teacher.eval()
        student.eval()
        best_cost = 1e6
        best_inputs = None
        z = torch.randn(size=(self.sample_batch_size, self.nz)).cuda()  #
        z.requires_grad = True
        # targets = torch.randint(low=0, high=self.num_classes, size=(self.sample_batch_size,))
        targets = torch.tensor([i for i in range(self.num_classes)])
        targets = targets.repeat(int(self.sample_batch_size / self.num_classes))
        targets = targets.cuda()

        reset_model(self.generator)
        optimizer = torch.optim.Adam(self.generator.parameters(), self.lr_g, betas=[0.5, 0.999])
        for it in range(self.iterations):
            optimizer.zero_grad()
            inputs = self.generator(z)  # bs,nz
            global_view, _ = self.aug(inputs)  # crop and normalize
            # s_out = teacher(global_view)
            # -------------- ce loss ----------------------
            s_out = student(global_view)
            loss_ce = F.cross_entropy(s_out, targets)

            prob = F.softmax(s_out, dim=-1)
            loss_infor = 5 * torch.mul(prob, torch.log(prob)).mean()
            loss = loss_infor + loss_ce

            if best_cost > loss.item() or best_inputs is None:
                best_cost = loss.item()
                best_inputs = inputs.data
            loss.backward()
            optimizer.step()
        self.data_pool.add(best_inputs)


def args_parser():
    parser = argparse.ArgumentParser()
    # for wandb

    parser.add_argument('--wandb', type=int, default=1, help='use wandb')
    parser.add_argument('--group', type=str, default="distill", help='wandb group')
    parser.add_argument('--exp_name', type=str, default='', help='name of this experiment')

    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--score', type=float, default=0,
                        help="number of rounds of training")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--save_dir', default='run/mnist', type=str)
    parser.add_argument('--lr_g', default=1e-3, type=float,
                        help='initial learning rate for generation')
    parser.add_argument('--g_steps', default=30, type=int, metavar='N',
                        help='number of iterations for generation')
    parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--nz', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--synthesis_batch_size', default=256, type=int)
    # Misc
    parser.add_argument('--seed', default=2021, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--type', default="score", type=str,
                        help='score or label')
    parser.add_argument('--model', default="", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--other', default="", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--net', type=str, default="cifar10",
                        help='dataset')
    args = parser.parse_args()
    return args


def kd_train(synthesizer, model, optimizer, criterion):
    sub_net, blackBox_net = model
    sub_net.train()
    blackBox_net.eval()
    data_loader = synthesizer.get_data()
    empty = True
    s_label = None
    t_label = None
    total = 0.0
    total_loss = 0.0
    correct = 0.0
    for idx, (images) in enumerate(data_loader):
        optimizer.zero_grad()
        images = images.cuda()
        # label-only
        substitute_outputs = sub_net(images)
        label = cal_label(blackBox_net, images)  # label
        loss = criterion(substitute_outputs, label)

        loss.backward()
        optimizer.step()
        if empty:
            t_label = label.cpu().numpy()
            s_label = torch.max(substitute_outputs.data, 1)[1].cpu().numpy()
            empty = False
        else:
            t_label = np.concatenate((t_label, label.cpu().numpy()), axis=0)
            s_label = np.concatenate((s_label, torch.max(substitute_outputs.data, 1)[1].cpu().numpy()), axis=0)
        total += images.shape[0]
        total_loss += loss.item()
        pred = torch.max(substitute_outputs, 1)[1]
        correct += pred.eq(label.view_as(pred)).sum().item()
    print(
        "Train loss:{}, Acc:{}, \t t_label:{}, s_label:{}".format(total_loss / total, correct / total, Counter(t_label),
                                                                  Counter(s_label)))
    return Counter(t_label)


def adjust_learning_rate(lr, optimizer, epoch, lens):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # step = int(lens / 2) - 1
    step = 210
    lr = lr * (0.1 ** (epoch // step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class maxMarginLoss(torch.nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(maxMarginLoss, self).__init__()
        m_list = torch.cuda.FloatTensor(cls_num_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        # ========== ours-2 ==========
        output = x + 0.1 * torch.log(self.m_list + 1e-7)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


def get_cls_num(cls_list_counter, num_class):
    cls_list = [0 for i in range(num_class)]
    for num, count in enumerate(dict(cls_list_counter).items()):
        cls_list[count[0]] = count[1]
    for i in range(num_class):
        if cls_list[i] == 0:
            cls_list[i] = 2
    return cls_list


if __name__ == '__main__':

    args = args_parser()
    if args.wandb == 1:
        wandb.init(config=args, project="IDEAL", group=args.group, name=args.exp_name)
    setup_seed(args.seed)

    train_loader, val_loader = get_dataset(args.dataset)

    save_dir = 'saved/df_{}_{}_{}'.format(args.dataset, args.net, args.exp_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    log = open('{}/log.txt'.format(save_dir), 'w')
    sub_net = get_model(args.dataset, args.net)
    blackBox_net = get_model(args.dataset, args.net)
    pretrained = "pretrained/{}_{}".format(args.dataset, args.net)
    state_dict = torch.load('{}/{}_{}.pkl'.format(pretrained, args.dataset, args.net))
    blackBox_net.load_state_dict(state_dict)

    print_log("===================================== \n", log)
    acc, _ = test(blackBox_net, val_loader)
    print_log("Accuracy of the black-box model:{:.3} % \n".format(acc), log)
    acc, _ = test(sub_net, val_loader)
    print_log("Accuracy of the substitute model:{:.3} % \n".format(acc), log)

    print_log("===================================== \n", log)
    log.flush()

    ################################################
    # data generator
    ################################################
    nz = args.nz
    nc = 3 if "cifar" in args.dataset or args.dataset == "svhn" or args.dataset == "imagenet" else 1
    img_size = 32 if "cifar" in args.dataset or args.dataset == "svhn" or args.dataset == "imagenet" else 28
    generator = Generator_2(nz=nz, ngf=64, img_size=img_size, nc=nc).cuda()
    img_size2 = (3, 32, 32) if "cifar" in args.dataset or args.dataset == "svhn" or args.dataset == "imagenet" else (
        1, 28, 28)
    num_class = 100 if args.dataset == "cifar100" else 10
    if args.dataset == "imagenet":
        num_class = 12
    synthesizer = Synthesizer(generator,
                              nz=nz,
                              num_classes=num_class,
                              img_size=img_size2,
                              iterations=args.g_steps,
                              lr_g=args.lr_g,
                              sample_batch_size=args.batch_size,
                              save_dir=args.save_dir,
                              dataset=args.dataset)

    optimizer = optim.SGD(sub_net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    sub_net.train()
    best_acc = -1
    best_acc_ckpt = '{}/acc.pth'.format(save_dir)
    ################################################
    # parallel
    sub_net = torch.nn.DataParallel(sub_net)
    blackBox_net = torch.nn.DataParallel(blackBox_net)
    generator = torch.nn.DataParallel(generator)
    ################################################
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = LabelSmoothingCrossEntropy()
    stop_gen = 400
    for epoch in tqdm(range(args.epochs)):
        # adjust_learning_rate(args.lr, optimizer, epoch, args.epochs)
        # 1. Data synthesis
        if epoch < stop_gen:
            synthesizer.gen_data(sub_net, blackBox_net, epoch)  # g_steps
        cls_list_counter = kd_train(synthesizer, [sub_net, blackBox_net], optimizer, criterion)
        # if epoch < stop_gen:
        #     synthesizer.gen_data(sub_net, blackBox_net, epoch)  # g_steps
        #     cls_list_counter = kd_train(synthesizer, [sub_net, blackBox_net], optimizer, criterion)
        # else:
        #     cls_list = get_cls_num(cls_list_counter, num_class)
        #     criterion = maxMarginLoss(cls_num_list=cls_list, max_m=0.8, s=10, weight=None).cuda()
        #     cls_list_counter = kd_train(synthesizer, [sub_net, blackBox_net], optimizer, criterion)

        acc, test_loss = test(sub_net, val_loader)

        save_checkpoint({
            'state_dict': sub_net.state_dict(),
            'epoch': epoch,
        }, acc > best_acc, best_acc_ckpt)

        best_acc = max(best_acc, acc)
        # tf_writer.add_scalar('test_acc', acc, epoch)
        wandb.log({'accuracy': acc})
        print_log("Dataset:{}, Epoch: {}, Accuracy of the substitute model:{:.3} %, best accuracy:{:.3} % \n".format(
            args.dataset, epoch, acc, best_acc), log)
        log.flush()


"""
CUDA_VISIBLE_DEVICES=2 python ideal.py --epochs=100 --save_dir=run/mnist_1  --dataset=mnist --net=lenet  --exp_name=kd_mnist_lenet --batch_size=250 

CUDA_VISIBLE_DEVICES=3 python ideal.py --epochs=400 --save_dir=run/cifar10_1  --dataset=cifar10 --net=cnn --g_steps=5 --exp_name=kd_cifar10_cnn --batch_size=250 


"""