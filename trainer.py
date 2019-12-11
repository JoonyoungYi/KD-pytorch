import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from models import *
from utils import progress_bar
import dataset


def __train_epoch(student_net, teacher_net, trainloader, device, criterion,
                  optimizer):
    student_net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        pseudo_targets = teacher_net(inputs)

        optimizer.zero_grad()
        outputs = student_net(inputs)
        loss = criterion(outputs, pseudo_targets, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx,
                     len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                     (train_loss / (batch_idx + 1), 100. * correct / total,
                      correct, total))


def __test_epoch(student_net, testloader, device, criterion):
    student_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = student_net(inputs)
            # loss = criterion(outputs, targets)

            # test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx,
                         len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                         (test_loss / (batch_idx + 1), 100. * correct / total,
                          correct, total))

    # Save checkpoint.
    acc = 100. * correct / total


def _make_criterion(alpha=0.5, T=4.0, mode='cse'):
    def criterion(outputs, targets, labels):
        if mode == 'cse':
            _p = F.log_softmax(outputs / T, dim=1)
            _q = F.softmax(targets / T, dim=1)
            _soft_loss = -torch.mean(torch.sum(_q * _p, dim=1))
        elif mode == 'mse':
            _p = F.softmax(outputs / T, dim=1)
            _q = F.softmax(targets / T, dim=1)
            _soft_loss = nn.MSELoss()(_p, _q) / 2
        else:
            raise NotImplementedError()

        _soft_loss = _soft_loss * T * T
        _hard_loss = F.cross_entropy(outputs, labels)
        loss = alpha * _soft_loss + (1. - alpha) * _hard_loss
        return loss

    return criterion


def _train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    print('==> Preparing data..')
    trainloader, testloader = dataset.get_loader()

    # Model
    print('==> Building model..')
    teacher_net = VGG('VGG16')
    teacher_net = teacher_net.to(device)
    if device == 'cuda':
        teacher_net = torch.nn.DataParallel(teacher_net)
        cudnn.benchmark = True

    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(
        './checkpoint/{}.pth'.format(args.teacher_model_name))
    teacher_net.load_state_dict(checkpoint['net'])

    student_net = StudentNet()
    student_net = student_net.to(device)
    if device == 'cuda':
        student_net = torch.nn.DataParallel(student_net)
        cudnn.benchmark = True

    criterion = _make_criterion(alpha=args.alpha, T=args.T, mode=args.kd_mode)
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            student_net.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'sgd-cifar10':
        optimizer = optim.SGD(student_net.parameters(), lr=0.1, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(student_net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError()

    for epoch_idx in range(1, args.n_epoch + 1):
        print('\nEpoch: %d' % epoch_idx)

        __train_epoch(student_net, teacher_net, trainloader, device, criterion,
                      optimizer)
        __test_epoch(student_net, testloader, device, criterion)

        if args.optimizer == 'sgd-cifar10':
            if epoch_idx == 150:
                optimizer = optim.SGD(
                    student_net.parameters(), lr=0.01, momentum=0.9)
            elif epoch_idx == 250:
                optimizer = optim.SGD(
                    student_net.parameters(), lr=0.001, momentum=0.9)


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument(
        '--lr',
        # default=0.1,
        default=0.0001,
        type=float,
        help='learning rate')
    parser.add_argument('--n_epoch', default=300, type=int, help='epoch')
    parser.add_argument(
        '--alpha', default=1.0, type=float, help='weight for soft loss.')
    parser.add_argument(
        '--T', default=1.0, type=float, help='T for Temperature scaling')
    parser.add_argument(
        '--teacher_model_name',
        default='ckpt',
        type=str,
        help='name for teacher model')
    parser.add_argument(
        '--optimizer',
        # default='sgd',
        # default='sgd-cifar10',
        default='adam',
        choices=[
            'sgd',
            'sgd-cifar10',  # pytorch-cifar 에서 썼던 optimizer.
            'adam'
        ],
        type=str,
        help='name for optimizer')
    parser.add_argument(
        '--kd_mode', default='cse', choices=['cse', 'mse'], type=str, help='')
    args = parser.parse_args()

    assert 0 <= args.alpha <= 1, "alpha should be between 0 and 1."

    if args.optimizer == 'sgd-cifar10':
        args.n_epoch = 300
        args.lr = 0.1

    _train(args)


if __name__ == '__main__':
    main()
