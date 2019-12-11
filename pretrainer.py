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


def __train_epoch(net, trainloader, device, criterion, optimizer):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
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


def __test_epoch(net, testloader, device, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx,
                         len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                         (test_loss / (batch_idx + 1), 100. * correct / total,
                          correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    return acc


def _pretrain(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0.0
    start_epoch = args.start_epoch

    # Data
    print('==> Preparing data..')
    trainloader, testloader = dataset.get_loader()

    # Model
    print('==> Building model..')
    if args.network == 'vgg':
        net = VGG('VGG16')
    elif args.network == 'studentnet':
        net = StudentNet()
    else:
        raise NotImplementedError()

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(
            'checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/{}.pth'.format(args.model_name))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError()

    for epoch_idx in range(start_epoch, start_epoch + args.n_epoch):
        print('\nEpoch: %d' % epoch_idx)

        __train_epoch(net, trainloader, device, criterion, optimizer)
        acc = __test_epoch(net, testloader, device, criterion)

        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch_idx,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/{}.pth'.format(args.model_name))
            best_acc = acc


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Pretraining')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument(
        '--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--n_epoch', default=300, type=int, help='epoch')
    parser.add_argument(
        '--model_name',
        default='test',
        # default='ckpt',
        # default='student-scratch',
        type=str,
        help='name for model')
    parser.add_argument(
        '--optimizer',
        default='sgd',
        choices=['sgd', 'adam'],
        type=str,
        help='name for optimizer')
    parser.add_argument(
        '--network',
        default='vgg',
        choices=['vgg', 'studentnet'],
        type=str,
        help='name for network')
    parser.add_argument(
        '--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()

    _pretrain(args)


if __name__ == '__main__':
    main()
