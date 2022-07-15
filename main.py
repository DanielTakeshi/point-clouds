"""Doing some brief tests.

Careful, pytorch_geometric gets updated rapidly. Changes to watch out for:
- Using `norm` vs `batch_norm` for the MLP.
"""
import time
import argparse
import os.path as osp
import wandb
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_exp_dir(args):
    """TODO: get experiment dir for wandb."""
    exp_dir = f'{args.model}_{args.data}'
    exp_dir += f'_{str(args.seed).zfill(3)}'
    if 'pointnet2' in exp_dir:
        exp_dir = exp_dir.replace('pointnet2', 'PNet2')
    if 'point_transformer' in exp_dir:
        exp_dir = exp_dir.replace('point_transformer', 'PtTransf')
    return exp_dir


def train(model, train_loader):
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.pos, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(model, test_loader):
    model.eval()
    total_loss = 0
    correct = 0

    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.pos, data.batch)
        loss = F.nll_loss(out, data.y)
        pred = out.max(dim=1)[1]
        total_loss += loss.item() * data.num_graphs
        correct += pred.eq(data.y).sum().item()

    return (total_loss / len(test_loader.dataset),
            correct / len(test_loader.dataset))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='modelnet10')
    p.add_argument('--model', type=str, default='pointnet2')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=32)
    args = p.parse_args()

    # Bells and whistles.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load data if it's not there already. Handle data loaders, etc.
    if args.data == 'modelnet10':
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data/ModelNet10')
        pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
        train_dataset = ModelNet(path, '10', True, transform, pre_transform)
        test_dataset = ModelNet(path, '10', False, transform, pre_transform)
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=6)
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=6)
    elif args.data == 'modelnet40':
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data/ModelNet40')
        pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
        train_dataset = ModelNet(path, '40', True, transform, pre_transform)
        test_dataset = ModelNet(path, '40', False, transform, pre_transform)
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=6)
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=6)
    else:
        raise ValueError(args.data)

    # Create the model and optimizer.
    if args.model == 'pointnet2':
        from pointnet2_classification import Net
        model = Net(
            out_channels=train_dataset.num_classes
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = None
    elif args.model == 'point_transformer':
        from point_transformer_classification import Net
        model = Net(
            in_channels=0,
            out_channels=train_dataset.num_classes,
            dim_model=[32, 64, 128, 256, 512],
            k=16
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=0.5)
    else:
        raise ValueError(args.model)

    # Bells and whistles.
    exp_dir = get_exp_dir(args)
    wandb.init(project="point-cloud", entity="danieltakeshi", name=exp_dir)
    wandb.config.update(args)

    print(f'The classification model:\n{model}')
    print(f'Parameters: {count_parameters(model)}.\n')
    start = time.time()

    # Train and test!
    for epoch in range(1, args.epochs+1):
        train_loss = train(model, train_loader)
        test_loss, test_acc = test(model, test_loader)
        if scheduler is not None:
            scheduler.step()

        print((f'Epoch {epoch:03d}, Loss(Tr/Te): {train_loss:.4f},{test_loss:.4f}, '
            f'Test: {test_acc:.4f}'))
        wandb_dict = {
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'elapsed_t': time.time() - start,
        }
        wandb.log(wandb_dict)

    print(f'Done!')
