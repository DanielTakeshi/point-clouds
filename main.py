"""Doing some brief tests.

Careful, pytorch_geometric gets updated rapidly. Changes to watch out for:
- Using `norm` vs `batch_norm` for the MLP.

Also need to watch out: if we install with conda, then those will import layers
that are out of date compared to the files from the GitHub? How does that dynamic
work?
"""
import time
import argparse
import os.path as osp
import wandb
import random
import numpy as np
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    return total_loss / len(train_dataset)


@torch.no_grad()
def test(model, test_loader):
    model.eval()
    correct = 0
    for data in test_loader:
        data = data.to(device)
        pred = model(data.x, data.pos, data.batch).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(test_loader.dataset)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='modelnet10')
    p.add_argument('--model', type=str, default='pointnet2')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    # Bells and whistles.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load ModelNet10 if it's not there already. Handle data loaders, etc.
    if args.data == 'modelnet10':
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/ModelNet10')
        pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
        train_dataset = ModelNet(path, '10', True, transform, pre_transform)
        test_dataset = ModelNet(path, '10', False, transform, pre_transform)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                                  num_workers=6)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
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
        model = Net(in_channels=0,
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
    print(f'The classification model:\n{model}')
    print(f'Parameters: {count_parameters(model)}.\n')
    wandb.init(project="point-cloud")
    stats = defaultdict(list)
    start = time.time()

    # Train and test!
    for epoch in range(1, 201):
        loss = train(model, train_loader)
        t_test = time.time()
        test_acc = test(model, test_loader)
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Test: {test_acc:.4f}')
        stats['train_loss'].append(loss)
        stats['test_acc'].append(test_acc)
        stats['elapsed_t'].append(time.time() - start)
        stats['individual_t_test'].append(time.time() - t_test)
        if scheduler is not None:
            scheduler.step()

    elapsed = time.time() - start
    elapsed_test = np.sum(stats['individual_t_test'])
    print(f'Elapsed time (total): {elapsed:0.1f}s')
    print(f'Elapsed time (test): {elapsed_test:0.1f}s')
