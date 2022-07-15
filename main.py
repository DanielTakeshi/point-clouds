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
from torch_geometric.datasets import ModelNet, ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.utils import intersection_and_union as i_and_u


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_exp_dir(args):
    """Experiment dir for wandb."""
    exp_dir = f'{args.model}_{args.data}'
    exp_dir += f'_{str(args.seed).zfill(3)}'
    if 'pointnet2' in exp_dir:
        exp_dir = exp_dir.replace('pointnet2', 'PNet2')
    if 'point_transformer' in exp_dir:
        exp_dir = exp_dir.replace('point_transformer', 'PtTransf')
    return exp_dir


def train_classify(model, train_loader):
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
def test_classify(model, test_loader):
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


def train_segment(model, train_loader):
    model.train()

    total_loss = correct_nodes = total_nodes = 0
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.pos, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes

    train_loss = total_loss / i
    train_acc = correct_nodes / total_nodes
    return (train_loss, train_acc)


@torch.no_grad()
def test_segment(model, test_loader):
    model.eval()

    y_mask = test_loader.dataset.y_mask
    ious = [[] for _ in range(len(test_loader.dataset.categories))]

    for data in test_loader:
        data = data.to(device)
        pred = model(data.x, data.pos, data.batch).argmax(dim=1)

        i, u = i_and_u(pred, data.y, test_loader.dataset.num_classes, data.batch)
        iou = i.cpu().to(torch.float) / u.cpu().to(torch.float)
        iou[torch.isnan(iou)] = 1

        # Find and filter the relevant classes for each category.
        for iou, category in zip(iou.unbind(), data.category.unbind()):
            ious[category.item()].append(iou[y_mask[category]])

    # Compute mean IoU.
    ious = [torch.stack(iou).mean(0).mean(0) for iou in ious]
    return torch.tensor(ious).mean().item()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='modelnet10')
    p.add_argument('--model', type=str, default='pointnet2')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--dropout', type=float, default=0.5,
        help='Dropout for PointNet++, does not seem used in PointTransformer')
    args = p.parse_args()

    # Bells and whistles.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    task_type = 'classification'

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

    elif args.data == 'shapenet':
        category = 'Airplane'  # Pass in `None` to train on all categories.
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'ShapeNet')
        transform = T.Compose([
            T.RandomTranslate(0.01),
            T.RandomRotate(15, axis=0),
            T.RandomRotate(15, axis=1),
            T.RandomRotate(15, axis=2)
        ])
        pre_transform = T.NormalizeScale()
        train_dataset = ShapeNet(path, category, split='trainval', transform=transform,
                                 pre_transform=pre_transform)
        test_dataset = ShapeNet(path, category, split='test', pre_transform=pre_transform)

        # Not sure if needed but we might want a smaller batch size, like 12?
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=6)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=6)
        task_type = 'segmentation'

    else:
        raise ValueError(args.data)

    # Create the model and optimizer.
    if args.model == 'pointnet2' and task_type == 'classification':
        from pointnet2_classification import Net
        model = Net(
            out_channels=train_dataset.num_classes,
            dropout=args.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None

    elif args.model == 'pointnet2' and task_type == 'segmentation':
        from pointnet2_segmentation import Net
        model = Net(
            num_classes=train_dataset.num_classes,
            dropout=args.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None

    elif args.model == 'point_transformer' and task_type == 'classification':
        from point_transformer_classification import Net
        model = Net(
            in_channels=0,
            out_channels=train_dataset.num_classes,
            dim_model=[32, 64, 128, 256, 512],
            k=16,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=0.5)

    elif args.model == 'point_transformer' and task_type == 'segmentation':
        from point_transformer_segmentation import Net
        model = Net(
            in_channels=3,
            out_channels=train_dataset.num_classes,
            dim_model=[32, 64, 128, 256, 512],
            k=16,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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

        if task_type == 'classification':
            train_loss = train_classify(model, train_loader)
            test_loss, test_acc = test_classify(model, test_loader)
            if scheduler is not None:
                scheduler.step()

            print((f'Epoch {epoch:03d}, Loss(Tr/Te): {train_loss:.4f},{test_loss:.4f}, '
                f'Test: {test_acc:.4f}'))
            wandb_dict = {
                'epoch': epoch,
                'elapsed_t': time.time() - start,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'test_acc': test_acc,
            }
        elif task_type == 'segmentation':
            train_loss, train_acc = train_segment(model, train_loader)
            test_iou = test_segment(model, test_loader)
            if scheduler is not None:
                scheduler.step()

            print(f'Epoch: {epoch:02d}, TrLoss: {train_loss:.4f}, Test IoU: {test_iou:.4f}')
            wandb_dict = {
                'epoch': epoch,
                'elapsed_t': time.time() - start,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_iou': test_iou,
            }
        else:
            raise ValueError(task_type)

        wandb.log(wandb_dict)
    print(f'Done!')
