"""Precompute frozen-encoder features once and train the LTIC head on them (--cache_features)."""
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


class CachedViews(Dataset):
    """Train features stored as [views, N, D]; each sample draws one augmented view at random."""
    def __init__(self, feats, labels):
        self.feats = feats
        self.labels = labels

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, index):
        view = random.randrange(self.feats.size(0))
        return self.feats[view, index], self.labels[index]


def extract(encoder, loader, device, desc):
    net = nn.DataParallel(encoder) if torch.cuda.device_count() > 1 else encoder
    feats, labels = [], []
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == 'cuda'):
        for i, (images, target) in enumerate(loader):
            feats.append(net(images.to(device, non_blocking=True)).half().cpu())
            labels.append(target)
            if i % 100 == 0:
                print('=> extracting {} [{}/{}]'.format(desc, i, len(loader)))
    return torch.cat(feats), torch.cat(labels)


def _cache_path(args, txt, suffix=''):
    split = os.path.splitext(os.path.basename(txt))[0]
    norm = '_dsnorm' if args.dataset_norm else ''
    return os.path.join(args.cache_dir, '{}_{}{}_{}{}.pt'.format(args.dataset, args.backbone, norm, split, suffix))


def _load_or_extract(path, make):
    if os.path.isfile(path):
        print("=> loading cached features '{}'".format(path))
        return torch.load(path)
    cache = make()
    torch.save(cache, path)
    print("=> saved cached features '{}'".format(path))
    return cache


def build_cached_loaders(encoder, data, args):
    """Return (train_loader, val_loader) yielding (features, target) instead of (images, target)."""
    os.makedirs(args.cache_dir, exist_ok=True)
    device = next(encoder.parameters()).device

    def make_train():
        # Same augmentations as online training, one full pass per view
        loader = DataLoader(data.trainset, batch_size=data.batch_size, shuffle=False,
                            num_workers=data.num_works, pin_memory=True)
        views = [extract(encoder, loader, device, 'train view {}/{}'.format(v + 1, args.cache_views))
                 for v in range(args.cache_views)]
        return {'feats': torch.stack([f for f, _ in views]), 'labels': views[0][1]}

    def make_val():
        feats, labels = extract(encoder, data.test, device, 'val')
        return {'feats': feats, 'labels': labels}

    train = _load_or_extract(_cache_path(args, data.train_txt, '_x{}'.format(args.cache_views)), make_train)
    val = _load_or_extract(_cache_path(args, data.val_txt), make_val)

    train_loader = DataLoader(CachedViews(train['feats'], train['labels']), batch_size=data.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(TensorDataset(val['feats'], val['labels']), batch_size=data.batch_size,
                            shuffle=False, num_workers=0)
    return train_loader, val_loader
