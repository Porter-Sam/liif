import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import to_pixel_samples


@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'gt_img': crop_hr
        }


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))


def to_gray_tensor(img):
    if img.shape[0] == 1:
        return img
    weight = img.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
    return (img[:3] * weight).sum(dim=0, keepdim=True)


def center_or_random_crop(img, size, random_crop=True):
    h, w = img.shape[-2:]
    if h < size or w < size:
        scale = size / min(h, w)
        img = resize_fn(img, (math.ceil(h * scale), math.ceil(w * scale)))
        h, w = img.shape[-2:]
    if random_crop:
        x0 = random.randint(0, h - size)
        y0 = random.randint(0, w - size)
    else:
        x0 = (h - size) // 2
        y0 = (w - size) // 2
    return img[:, x0:x0 + size, y0:y0 + size]


def resize_short_side_at_least(img, size):
    h, w = img.shape[-2:]
    if h >= size and w >= size:
        return img
    scale = size / min(h, w)
    return resize_fn(img, (math.ceil(h * scale), math.ceil(w * scale)))


def crop_with_params(img, x0, y0, size):
    return img[:, x0:x0 + size, y0:y0 + size]


@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'gt_img': crop_hr
        }


@register('fusion-implicit-paired')
class FusionImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=128, out_size=256,
                 augment=False, sample_q=None, target_mode='avg'):
        self.dataset = dataset
        self.inp_size = inp_size
        self.out_size = out_size
        self.augment = augment
        self.sample_q = sample_q
        self.target_mode = target_mode

    def __len__(self):
        return len(self.dataset)

    def _make_target(self, vi, ir):
        if self.target_mode == 'vi':
            return vi
        if self.target_mode == 'ir':
            return ir
        if self.target_mode == 'max':
            return torch.maximum(vi, ir)
        if self.target_mode == 'avg':
            return 0.5 * (vi + ir)
        raise ValueError('Unknown target_mode: {}'.format(self.target_mode))

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if not isinstance(item, (tuple, list)):
            raise TypeError('fusion-implicit-paired expects paired or triple image dataset.')
        if len(item) == 2:
            vi, ir = item
            gt = None
        elif len(item) == 3:
            vi, ir, gt = item
        else:
            raise ValueError('Expected dataset item length 2 or 3, got {}'.format(len(item)))

        vi = to_gray_tensor(vi)
        ir = to_gray_tensor(ir)
        gt = to_gray_tensor(gt) if gt is not None else None

        vi = resize_short_side_at_least(vi, self.out_size)
        ir = resize_short_side_at_least(ir, self.out_size)
        if gt is not None:
            gt = resize_short_side_at_least(gt, self.out_size)

        h = min(vi.shape[-2], ir.shape[-2], gt.shape[-2] if gt is not None else vi.shape[-2])
        w = min(vi.shape[-1], ir.shape[-1], gt.shape[-1] if gt is not None else vi.shape[-1])
        if self.augment:
            x0 = random.randint(0, h - self.out_size)
            y0 = random.randint(0, w - self.out_size)
        else:
            x0 = (h - self.out_size) // 2
            y0 = (w - self.out_size) // 2

        vi_hr = crop_with_params(vi, x0, y0, self.out_size)
        ir_hr = crop_with_params(ir, x0, y0, self.out_size)
        if gt is not None:
            gt_hr = crop_with_params(gt, x0, y0, self.out_size)
        else:
            gt_hr = self._make_target(vi_hr, ir_hr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            vi_hr = augment(vi_hr)
            ir_hr = augment(ir_hr)
            gt_hr = augment(gt_hr)

        vi_lr = resize_fn(vi_hr, self.inp_size)
        ir_lr = resize_fn(ir_hr, self.inp_size)
        hr_coord, hr_rgb = to_pixel_samples(gt_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / gt_hr.shape[-2]
        cell[:, 1] *= 2 / gt_hr.shape[-1]

        return {
            'inp': vi_lr,
            'vi': vi_lr,
            'ir': ir_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'gt_img': gt_hr,
        }


@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min, size_max=None,
                 augment=False, gt_resize=None, sample_q=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        p = idx / (len(self.dataset) - 1)
        w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
        img_hr = resize_fn(img_hr, w_hr)

        if self.augment:
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)

        if self.gt_resize is not None:
            img_hr = resize_fn(img_hr, self.gt_resize)

        hr_coord, hr_rgb = to_pixel_samples(img_hr)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / img_hr.shape[-2]
        cell[:, 1] *= 2 / img_hr.shape[-1]

        return {
            'inp': img_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'gt_img': img_hr
        }
