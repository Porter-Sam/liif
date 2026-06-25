import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from skimage.filters import sobel
from skimage.transform import resize
from tqdm import tqdm

import models
import utils


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def read_rgb(path):
    return Image.open(path).convert('RGB')


def to_gray_np(img):
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        return arr
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def center_crop_square(img):
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def center_crop_resize(img, size):
    return center_crop_square(img).resize((size, size), Image.BICUBIC)


def gray_to_tensor(gray):
    return torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0)


def list_msrs_pairs(root, split='val'):
    root = Path(root)
    vi_dir = root / split / 'vi_color'
    ir_dir = root / split / 'ir'
    names = sorted(
        p.name for p in vi_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and (ir_dir / p.name).exists()
    )
    return [(name, vi_dir / name, ir_dir / name) for name in names]


def sample_evenly(items, count):
    if count >= len(items):
        return items
    idx = np.linspace(0, len(items) - 1, count, dtype=np.int64)
    return [items[int(i)] for i in idx]


def entropy(img, bins=256):
    hist, _ = np.histogram(np.clip(img, 0, 1), bins=bins, range=(0, 1), density=False)
    p = hist.astype(np.float64)
    p = p / max(p.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def standard_deviation(img):
    return float(np.std(img))


def qabf(src_a, src_b, fused):
    def grad_angle(x):
        gy, gx = np.gradient(x.astype(np.float32))
        g = np.sqrt(gx * gx + gy * gy)
        a = np.arctan2(gy, gx)
        return g, a

    def edge_preservation(src, fus):
        g_s, a_s = grad_angle(src)
        g_f, a_f = grad_angle(fus)
        eps = 1e-12
        ratio = np.minimum(g_s, g_f) / (np.maximum(g_s, g_f) + eps)
        angle = 1.0 - np.abs(a_s - a_f) / (math.pi / 2)
        angle = np.clip(angle, 0.0, 1.0)
        qg = 1.0 / (1.0 + np.exp(-10.0 * (ratio - 0.5)))
        qa = 1.0 / (1.0 + np.exp(-10.0 * (angle - 0.5)))
        return qg * qa, g_s

    q_a, w_a = edge_preservation(src_a, fused)
    q_b, w_b = edge_preservation(src_b, fused)
    return float(((q_a * w_a) + (q_b * w_b)).sum() / (w_a + w_b + 1e-12).sum())


def gaussian_kernel(size=5, sigma=1.0):
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    k /= k.sum()
    return k


def conv2_same(x, kernel):
    pad = kernel.shape[0] // 2
    xp = np.pad(x, pad, mode='reflect')
    out = np.zeros_like(x, dtype=np.float32)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(xp[i:i + kernel.shape[0], j:j + kernel.shape[1]] * kernel)
    return out


def vifp_single(ref, dist):
    ref = ref.astype(np.float32)
    dist = dist.astype(np.float32)
    sigma_nsq = 2.0
    eps = 1e-10
    num = 0.0
    den = 0.0
    for scale in range(4):
        if scale > 0:
            ref = resize(ref, (ref.shape[0] // 2, ref.shape[1] // 2), order=1,
                         mode='reflect', anti_aliasing=True).astype(np.float32)
            dist = resize(dist, (dist.shape[0] // 2, dist.shape[1] // 2), order=1,
                          mode='reflect', anti_aliasing=True).astype(np.float32)
        kernel = gaussian_kernel(5, 1.0)
        mu1 = conv2_same(ref, kernel)
        mu2 = conv2_same(dist, kernel)
        sigma1_sq = conv2_same(ref * ref, kernel) - mu1 * mu1
        sigma2_sq = conv2_same(dist * dist, kernel) - mu2 * mu2
        sigma12 = conv2_same(ref * dist, kernel) - mu1 * mu2
        sigma1_sq = np.maximum(sigma1_sq, 0)
        sigma2_sq = np.maximum(sigma2_sq, 0)
        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12
        g = np.where(sigma1_sq < eps, 0, g)
        sv_sq = np.where(sigma1_sq < eps, sigma2_sq, sv_sq)
        sigma1_sq = np.where(sigma1_sq < eps, 0, sigma1_sq)
        g = np.maximum(g, 0)
        sv_sq = np.maximum(sv_sq, eps)
        num += np.sum(np.log10(1.0 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1.0 + sigma1_sq / sigma_nsq))
    return float(num / (den + eps))


def fusion_vif(vi, ir, fused):
    return float(0.5 * (vifp_single(vi, fused) + vifp_single(ir, fused)))


def tensor_to_gray_image(pred, size):
    pred = pred.view(1, size, size, 1).permute(0, 3, 1, 2)[0, 0]
    return pred.detach().cpu().clamp(0, 1).numpy()


def predict(model, vi, ir, size=256, inp_size=128, bsize=65536):
    vi_lr = resize(vi, (inp_size, inp_size), order=3, mode='reflect',
                   anti_aliasing=True).astype(np.float32)
    ir_lr = resize(ir, (inp_size, inp_size), order=3, mode='reflect',
                   anti_aliasing=True).astype(np.float32)
    vi_t = gray_to_tensor(vi_lr).cuda()
    ir_t = gray_to_tensor(ir_lr).cuda()
    coord = utils.make_coord((size, size)).cuda().unsqueeze(0)
    cell = torch.ones_like(coord)
    cell[:, :, 0] *= 2 / size
    cell[:, :, 1] *= 2 / size
    preds = []
    with torch.no_grad():
        model.gen_feat((vi_t - 0.5) / 0.5, (ir_t - 0.5) / 0.5)
        ql = 0
        while ql < coord.shape[1]:
            qr = min(ql + bsize, coord.shape[1])
            pred = model.query_rgb(coord[:, ql:qr], cell[:, ql:qr])
            pred = pred * 0.5 + 0.5
            preds.append(pred)
            ql = qr
    return tensor_to_gray_image(torch.cat(preds, dim=1), size)


def save_gray(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr, mode='L').save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--msrs-root', required=True)
    parser.add_argument('--split', default='val')
    parser.add_argument('--num', type=int, default=100)
    parser.add_argument('--out-dir', default='eval_msrs100')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--inp-size', type=int, default=128)
    parser.add_argument('--eval-bsize', type=int, default=65536)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    ckpt = torch.load(args.model, map_location='cpu')
    model = models.make(ckpt['model'], load_sd=True).cuda()
    model.eval()

    out_dir = Path(args.out_dir)
    pred_dir = out_dir / 'pred'
    rows = []
    pairs = sample_evenly(list_msrs_pairs(args.msrs_root, args.split), args.num)
    for name, vi_path, ir_path in tqdm(pairs, desc='eval'):
        vi = to_gray_np(center_crop_resize(read_rgb(vi_path), args.size))
        ir = to_gray_np(center_crop_resize(read_rgb(ir_path), args.size))
        fused = predict(model, vi, ir, args.size, args.inp_size, args.eval_bsize)
        save_gray(pred_dir / name, fused)
        rows.append({
            'name': name,
            'en': entropy(fused),
            'sd': standard_deviation(fused),
            'qabf': qabf(vi, ir, fused),
            'vif': fusion_vif(vi, ir, fused),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'metrics.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'en', 'sd', 'qabf', 'vif'])
        writer.writeheader()
        writer.writerows(rows)
    means = {k: float(np.mean([r[k] for r in rows])) for k in ['en', 'sd', 'qabf', 'vif']}
    with open(out_dir / 'summary.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump({'num_samples': len(rows), **means}, f, sort_keys=False)
    print('summary:', means)
    print('saved:', out_dir)


if __name__ == '__main__':
    main()
