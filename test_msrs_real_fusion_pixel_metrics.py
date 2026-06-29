import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from tqdm import tqdm

import models


IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def list_images(root):
    root = Path(root)
    return sorted([p for p in root.iterdir() if p.suffix.lower() in IMG_EXTS])


def find_pairs(vi_dir, ir_dir):
    vi_files = list_images(vi_dir)
    ir_map = {p.stem: p for p in list_images(ir_dir)}
    pairs = []
    for vi_path in vi_files:
        ir_path = ir_map.get(vi_path.stem)
        if ir_path is not None:
            pairs.append((vi_path, ir_path))
    return pairs


def to_gray_tensor(path, image_size):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((image_size, image_size), Image.BICUBIC)
    x = transforms.ToTensor()(img)
    if x.shape[0] == 1:
        return x
    weight = x.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
    return (x[:3] * weight).sum(dim=0, keepdim=True)


def tensor_to_pil(x):
    x = x.detach().float().cpu().clamp(0, 1)
    if x.dim() == 3 and x.shape[0] == 1:
        x = x.expand(3, -1, -1)
    if x.dim() == 3:
        x = x.permute(1, 2, 0)
    arr = (x.numpy() * 255.0).round().astype('uint8')
    return Image.fromarray(arr)


def save_panel(path, panels):
    pil_panels = []
    label_h = 18
    for label, tensor in panels:
        pil = tensor_to_pil(tensor).convert('RGB')
        canvas = Image.new('RGB', (pil.width, pil.height + label_h), 'white')
        canvas.paste(pil, (0, label_h))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 2), label, fill=(0, 0, 0))
        pil_panels.append(canvas)

    width = sum(p.width for p in pil_panels)
    height = max(p.height for p in pil_panels)
    out = Image.new('RGB', (width, height), 'white')
    x = 0
    for panel in pil_panels:
        out.paste(panel, (x, 0))
        x += panel.width
    out.save(path)


def ensure_dirs(root):
    names = ['vi', 'ir', 'd_vi', 'd_ir', 'pred', 'comparison']
    for name in names:
        (root / name).mkdir(parents=True, exist_ok=True)


# -------------------- fusion metrics --------------------

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
    # VIF is defined for 8-bit dynamic range; scale [0,1] inputs to [0,255]
    ref = (ref.astype(np.float32) * 255.0)
    dist = (dist.astype(np.float32) * 255.0)
    sigma_nsq = 2.0
    eps = 1e-10
    num = 0.0
    den = 0.0
    for scale in range(4):
        if scale > 0:
            ref = _resize_half(ref)
            dist = _resize_half(dist)
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


def _resize_half(img):
    from skimage.transform import resize
    return resize(img, (img.shape[0] // 2, img.shape[1] // 2), order=1,
                  mode='reflect', anti_aliasing=True).astype(np.float32)


def fusion_vif(vi, ir, fused):
    return float(0.5 * (vifp_single(vi, fused) + vifp_single(ir, fused)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data-root', default='/datasdb/wl/dataset/msrs/test')
    parser.add_argument('--vi-dir', default='vi_color')
    parser.add_argument('--ir-dir', default='ir')
    parser.add_argument('--out-dir', default='save/msrs_real_test_fusion_pixel_softdist_metrics')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_root = Path(args.data_root)
    pairs = find_pairs(data_root / args.vi_dir, data_root / args.ir_dir)
    if not pairs:
        raise RuntimeError(f'No paired images found under {data_root}')

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model = models.make(ckpt['model'], load_sd=True).to(device).eval()
    if not getattr(model, 'is_pixel_fusion', False):
        raise TypeError('Checkpoint is not a fusion-pixel-soft-distance model.')

    out_root = Path(args.out_dir)
    ensure_dirs(out_root)

    rows = []
    max_sum_err = 0.0
    for vi_path, ir_path in tqdm(pairs, desc='test'):
        name = vi_path.stem + '.png'
        vi_t = to_gray_tensor(vi_path, args.image_size).unsqueeze(0).to(device)
        ir_t = to_gray_tensor(ir_path, args.image_size).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_t = model(vi_t, ir_t).clamp(0, 1)
        d_vi = model.last_d_vi.clamp(0, 1)
        d_ir = model.last_d_ir.clamp(0, 1)
        max_sum_err = max(max_sum_err, float((d_vi + d_ir - 1.0).abs().max().item()))

        vi_np = vi_t[0, 0].detach().cpu().clamp(0, 1).numpy()
        ir_np = ir_t[0, 0].detach().cpu().clamp(0, 1).numpy()
        pred_np = pred_t[0, 0].detach().cpu().clamp(0, 1).numpy()

        rows.append({
            'name': name,
            'en': entropy(pred_np),
            'sd': standard_deviation(pred_np),
            'qabf': qabf(vi_np, ir_np, pred_np),
            'vif': fusion_vif(vi_np, ir_np, pred_np),
        })

        tensor_to_pil(vi_t[0]).save(out_root / 'vi' / name)
        tensor_to_pil(ir_t[0]).save(out_root / 'ir' / name)
        tensor_to_pil(d_vi[0]).save(out_root / 'd_vi' / name)
        tensor_to_pil(d_ir[0]).save(out_root / 'd_ir' / name)
        tensor_to_pil(pred_t[0]).save(out_root / 'pred' / name)
        save_panel(out_root / 'comparison' / name, [
            ('VI(gray)', vi_t[0]),
            ('IR', ir_t[0]),
            ('d_vi', d_vi[0]),
            ('d_ir', d_ir[0]),
            ('Pred', pred_t[0]),
        ])

    # save per-image metrics
    csv_path = out_root / 'metrics.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'en', 'sd', 'qabf', 'vif'])
        writer.writeheader()
        writer.writerows(rows)

    # save summary
    means = {k: float(np.mean([r[k] for r in rows])) for k in ['en', 'sd', 'qabf', 'vif']}
    summary_path = out_root / 'summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f'num_samples: {len(rows)}\n')
        for k, v in means.items():
            f.write(f'{k}: {v:.6f}\n')
        f.write(f'max_sum_err: {max_sum_err:.8f}\n')

    print(f'Saved {len(pairs)} samples to {out_root}')
    print(f'mean metrics: {means}')
    print(f'max |d_vi + d_ir - 1| = {max_sum_err:.8f}')


if __name__ == '__main__':
    main()
