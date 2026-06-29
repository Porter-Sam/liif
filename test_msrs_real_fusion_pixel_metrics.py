import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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


def load_gray_pair(vi_path, ir_path, device):
    vi_img = Image.open(vi_path).convert('RGB')
    ir_img = Image.open(ir_path).convert('RGB')
    if ir_img.size != vi_img.size:
        ir_img = ir_img.resize(vi_img.size, Image.BICUBIC)

    vi = transforms.ToTensor()(vi_img)
    ir = transforms.ToTensor()(ir_img)
    weight = vi.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
    vi = (vi[:3] * weight).sum(dim=0, keepdim=True)
    ir = (ir[:3] * weight).sum(dim=0, keepdim=True)
    return vi.unsqueeze(0).to(device), ir.unsqueeze(0).to(device)


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


def ensure_dirs(root, save_maps):
    names = ['vi', 'ir', 'pred', 'comparison']
    if save_maps:
        names += ['d_vi', 'd_ir', 'r_vi', 'r_ir', 'w_vi', 'w_ir']
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
    return float(((q_a * w_a) + (q_b * w_b)).sum() / ((w_a + w_b).sum() + 1e-12))


def gaussian_kernel_torch(size=5, sigma=1.0, device='cpu'):
    ax = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ax, ax, indexing='ij')
    k = torch.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    k = k / k.sum()
    return k.view(1, 1, size, size)


def conv2_same_torch(x, kernel):
    pad = kernel.shape[-1] // 2
    x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    return F.conv2d(x, kernel)


def vifp_single(ref, dist, device='cpu'):
    ref = torch.from_numpy(ref.astype(np.float32)).view(1, 1, *ref.shape).to(device) * 255.0
    dist = torch.from_numpy(dist.astype(np.float32)).view(1, 1, *dist.shape).to(device) * 255.0
    kernel = gaussian_kernel_torch(5, 1.0, device=device)
    sigma_nsq = 2.0
    eps = 1e-10
    num = ref.new_tensor(0.0)
    den = ref.new_tensor(0.0)

    for scale in range(4):
        if scale > 0:
            ref = F.interpolate(ref, scale_factor=0.5, mode='bilinear', align_corners=False)
            dist = F.interpolate(dist, scale_factor=0.5, mode='bilinear', align_corners=False)
        mu1 = conv2_same_torch(ref, kernel)
        mu2 = conv2_same_torch(dist, kernel)
        sigma1_sq = conv2_same_torch(ref * ref, kernel) - mu1 * mu1
        sigma2_sq = conv2_same_torch(dist * dist, kernel) - mu2 * mu2
        sigma12 = conv2_same_torch(ref * dist, kernel) - mu1 * mu2
        sigma1_sq = torch.clamp(sigma1_sq, min=0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0)
        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12
        g = torch.where(sigma1_sq < eps, torch.zeros_like(g), g)
        sv_sq = torch.where(sigma1_sq < eps, sigma2_sq, sv_sq)
        sigma1_sq = torch.where(sigma1_sq < eps, torch.zeros_like(sigma1_sq), sigma1_sq)
        g = torch.clamp(g, min=0)
        sv_sq = torch.clamp(sv_sq, min=eps)
        num = num + torch.log10(1.0 + g * g * sigma1_sq / (sv_sq + sigma_nsq)).sum()
        den = den + torch.log10(1.0 + sigma1_sq / sigma_nsq).sum()
    return float((num / (den + eps)).detach().cpu().item())


def fusion_vif(vi, ir, fused, device='cpu'):
    return float(0.5 * (vifp_single(vi, fused, device=device) + vifp_single(ir, fused, device=device)))


def get_model_map(model, name, fallback):
    value = getattr(model, name, None)
    if value is None:
        return fallback
    return value.clamp(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='Path to epoch-last.pth or another checkpoint.')
    parser.add_argument('--data-root', default='E:/dataset/msrs/test',
                        help='Test set root. Put your test path here, e.g. E:/dataset/msrs/test.')
    parser.add_argument('--vi-dir', default='vi_color',
                        help='Visible image subfolder under --data-root.')
    parser.add_argument('--ir-dir', default='ir',
                        help='Infrared image subfolder under --data-root.')
    parser.add_argument('--out-dir', default='save/msrs_real_full_test_metrics',
                        help='Directory for predictions, comparison images, and metrics.')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--metrics-device', default='cpu', choices=['cpu', 'cuda'],
                        help='Device for VIF metric. Use cuda if CPU VIF is slow.')
    parser.add_argument('--save-maps', action='store_true',
                        help='Save distance/reliability/weight maps when the model exposes them.')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metrics_device = 'cuda' if args.metrics_device == 'cuda' and torch.cuda.is_available() else 'cpu'

    data_root = Path(args.data_root)
    pairs = find_pairs(data_root / args.vi_dir, data_root / args.ir_dir)
    if not pairs:
        raise RuntimeError(f'No paired images found under {data_root / args.vi_dir} and {data_root / args.ir_dir}')

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model = models.make(ckpt['model'], load_sd=True).to(device).eval()
    if not getattr(model, 'is_pixel_fusion', False):
        raise TypeError('Checkpoint is not a fusion-pixel-soft-distance model.')

    out_root = Path(args.out_dir)
    ensure_dirs(out_root, args.save_maps)

    rows = []
    for vi_path, ir_path in tqdm(pairs, desc='full-size test'):
        name = vi_path.stem + '.png'
        vi_t, ir_t = load_gray_pair(vi_path, ir_path, device)
        with torch.no_grad():
            pred_t = model(vi_t, ir_t).clamp(0, 1)

        vi_np = vi_t[0, 0].detach().cpu().clamp(0, 1).numpy()
        ir_np = ir_t[0, 0].detach().cpu().clamp(0, 1).numpy()
        pred_np = pred_t[0, 0].detach().cpu().clamp(0, 1).numpy()

        rows.append({
            'name': name,
            'height': pred_np.shape[0],
            'width': pred_np.shape[1],
            'en': entropy(pred_np),
            'sd': standard_deviation(pred_np),
            'vif': fusion_vif(vi_np, ir_np, pred_np, device=metrics_device),
            'qabf': qabf(vi_np, ir_np, pred_np),
        })

        d_vi = get_model_map(model, 'last_d_vi', pred_t)
        d_ir = get_model_map(model, 'last_d_ir', pred_t)
        r_vi = get_model_map(model, 'last_r_vi', d_vi)
        r_ir = get_model_map(model, 'last_r_ir', d_ir)
        w_vi = get_model_map(model, 'last_w_vi', d_vi)
        w_ir = get_model_map(model, 'last_w_ir', d_ir)

        tensor_to_pil(vi_t[0]).save(out_root / 'vi' / name)
        tensor_to_pil(ir_t[0]).save(out_root / 'ir' / name)
        tensor_to_pil(pred_t[0]).save(out_root / 'pred' / name)
        if args.save_maps:
            tensor_to_pil(d_vi[0]).save(out_root / 'd_vi' / name)
            tensor_to_pil(d_ir[0]).save(out_root / 'd_ir' / name)
            tensor_to_pil(r_vi[0]).save(out_root / 'r_vi' / name)
            tensor_to_pil(r_ir[0]).save(out_root / 'r_ir' / name)
            tensor_to_pil(w_vi[0]).save(out_root / 'w_vi' / name)
            tensor_to_pil(w_ir[0]).save(out_root / 'w_ir' / name)

        save_panel(out_root / 'comparison' / name, [
            ('VI(gray)', vi_t[0]),
            ('IR', ir_t[0]),
            ('w_vi', w_vi[0]),
            ('w_ir', w_ir[0]),
            ('Pred', pred_t[0]),
        ])

    csv_path = out_root / 'metrics.csv'
    fieldnames = ['name', 'height', 'width', 'en', 'sd', 'vif', 'qabf']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        'num_samples': len(rows),
        'data_root': str(data_root),
        'vi_dir': args.vi_dir,
        'ir_dir': args.ir_dir,
        'ckpt': args.ckpt,
        'mean': {k: float(np.mean([r[k] for r in rows])) for k in ['en', 'sd', 'vif', 'qabf']},
    }
    with open(out_root / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f'Saved {len(rows)} samples to {out_root}')
    print('Mean metrics:')
    for k, v in summary['mean'].items():
        print(f'  {k}: {v:.6f}')
    print(f'Metrics CSV: {csv_path}')
    print(f"Summary JSON: {out_root / 'summary.json'}")


if __name__ == '__main__':
    main()
