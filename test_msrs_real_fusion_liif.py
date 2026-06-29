import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from skimage.transform import resize
from tqdm import tqdm

import models
import utils


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def read_gray(path):
    img = Image.open(path)
    if img.mode != 'L':
        img = img.convert('RGB')
        arr = np.asarray(img).astype(np.float32) / 255.0
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        return gray
    return np.asarray(img).astype(np.float32) / 255.0


def center_crop_square_np(img):
    h, w = img.shape[:2]
    side = min(h, w)
    y = (h - side) // 2
    x = (w - side) // 2
    return img[y:y + side, x:x + side]


def prepare_image(path, size):
    gray = read_gray(path)
    gray = center_crop_square_np(gray)
    gray = resize(gray, (size, size), order=3, mode='reflect', anti_aliasing=True)
    return gray.astype(np.float32)


def list_pairs(msrs_test_root):
    root = Path(msrs_test_root)
    vi_dir = root / 'vi_color'
    ir_dir = root / 'ir'
    if not vi_dir.is_dir():
        raise FileNotFoundError(f'Visible directory not found: {vi_dir}')
    if not ir_dir.is_dir():
        raise FileNotFoundError(f'IR directory not found: {ir_dir}')

    names = []
    for p in sorted(vi_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        if (ir_dir / p.name).is_file():
            names.append(p.name)
    return [(name, vi_dir / name, ir_dir / name) for name in names]


def gray_to_tensor(gray):
    return torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0)


def save_gray(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr, mode='L').save(path)


def make_panel(img, label):
    arr = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    pil = Image.fromarray(arr, mode='L').convert('RGB')
    label_h = 18
    canvas = Image.new('RGB', (pil.width, pil.height + label_h), color=(255, 255, 255))
    canvas.paste(pil, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2), label, fill=(0, 0, 0))
    return canvas


def save_comparison(path, vi, ir, dist, pred):
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        make_panel(vi, 'VI(gray)'),
        make_panel(ir, 'IR'),
        make_panel(dist, 'Distance'),
        make_panel(pred, 'Pred'),
    ]
    width = sum(p.width for p in panels)
    height = max(p.height for p in panels)
    canvas = Image.new('RGB', (width, height), color=(255, 255, 255))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    canvas.save(path)


def normalize_map(x, eps=1e-6):
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    return (x - x_min) / (x_max - x_min + eps)


def predict_one(model, vi, ir, size, inp_size, bsize, device):
    vi_lr = resize(vi, (inp_size, inp_size), order=3, mode='reflect',
                   anti_aliasing=True).astype(np.float32)
    ir_lr = resize(ir, (inp_size, inp_size), order=3, mode='reflect',
                   anti_aliasing=True).astype(np.float32)
    vi_t = gray_to_tensor(vi_lr).to(device)
    ir_t = gray_to_tensor(ir_lr).to(device)

    coord = utils.make_coord((size, size)).to(device).unsqueeze(0)
    cell = torch.ones_like(coord)
    cell[:, :, 0] *= 2 / size
    cell[:, :, 1] *= 2 / size

    preds = []
    distances = []
    weights = []
    with torch.no_grad():
        model.gen_feat((vi_t - 0.5) / 0.5, (ir_t - 0.5) / 0.5)
        ql = 0
        while ql < coord.shape[1]:
            qr = min(ql + bsize, coord.shape[1])
            pred = model.query_rgb(coord[:, ql:qr], cell[:, ql:qr])
            preds.append(pred)
            distances.append(model.last_modality_distance)
            weights.append(model.last_modality_weight)
            ql = qr

    pred = torch.cat(preds, dim=1) * 0.5 + 0.5
    pred = pred.view(1, size, size, 1).permute(0, 3, 1, 2)[0, 0]
    pred_np = pred.detach().cpu().clamp(0, 1).numpy()

    dist = torch.cat(distances, dim=1)
    # last_modality_distance is [d_vi, d_ir]. Positive means IR is farther than VI.
    dist_diff = (dist[..., 1] - dist[..., 0]).view(1, 1, size, size)[0, 0]
    dist_np = normalize_map(dist_diff.detach().cpu().numpy())

    weight = torch.cat(weights, dim=1)
    w_vi = weight[..., 0].view(1, 1, size, size)[0, 0]
    w_ir = weight[..., 1].view(1, 1, size, size)[0, 0]
    w_vi_np = w_vi.detach().cpu().clamp(0, 1).numpy()
    w_ir_np = w_ir.detach().cpu().clamp(0, 1).numpy()
    return pred_np, dist_np, w_vi_np, w_ir_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='save/msrs_synth_fusion_liif_gray_x2_e100_from_scratch/epoch-last.pth')
    parser.add_argument('--msrs-test-root', default='E:/dataset/msrs/test')
    parser.add_argument('--out-dir', default='save/msrs_real_test_fusion_liif_gray_x2_epoch_last')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--inp-size', type=int, default=128)
    parser.add_argument('--eval-bsize', type=int, default=65536)
    parser.add_argument('--temperature', type=float, default=None,
                        help='Override model.temperature at test time.')
    parser.add_argument('--sample-mode', default=None, choices=['nearest', 'bilinear'],
                        help='Override FusionLIIF grid_sample mode at test time.')
    args = parser.parse_args()

    if torch.cuda.is_available() and args.gpu.lower() != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model = models.make(ckpt['model'], load_sd=True).to(device)
    if args.temperature is not None:
        if not hasattr(model, 'temperature'):
            raise AttributeError('Loaded model has no temperature attribute.')
        model.temperature = float(args.temperature)
        print(f'Override model.temperature = {model.temperature}')
    if args.sample_mode is not None:
        if not hasattr(model, 'sample_mode'):
            raise AttributeError('Loaded model has no sample_mode attribute.')
        model.sample_mode = args.sample_mode
        print(f'Override model.sample_mode = {model.sample_mode}')
    model.eval()

    out_dir = Path(args.out_dir)
    pairs = list_pairs(args.msrs_test_root)
    if not pairs:
        raise RuntimeError(f'No paired MSRS test images found under {args.msrs_test_root}')

    for name, vi_path, ir_path in tqdm(pairs, desc='MSRS real test'):
        vi = prepare_image(vi_path, args.size)
        ir = prepare_image(ir_path, args.size)
        pred, dist, w_vi, w_ir = predict_one(
            model, vi, ir, args.size, args.inp_size, args.eval_bsize, device)

        save_gray(out_dir / 'vi' / name, vi)
        save_gray(out_dir / 'ir' / name, ir)
        save_gray(out_dir / 'distance' / name, dist)
        save_gray(out_dir / 'w_vi' / name, w_vi)
        save_gray(out_dir / 'w_ir' / name, w_ir)
        save_gray(out_dir / 'pred' / name, pred)
        save_comparison(out_dir / 'comparison' / name, vi, ir, dist, pred)

    print(f'Saved {len(pairs)} samples to: {out_dir}')


if __name__ == '__main__':
    main()
