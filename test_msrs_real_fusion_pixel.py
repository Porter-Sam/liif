import argparse
import os
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data-root', default='E:/dataset/msrs/test')
    parser.add_argument('--vi-dir', default='vi_color')
    parser.add_argument('--ir-dir', default='ir')
    parser.add_argument('--out-dir', default='save/msrs_real_test_fusion_pixel_softdist')
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

    max_sum_err = 0.0
    for vi_path, ir_path in tqdm(pairs, desc='test'):
        name = vi_path.stem + '.png'
        vi = to_gray_tensor(vi_path, args.image_size).unsqueeze(0).to(device)
        ir = to_gray_tensor(ir_path, args.image_size).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(vi, ir).clamp(0, 1)
        d_vi = model.last_d_vi.clamp(0, 1)
        d_ir = model.last_d_ir.clamp(0, 1)
        max_sum_err = max(max_sum_err, float((d_vi + d_ir - 1.0).abs().max().item()))

        tensor_to_pil(vi[0]).save(out_root / 'vi' / name)
        tensor_to_pil(ir[0]).save(out_root / 'ir' / name)
        tensor_to_pil(d_vi[0]).save(out_root / 'd_vi' / name)
        tensor_to_pil(d_ir[0]).save(out_root / 'd_ir' / name)
        tensor_to_pil(pred[0]).save(out_root / 'pred' / name)
        save_panel(out_root / 'comparison' / name, [
            ('VI(gray)', vi[0]),
            ('IR', ir[0]),
            ('d_vi', d_vi[0]),
            ('d_ir', d_ir[0]),
            ('Pred', pred[0]),
        ])

    print(f'Saved {len(pairs)} samples to {out_root}')
    print(f'max |d_vi + d_ir - 1| = {max_sum_err:.8f}')


if __name__ == '__main__':
    main()
