import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def list_images(root):
    root = Path(root)
    return sorted([p for p in root.iterdir() if p.suffix.lower() in IMG_EXTS])


def normalize_depth(depth):
    depth = depth.astype(np.float32)
    d_min, d_max = float(depth.min()), float(depth.max())
    depth = (depth - d_min) / (d_max - d_min + 1e-8)
    return (depth * 255.0).round().clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--da-root', default='/datasdb/ysh/lightningSR/da-v2')
    parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--device', default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    sys.path.insert(0, args.da_root)
    from inference import DepthEstimator

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = list_images(input_dir)
    if not paths:
        raise RuntimeError(f'No images found in {input_dir}')

    estimator = DepthEstimator(encoder=args.encoder, checkpoint_path=args.checkpoint, device=args.device)
    for path in tqdm(paths, desc='Depth Anything V2'):
        out_path = output_dir / (path.stem + '.png')
        if out_path.exists() and not args.overwrite:
            continue
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f'Failed to read image: {path}')
        depth = estimator.infer_image(image, input_size=args.input_size)
        cv2.imwrite(str(out_path), normalize_depth(depth))
    print(f'Done. depth maps: {output_dir}, count={len(list(output_dir.glob("*.png")))}')


if __name__ == '__main__':
    main()
