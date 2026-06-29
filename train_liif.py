""" Train for generating LIIF, from image to implicit representation.

    Config:
        train_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        val_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        (data_norm):
            inp: {sub: []; div: []}
            gt: {sub: []; div: []}
        (eval_type):
        (eval_bsize):

        model: $spec
        optimizer: $spec
        epoch_max:
        (multi_step_lr):
            milestones: []; gamma: 0.5
        (resume): *.pth

        (epoch_val): ; (epoch_save):
"""

import argparse
import os
from datetime import datetime

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import datasets
import models
import utils
from test import eval_psnr


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers=8, pin_memory=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training():
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        if config.get('resume_model_only', False):
            optimizer = utils.make_optimizer(
                model.parameters(), config['optimizer'])
        else:
            optimizer = utils.make_optimizer(
                model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def is_fusion_model(model):
    model_ = model.module if isinstance(model, nn.DataParallel) else model
    return getattr(model_, 'is_fusion_liif', False)


def is_pixel_fusion_model(model):
    model_ = model.module if isinstance(model, nn.DataParallel) else model
    return getattr(model_, 'is_pixel_fusion', False)


def to_gray_image(x):
    if x.shape[1] == 1:
        return x
    weight = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (x[:, :3] * weight).sum(dim=1, keepdim=True)


def to_gray_query(x):
    if x.shape[-1] == 1:
        return x
    weight = x.new_tensor([0.299, 0.587, 0.114]).view(1, 1, 3)
    return (x[..., :3] * weight).sum(dim=-1, keepdim=True)


def get_first(batch, names):
    for name in names:
        if name in batch:
            return batch[name]
    return None


def make_fusion_inputs(batch, inp_sub, inp_div, gt_sub, gt_div):
    vi = get_first(batch, ['vi', 'vis', 'img_vi', 'img_vis'])
    ir = get_first(batch, ['ir', 'img_ir'])
    if vi is None:
        vi = batch['inp']
    if ir is None:
        ir = vi

    vi = (to_gray_image(vi) - inp_sub) / inp_div
    ir = (to_gray_image(ir) - inp_sub) / inp_div
    gt = (to_gray_query(batch['gt']) - gt_sub) / gt_div
    return vi, ir, gt


def make_pixel_fusion_inputs(batch):
    vi = get_first(batch, ['vi', 'vis', 'img_vi', 'img_vis'])
    ir = get_first(batch, ['ir', 'img_ir'])
    if vi is None:
        vi = batch['inp']
    if ir is None:
        ir = vi
    gt = batch['gt_img']
    return to_gray_image(vi).clamp(0, 1), to_gray_image(ir).clamp(0, 1), to_gray_image(gt).clamp(0, 1)


def image_grad(x):
    grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
    grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
    grad_x = F.pad(grad_x.abs(), (0, 1, 0, 0))
    grad_y = F.pad(grad_y.abs(), (0, 0, 0, 1))
    return grad_x + grad_y


def fusion_loss(pred, vi, ir, int_weight=1.0, grad_weight=1.0):
    target_int = torch.maximum(vi, ir)
    target_grad = torch.maximum(image_grad(vi), image_grad(ir))
    loss_int = F.l1_loss(pred, target_int)
    loss_grad = F.l1_loss(image_grad(pred), target_grad)
    return int_weight * loss_int + grad_weight * loss_grad


def distance_tv_loss(d):
    loss_h = (d[:, :, 1:, :] - d[:, :, :-1, :]).abs().mean()
    loss_w = (d[:, :, :, 1:] - d[:, :, :, :-1]).abs().mean()
    return loss_h + loss_w


def distance_soft_loss(d, eps):
    return (F.relu(eps - d) + F.relu(d - (1.0 - eps))).mean()


def distance_entropy(d, eps=1e-6):
    d = d.clamp(eps, 1.0 - eps)
    return -(d * torch.log(d) + (1.0 - d) * torch.log(1.0 - d)).mean()


def glare_reliability_loss(w_vi, vi, ir, threshold=0.85, delta=0.12,
                           sharpness=20.0, ir_info_threshold=0.04):
    glare = (
        torch.sigmoid(sharpness * (vi - threshold))
        * torch.sigmoid(sharpness * (vi - ir - delta))
    )
    ir_info = torch.sigmoid(sharpness * ((ir + image_grad(ir)) - ir_info_threshold))
    return (glare * ir_info * w_vi).mean()


def apply_pixel_runtime_config(model):
    model_ = model.module if isinstance(model, nn.DataParallel) else model
    if not getattr(model_, 'is_pixel_fusion', False):
        return
    model_args = config.get('model', {}).get('args', {})
    if 'temperature' in model_args:
        model_.temperature = float(model_args['temperature'])
    if 'residual_scale' in model_args:
        model_.residual_scale = float(model_args['residual_scale'])
    if 'reliability_temperature' in model_args:
        model_.reliability_temperature = float(model_args['reliability_temperature'])
    if 'glare_threshold' in model_args:
        model_.glare_threshold = float(model_args['glare_threshold'])
    if 'glare_delta' in model_args:
        model_.glare_delta = float(model_args['glare_delta'])
    if 'glare_sharpness' in model_args:
        model_.glare_sharpness = float(model_args['glare_sharpness'])
    if 'use_reliability' in model_args:
        model_.use_reliability = bool(model_args['use_reliability'])


def _tensor_to_pil(img):
    img = img.detach().float().cpu().clamp(0, 1)
    if img.dim() == 3:
        if img.shape[0] == 1:
            img = img.expand(3, -1, -1)
        img = img.permute(1, 2, 0)
    arr = (img.numpy() * 255.0).round().astype('uint8')
    return Image.fromarray(arr)


def _panel(img, label):
    pil = _tensor_to_pil(img)
    label_h = 18
    canvas = Image.new('RGB', (pil.width, pil.height + label_h), color=(255, 255, 255))
    canvas.paste(pil.convert('RGB'), (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2), label, fill=(0, 0, 0))
    return canvas


def _save_visual_row(path, panels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pil_panels = [_panel(img, label) for label, img in panels]
    width = sum(p.width for p in pil_panels)
    height = max(p.height for p in pil_panels)
    canvas = Image.new('RGB', (width, height), color=(255, 255, 255))
    x = 0
    for panel in pil_panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    canvas.save(path)


def _make_visual_row(panels):
    pil_panels = [_panel(img, label) for label, img in panels]
    width = sum(p.width for p in pil_panels)
    height = max(p.height for p in pil_panels)
    canvas = Image.new('RGB', (width, height), color=(255, 255, 255))
    x = 0
    for panel in pil_panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    return canvas


def _save_visual_grid(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row_imgs = [_make_visual_row(row) for row in rows]
    width = max(row.width for row in row_imgs)
    height = sum(row.height for row in row_imgs)
    canvas = Image.new('RGB', (width, height), color=(255, 255, 255))
    y = 0
    for row in row_imgs:
        canvas.paste(row, (0, y))
        y += row.height
    canvas.save(path)


def save_fusion_visualization(loader, model, save_path, epoch, bsize=65536):
    if loader is None or (not is_fusion_model(model) and not is_pixel_fusion_model(model)):
        return

    model.eval()
    if is_pixel_fusion_model(model):
        max_images = config.get('vis_num', 10)
        saved = 0
        rows = []
        model_ = model.module if isinstance(model, nn.DataParallel) else model
        with torch.no_grad():
            for batch in loader:
                for k, v in batch.items():
                    batch[k] = v.cuda()
                vi, ir, gt_img = make_pixel_fusion_inputs(batch)
                pred_img = model(vi, ir).clamp(0, 1)
                d_vi = model_.last_d_vi.clamp(0, 1)
                d_ir = model_.last_d_ir.clamp(0, 1)
                r_vi = model_.last_r_vi.clamp(0, 1)
                r_ir = model_.last_r_ir.clamp(0, 1)
                w_vi = model_.last_w_vi.clamp(0, 1)
                w_ir = model_.last_w_ir.clamp(0, 1)

                for i in range(vi.shape[0]):
                    if saved >= max_images:
                        break
                    rows.append([
                        ('VI', vi[i]),
                        ('IR', ir[i]),
                        ('d_vi', d_vi[i]),
                        ('d_ir', d_ir[i]),
                        ('r_vi', r_vi[i]),
                        ('r_ir', r_ir[i]),
                        ('w_vi', w_vi[i]),
                        ('w_ir', w_ir[i]),
                        ('Pred', pred_img[i]),
                        ('GT', gt_img[i]),
                    ])
                    saved += 1
                if saved >= max_images:
                    break
        out_path = os.path.join(save_path, 'visual', 'epoch-{:06d}.png'.format(epoch))
        _save_visual_grid(out_path, rows)
        log('visual saved: {} images in {}'.format(saved, out_path))
        return

    data_norm = config['data_norm']
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub_img = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    gt_div_img = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()

    batch = next(iter(loader))
    for k, v in batch.items():
        batch[k] = v.cuda()

    vi_raw = get_first(batch, ['vi', 'vis', 'img_vi', 'img_vis'])
    ir_raw = get_first(batch, ['ir', 'img_ir'])
    if vi_raw is None:
        vi_raw = batch['inp']
    if ir_raw is None:
        ir_raw = vi_raw

    vi_show = to_gray_image(vi_raw[:1]).clamp(0, 1)
    ir_show = to_gray_image(ir_raw[:1]).clamp(0, 1)
    vi = (vi_show - inp_sub) / inp_div
    ir = (ir_show - inp_sub) / inp_div

    if 'gt_img' in batch:
        gt_img = to_gray_image(batch['gt_img'][:1]).clamp(0, 1)
    else:
        h = round(batch['coord'].shape[1] ** 0.5)
        gt_img = to_gray_query(batch['gt'][:1]).view(1, h, h, 1).permute(0, 3, 1, 2).clamp(0, 1)

    h, w = gt_img.shape[-2:]
    coord = utils.make_coord((h, w)).cuda().unsqueeze(0)
    cell = torch.ones_like(coord)
    cell[:, :, 0] *= 2 / h
    cell[:, :, 1] *= 2 / w

    preds = []
    distances = []
    with torch.no_grad():
        model.gen_feat(vi, ir)
        ql = 0
        while ql < coord.shape[1]:
            qr = min(ql + bsize, coord.shape[1])
            pred = model.query_rgb(coord[:, ql:qr], cell[:, ql:qr])
            preds.append(pred)
            distances.append(model.last_modality_distance)
            ql = qr

    pred = torch.cat(preds, dim=1)
    pred = pred * gt_div_img.view(1, 1, -1) + gt_sub_img.view(1, 1, -1)
    pred_img = pred.view(1, h, w, -1).permute(0, 3, 1, 2).clamp(0, 1)

    dist = torch.cat(distances, dim=1)
    dist_map = (dist[..., 1] - dist[..., 0]).view(1, 1, h, w)
    dist_min = dist_map.amin(dim=(-2, -1), keepdim=True)
    dist_max = dist_map.amax(dim=(-2, -1), keepdim=True)
    dist_map = (dist_map - dist_min) / (dist_max - dist_min + 1e-6)

    vi_show = F.interpolate(vi_show, size=(h, w), mode='bilinear', align_corners=False)
    ir_show = F.interpolate(ir_show, size=(h, w), mode='bilinear', align_corners=False)

    out_path = os.path.join(save_path, 'visual', 'epoch-{:06d}.png'.format(epoch))
    _save_visual_row(out_path, [
        ('IR', ir_show[0]),
        ('VI', vi_show[0]),
        ('Distance', dist_map[0]),
        ('Pred', pred_img[0]),
        ('GT', gt_img[0]),
    ])
    log('visual saved: {}'.format(out_path))


def train(train_loader, model, optimizer):
    model.train()
    loss_fn = nn.L1Loss()
    train_loss = utils.Averager()

    data_norm = config['data_norm']
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()
    fusion_mode = is_fusion_model(model)
    pixel_fusion_mode = is_pixel_fusion_model(model)
    loss_tv_weight = config.get('loss_tv_weight', 0.02)
    loss_soft_weight = config.get('loss_soft_weight', 0.01)
    loss_entropy_weight = config.get('loss_entropy_weight', 0.0)
    loss_glare_rel_weight = config.get('loss_glare_rel_weight', 0.0)
    fusion_int_weight = config.get('fusion_int_weight', 1.0)
    fusion_grad_weight = config.get('fusion_grad_weight', 1.0)
    pixel_loss_mode = config.get('pixel_loss_mode', 'supervised')
    distance_eps = config.get('distance_eps', 0.03)
    glare_threshold = config.get('glare_threshold', 0.85)
    glare_delta = config.get('glare_delta', 0.12)
    glare_sharpness = config.get('glare_sharpness', 20.0)
    ir_info_threshold = config.get('ir_info_threshold', 0.04)
    apply_pixel_runtime_config(model)

    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda()

        if pixel_fusion_mode:
            vi, ir, gt_img = make_pixel_fusion_inputs(batch)
            pred = model(vi, ir)
            model_ = model.module if isinstance(model, nn.DataParallel) else model
            d_ir = model_.last_d_ir
            w_vi = model_.last_w_vi
            if pixel_loss_mode == 'fusion':
                loss_rec = fusion_loss(pred, vi, ir, fusion_int_weight, fusion_grad_weight)
            else:
                loss_rec = loss_fn(pred, gt_img)
            loss_tv = distance_tv_loss(d_ir)
            loss_soft = distance_soft_loss(d_ir, distance_eps)
            entropy = distance_entropy(d_ir)
            loss_glare_rel = glare_reliability_loss(
                w_vi, vi, ir,
                threshold=glare_threshold,
                delta=glare_delta,
                sharpness=glare_sharpness,
                ir_info_threshold=ir_info_threshold)
            loss = (
                loss_rec
                + loss_tv_weight * loss_tv
                + loss_soft_weight * loss_soft
                - loss_entropy_weight * entropy
                + loss_glare_rel_weight * loss_glare_rel
            )
        elif fusion_mode:
            vi, ir, gt = make_fusion_inputs(batch, inp_sub, inp_div, gt_sub, gt_div)
            pred = model(vi, ir, batch['coord'], batch['cell'])
            loss = loss_fn(pred, gt)
        else:
            inp = (batch['inp'] - inp_sub) / inp_div
            pred = model(inp, batch['coord'], batch['cell'])
            gt = (batch['gt'] - gt_sub) / gt_div
            loss = loss_fn(pred, gt)

        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = None; loss = None

    return train_loss.item()


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer)
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model
            val_res = eval_psnr(val_loader, model_,
                data_norm=config['data_norm'],
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize'))
            if config.get('vis_val', is_fusion_model(model_)):
                save_fusion_visualization(
                    val_loader, model_, save_path, epoch,
                    bsize=config.get('vis_bsize', config.get('eval_bsize', 65536)))

            log_info.append('val: psnr={:.4f}'.format(val_res))
            writer.add_scalars('psnr', {'val': val_res}, epoch)
            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--no_timestamp', action='store_true',
                        help='Disable timestamp suffix in save directory name.')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    if not args.no_timestamp:
        save_name += '_' + datetime.now().strftime('%Y%m%d-%H%M%S')
    save_path = os.path.join('./save', save_name)

    main(config, save_path)
