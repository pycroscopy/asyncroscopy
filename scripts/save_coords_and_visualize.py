"""Run pretrained model, extract coordinates, save CSV and a coordinates overlay PNG.

Usage:
    python3 scripts/save_coords_and_visualize.py --model ~/Downloads/G_MD.tar

Outputs:
    notebooks/output/coords.csv
    notebooks/output/coords_overlay.png
    notebooks/output/segmented_mask.npy  (updated)
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
import torch

import atomai as aai


def generate_perfect_crystal(size=512, period_x=16, period_y=16, seed=0):
    import numpy as _np
    rng = _np.random.default_rng(seed)
    x = _np.arange(size)
    y = _np.arange(size)
    X, Y = _np.meshgrid(x, y)

    image = 0.5 * (_np.cos(2 * _np.pi * X / period_x) + 1)
    image += 0.5 * (_np.cos(2 * _np.pi * Y / period_y) + 1)
    image = (image - image.min()) / (image.max() - image.min())
    noise = rng.normal(0, 0.05, image.shape)
    image = _np.clip(image + noise, 0, 1)
    return image.astype(_np.float32)


def robust_segment_from_nn_output(nn_output):
    import numpy as _np
    # nn_output may be numpy array or torch tensor; convert to numpy
    if hasattr(nn_output, 'detach'):
        arr = nn_output.detach().cpu().numpy()
    else:
        arr = _np.array(nn_output)

    # possible shapes observed:
    # (1, H, W, 1)
    # (1, 1, H, W)
    # (1, C, H, W) where C>1
    arr_shape = arr.shape
    if arr.ndim == 4:
        # if last dim is 1, assume channels-last
        if arr_shape[-1] == 1 and arr_shape[0] == 1:
            # squeeze first and last dims -> (H,W,1) then argmax over last
            seg = _np.argmax(arr[0, :, :, :], axis=-1)
            return seg
        # if shape (1,1,H,W)
        if arr_shape[1] == 1:
            seg = _np.argmax(arr[0, 0, :, :], axis=0) if arr.ndim == 4 else _np.argmax(arr, axis=0)
            # above may not be right; safer: if channels dim present
            try:
                seg = _np.argmax(arr[0, :, :, :], axis=0)
            except Exception:
                seg = (arr.squeeze() > arr.mean()).astype(_np.int32)
            return seg
        # generic: assume (1, C, H, W)
        try:
            seg = _np.argmax(arr[0], axis=0)
            return seg
        except Exception:
            return (arr.squeeze() > arr.mean()).astype(_np.int32)
    elif arr.ndim == 3:
        # maybe (C, H, W) or (H, W, C)
        if arr.shape[0] <= 4 and arr.shape[0] < arr.shape[-1]:
            # channels-first?
            seg = _np.argmax(arr, axis=0)
            return seg
        else:
            seg = _np.argmax(arr, axis=-1)
            return seg
    else:
        # fallback threshold
        return (arr.squeeze() > arr.mean()).astype(_np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to model archive (e.g., ~/Downloads/G_MD.tar)')
    parser.add_argument('--out', default='notebooks/output', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # load/generate image (we'll reuse synthetic generator — consistent with previous run)
    img = generate_perfect_crystal()

    # load model
    model_path = os.path.expanduser(args.model)
    print('Loading model from', model_path)
    model = aai.models.load_model(model_path)
    print('Model loaded')

    X = torch.from_numpy(img[None, None, :, :]).float()

    nn_output, coords = model.predict(X, method='atom_find')
    print('Predict done')

    seg = robust_segment_from_nn_output(nn_output)
    print('Segment shape', seg.shape)

    # save segmented mask
    np.save(os.path.join(args.out, 'segmented_mask.npy'), seg)

    # save coords CSV if present
    if coords is not None and 0 in coords:
        coords_arr = np.asarray(coords[0])
        # ensure shape (N,3)
        if coords_arr.ndim == 2 and coords_arr.shape[1] >= 2:
            # columns: x,y,class (if available)
            if coords_arr.shape[1] == 2:
                np.savetxt(os.path.join(args.out, 'coords.csv'), coords_arr, delimiter=',', header='x,y', comments='')
            else:
                np.savetxt(os.path.join(args.out, 'coords.csv'), coords_arr[:, :3], delimiter=',', header='x,y,class', comments='')
            print('Saved coords to', os.path.join(args.out, 'coords.csv'))
    else:
        print('No coords found in model output')

    # create overlay visualization (scatter coords)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img, cmap='gray', origin='lower')
    # overlay segmented mask
    cmap_colors = ['k', 'red', 'blue', 'green', 'yellow']
    cmap = ListedColormap(cmap_colors[: max(2, int(seg.max())+1)])
    ax.imshow(seg, cmap=cmap, alpha=0.4, origin='lower')

    if coords is not None and 0 in coords:
        coords_arr = np.asarray(coords[0])
        if coords_arr.ndim == 2 and coords_arr.shape[1] >= 2:
            x = coords_arr[:, 0]
            y = coords_arr[:, 1]
            if coords_arr.shape[1] >= 3:
                classes = coords_arr[:, 2].astype(int)
            else:
                classes = np.zeros_like(x, dtype=int)
            colors_map = {1: 'red', 2: 'blue', 3: 'green'}
            for cl in np.unique(classes):
                if cl == 0:
                    continue
                mask = classes == cl
                ax.scatter(x[mask], y[mask], s=30, c=colors_map.get(cl, 'white'), edgecolor='yellow', label=f'Class {int(cl)}')
            ax.legend()

    ax.set_title('Segmentation + Coordinates')
    ax.axis('off')
    out_png = os.path.join(args.out, 'coords_overlay.png')
    fig.savefig(out_png, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print('Saved visualization to', out_png)


if __name__ == '__main__':
    main()
