"""Run semantic segmentation on an input image (or generated crystal) using AtomAI.

Usage (from repo root):
    python scripts/run_segmentation.py --image path/to/image.png [--model path/to/model.tar]

If --image is omitted, the script will generate a synthetic perfect crystal image (same as Segmentation2 notebook).
If --model is a .tar containing a .pt/.pth file, the script will extract and try to load it with atomai.

Outputs (saved to notebooks/output/):
 - segmented_mask.npy    : numeric mask (H x W)
 - overlay.png           : overlay of mask on input image

Note: This script expects `atomai` to be installed in the environment. If you don't have a trained model
available, provide a model archive via --model; otherwise the script will attempt to instantiate a fresh
Segmentor but it won't be trained (so results may be meaningless).
"""

import argparse
import os
import tarfile
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

import torch
import atomai as aai


def generate_perfect_crystal(size=512, period_x=16, period_y=16, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)

    image = 0.5 * (np.cos(2 * np.pi * X / period_x) + 1)
    image += 0.5 * (np.cos(2 * np.pi * Y / period_y) + 1)
    image = (image - image.min()) / (image.max() - image.min())
    noise = rng.normal(0, 0.05, image.shape)
    image = np.clip(image + noise, 0, 1)
    return image.astype(np.float32)


def load_image(path):
    if path is None:
        return None
    path = os.path.expanduser(path)
    if path.lower().endswith(('.npy',)):
        return np.load(path)
    else:
        img = Image.open(path).convert('F')
        return np.array(img, dtype=np.float32)


def extract_model_from_tar(tar_path, extraction_dir=None):
    if not os.path.exists(tar_path):
        raise FileNotFoundError(tar_path)
    if extraction_dir is None:
        extraction_dir = tempfile.mkdtemp(prefix='model_extract_')
    else:
        os.makedirs(extraction_dir, exist_ok=True)

    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extraction_dir)

    # find common model files
    for root, _, files in os.walk(extraction_dir):
        for f in files:
            if f.endswith(('.pt', '.pth')):
                return os.path.join(root, f)
    return None


def save_outputs(image, segmented_mask, coords=None, out_dir='notebooks/output'):
    os.makedirs(out_dir, exist_ok=True)

    # Save mask
    mask_path = os.path.join(out_dir, 'segmented_mask.npy')
    np.save(mask_path, segmented_mask)

    # Create overlay
    cmap_colors = ['k', 'red', 'blue', 'green', 'yellow']
    cmap = ListedColormap(cmap_colors[: max(2, int(segmented_mask.max())+1)])

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image, cmap='gray', origin='lower')
    ax.imshow(segmented_mask, cmap=cmap, alpha=0.5, origin='lower')
    ax.axis('off')
    overlay_path = os.path.join(out_dir, 'overlay.png')
    fig.savefig(overlay_path, bbox_inches='tight', dpi=200)
    plt.close(fig)

    print(f"Saved segmented mask to: {mask_path}")
    print(f"Saved overlay to: {overlay_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default=None, help='Path to input image (png/tif/npy). If omitted, generate synthetic perfect crystal')
    parser.add_argument('--model', default=None, help='Path to model archive (.tar) or .pt/.pth file')
    parser.add_argument('--out', default='notebooks/output', help='Output directory')
    args = parser.parse_args()

    image = None
    if args.image:
        image = load_image(args.image)
        if image is None:
            raise RuntimeError('Failed to load image')
    else:
        image = generate_perfect_crystal()

    # Ensure image shape H x W (no channel)
    if image.ndim == 3:
        # if single channel in axis 0 or last, try squeeze
        if image.shape[0] == 1:
            image = image[0]
        elif image.shape[-1] == 1:
            image = image[..., 0]
        else:
            # convert to single-channel by mean
            image = image.mean(axis=-1)

    model = None
    if args.model:
        model_path = os.path.expanduser(args.model)
        # First try to let AtomAI loader handle the provided path directly (it can accept archives)
        try:
            model = aai.models.load_model(model_path)
            print('Loaded model via atomai.models.load_model')
        except Exception as e:
            print('atomai.models.load_model failed on provided path:', e)
            # If it's an archive, try extracting and searching for a .pt/.pth
            if model_path.endswith('.tar') or model_path.endswith('.zip'):
                print(f"Attempting to extract archive {model_path} and locate model file...")
                try:
                    model_file = extract_model_from_tar(model_path)
                except Exception:
                    model_file = None

                if model_file is None:
                    raise RuntimeError('No .pt/.pth model file found inside the archive and AtomAI loader failed')

                try:
                    model = aai.models.load_model(model_file)
                    print('Loaded model via atomai.models.load_model from extracted file')
                except Exception as e2:
                    print('Failed to load extracted model via atomai loader:', e2)
                    # fallback: load weights into a fresh Segmentor
                    try:
                        model = aai.models.Segmentor(nb_classes=3)
                        state = torch.load(model_file, map_location='cpu')
                        if isinstance(state, dict) and 'state_dict' in state:
                            state_dict = state['state_dict']
                        else:
                            state_dict = state
                        model.net.load_state_dict(state_dict)
                        print('Loaded weights into fresh Segmentor')
                    except Exception as e3:
                        raise RuntimeError(f'Cannot load model from archive: {e3}')
            else:
                raise FileNotFoundError(args.model)
    else:
        print('No model provided; instantiating untrained Segmentor (results likely not meaningful)')
        model = aai.models.Segmentor(nb_classes=3)

    # Prepare input tensor shape required by AtomAI methods: (1, 1, H, W)
    X = torch.from_numpy(image[None, None, :, :]).float()

    # Run prediction
    def _nn_to_mask(nn_output):
        arr = np.array(nn_output)
        # handle common shapes returned by different model versions
        if arr.ndim == 4:
            # Possible layouts: (B, C, H, W) or (B, H, W, C)
            if arr.shape[1] <= 16 and arr.shape[1] != arr.shape[2]:
                # (B, C, H, W)
                return np.argmax(arr[0], axis=0)
            else:
                # (B, H, W, C)
                if arr.shape[-1] > 1:
                    return np.argmax(arr[0], axis=-1)
                else:
                    return (arr[0, ..., 0] > arr[0, ..., 0].mean()).astype(np.int32)
        elif arr.ndim == 3:
            # (C, H, W) or (H, W, C)
            if arr.shape[0] <= 16:
                return np.argmax(arr, axis=0)
            elif arr.shape[-1] <= 16:
                return np.argmax(arr, axis=-1)
            else:
                return (arr > arr.mean()).astype(np.int32)
        else:
            return (arr.squeeze() > arr.squeeze().mean()).astype(np.int32)

    try:
        nn_output, coordinates = model.predict(X, method='atom_find')
        segmented = _nn_to_mask(nn_output)
        print('Segmentation completed with atom_find')
    except Exception as e:
        print('atom_find failed, falling back to predict without atom_find:', e)
        nn_out = model.predict(X)
        try:
            segmented = _nn_to_mask(nn_out)
        except Exception:
            segmented = (X.numpy().squeeze() > X.numpy().mean()).astype(np.int32)

    save_outputs(image, segmented, out_dir=args.out)


if __name__ == '__main__':
    main()
