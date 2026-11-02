"""Run the supplied ShiftLUT tables on one image, without retraining."""
import argparse
import importlib.util
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch

ROOT = Path(__file__).resolve().parent
CONFIGS = {
    'denoising': ('ShiftLUT_ultra_sigma=15_int', 7, 1),
    'deblocking': ('ShiftLUT_ultra_qf=10_dncnn_v3_int4', 7, 1),
}


def read_image(path, task):
    """Use the grayscale convention of the supplied denoising/deblocking models."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        mode = 'L' if task != 'sr' or image.mode in ('1', 'L') else 'RGB'
        return image.convert(mode)


def load_model(task, sr_stacks=7, device='cpu'):
    if task == 'sr':
        folder, stacks, scale = f'ShiftLUT_sr_s{sr_stacks}_int', sr_stacks, 4
    else:
        folder, stacks, scale = CONFIGS[task]
    module_path = ROOT / 'LUT_test' / task / 'model_LUT.py'
    lut_path = ROOT / 'LUT_test' / 'LUTs' / folder
    if not module_path.is_file() or not (lut_path / 'offset.npy').is_file():
        raise FileNotFoundError(f'Missing model code or tables for {folder}. Keep restore.py in the full project folder.')
    spec = importlib.util.spec_from_file_location(f'shiftlut_{task}', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.LUT(str(lut_path), stacks, scale).to(device).eval()
    return module, model


def restore_image(image, task, module, model, device='cpu'):
    pixels = np.asarray(image, dtype=np.float32).copy()
    if pixels.ndim == 2:
        pixels = pixels[:, :, None]
    tensor = torch.from_numpy(pixels.transpose(2, 0, 1).copy())[None].to(device)
    with torch.inference_mode():
        prediction = module.inference(model, tensor - 128.0)
        if task != 'sr':
            prediction = prediction + tensor
        prediction = prediction.clamp(0, 255).round()
    output = prediction[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    if output.shape[2] == 1:
        output = output[:, :, 0]
    return Image.fromarray(output)


def image_psnr(restored, reference):
    if restored.size != reference.size:
        raise ValueError(f'Reference must have output size {restored.size}; got {reference.size}.')
    a = np.asarray(restored, dtype=np.float64)
    b = np.asarray(reference.convert(restored.mode), dtype=np.float64)
    mse = float(np.mean((a - b) ** 2))
    return math.inf if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=('sr', 'denoising', 'deblocking'))
    parser.add_argument('--input', required=True, type=Path, help='Input image path; quote paths containing spaces.')
    parser.add_argument('--output', type=Path, help='New PNG file; default: outputs/<input>_<task>.png in the project.')
    parser.add_argument('--reference', type=Path, help='Optional clean image, same dimensions as the restored output.')
    parser.add_argument('--sr-stacks', type=int, choices=(0, 1, 7), default=7, help='SR model depth; 0 is the smallest.')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=1, help='CPU thread count; default: 1.')
    parser.add_argument('--overwrite', action='store_true', help='Allow replacement of an existing output, never the input/reference.')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be at least 1.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Use --device cpu or install a compatible CUDA-enabled PyTorch build.')
    torch.set_num_threads(args.threads)
    output = args.output or ROOT / 'outputs' / f'{args.input.stem}_{args.task}.png'
    if output.suffix.lower() != '.png':
        parser.error('--output must end in .png so the restored image is saved losslessly.')
    if output.resolve() == args.input.resolve() or (args.reference and output.resolve() == args.reference.resolve()):
        parser.error('Output must be different from the input and reference paths.')
    if output.exists() and not args.overwrite:
        parser.error(f'Output already exists: {output}. Choose a new filename or use --overwrite.')
    try:
        image = read_image(args.input, args.task)
        reference = read_image(args.reference, args.task) if args.reference else None
        expected_size = tuple(n * 4 for n in image.size) if args.task == 'sr' else image.size
        if reference and reference.size != expected_size:
            raise ValueError(f'Reference must have output size {expected_size}; got {reference.size}.')
        module, model = load_model(args.task, args.sr_stacks, args.device)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        restored = restore_image(image, args.task, module, model, args.device)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        output.parent.mkdir(parents=True, exist_ok=True)
        restored.save(output, format='PNG')
        print(f'Task: {args.task}; device: {args.device}; mode: {image.mode}')
        print(f'Size: {image.width} x {image.height} -> {restored.width} x {restored.height}')
        print(f'Inference time: {elapsed:.2f} seconds (one local run, excludes model loading)')
        if reference is not None:
            print(f'PSNR against reference: {image_psnr(restored, reference):.2f} dB')
        print(f'Saved: {output.resolve()}')
    except (OSError, UnidentifiedImageError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
