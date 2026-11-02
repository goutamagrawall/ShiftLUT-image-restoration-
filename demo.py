"""Generate test images and run all three supplied ShiftLUT restoration tasks."""
import io
import json
import numpy as np
from PIL import Image, ImageDraw
import torch

from restore import ROOT, image_psnr, load_model, restore_image


def main():
    torch.set_num_threads(1)
    folder = ROOT / 'outputs' / 'demo'
    folder.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:96, :128]
    pixels = np.clip(95 + 0.6 * xx + 25 * np.sin(xx / 12) + 20 * np.cos(yy / 13), 0, 255).astype(np.uint8)
    clean = Image.fromarray(pixels)
    draw = ImageDraw.Draw(clean)
    draw.rectangle((10, 12, 43, 45), fill=45)
    draw.ellipse((67, 18, 110, 64), fill=225)
    draw.line((7, 87, 113, 72), fill=20, width=2)
    clean.save(folder / 'clean.png')
    rng = np.random.default_rng(123)
    noise = rng.normal(0, 15, np.asarray(clean).shape)
    noisy = Image.fromarray(np.clip(np.round(np.asarray(clean, dtype=float) + noise), 0, 255).astype(np.uint8))
    compressed = io.BytesIO()
    clean.save(compressed, format='JPEG', quality=10)
    compressed.seek(0)
    with Image.open(compressed) as decoded:
        jpeg = decoded.convert('L')
    low_resolution = clean.resize((32, 24), Image.Resampling.BICUBIC)
    results = []
    for task, damaged in [('sr', low_resolution), ('denoising', noisy), ('deblocking', jpeg)]:
        module, model = load_model(task)
        restored = restore_image(damaged, task, module, model)
        assert restored.size == clean.size
        damaged.save(folder / f'{task}_input.png')
        restored.save(folder / f'{task}_restored.png')
        comparison = Image.new('RGB', (clean.width * 3, clean.height + 24), 'white')
        for index, (label, image) in enumerate([('Clean reference', clean), ('Input (display size)', damaged), ('Restored LUT', restored)]):
            comparison.paste(image.resize(clean.size, Image.Resampling.NEAREST), (index * clean.width, 24))
            ImageDraw.Draw(comparison).text((index * clean.width + 3, 5), label, fill='black')
        comparison.save(folder / f'{task}_comparison.png')
        result = {'task': task, 'input_size': list(damaged.size), 'output_size': list(restored.size), 'restored_psnr_db': image_psnr(restored, clean)}
        if task != 'sr':
            result['input_psnr_db'] = image_psnr(damaged, clean)
        results.append(result)
        print(f'{task}: {damaged.size} -> {restored.size}; restored PSNR {result["restored_psnr_db"]:.2f} dB', flush=True)
    (folder / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'Demo complete. Open the *_comparison.png files in {folder}')
    print('These are synthetic examples, not official benchmark scores. Re-running replaces only the named demo outputs.')


if __name__ == '__main__':
    main()
