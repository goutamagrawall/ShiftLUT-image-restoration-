"""Image preparation and preview helpers for the local restoration interface."""
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from restore import read_image

MAX_DECODE_PIXELS = 40_000_000
MAX_PROCESS_PIXELS = 1_048_576
SAMPLE_DIR = Path(__file__).resolve().parent / 'assets' / 'samples'
SAMPLES = {
    'chart': ('Test chart', None, 'Shapes, gradients, and edges.'),
    'portrait': ('Portrait', 'astronaut.png', 'A portrait with facial detail and fabric texture. Photo: NASA.'),
    'cat': ('Cat', 'chelsea.png', 'Fur, whiskers, and fine edges. Photo: Stefan van der Walt (CC0).'),
    'coffee': ('Coffee', 'coffee.png', 'Smooth surfaces and wood grain. Photo: Rachel Michetti (CC0).'),
    'camera': ('Camera scene', 'camera.png', 'A grayscale scene with contrasting detail. Photo: Lav Varshney (CC0).'),
}


def png_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def prepare_upload(data, task, max_edge):
    if len(data) > 20 * 1024 * 1024:
        raise ValueError('Choose an image smaller than 20 MB.')
    with Image.open(BytesIO(data)) as probe:
        if probe.width * probe.height > MAX_DECODE_PIXELS:
            raise ValueError('Choose an image with fewer than 40 million pixels.')
    image = read_image(BytesIO(data), task)
    original_size = image.size
    if max_edge:
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    if image.width * image.height > MAX_PROCESS_PIXELS:
        raise ValueError('For this image, choose a 256 px or 512 px processing size. Original-size processing supports up to 1 megapixel.')
    return image, original_size


def clean_sample(sample='chart'):
    """Load a bundled photo or create the original test chart."""
    if sample not in SAMPLES:
        raise ValueError(f'Unknown sample: {sample}')
    filename = SAMPLES[sample][1]
    if filename:
        with Image.open(SAMPLE_DIR / filename) as source:
            image = source.convert('RGB')
        image.thumbnail((256, 192), Image.Resampling.LANCZOS)
        return image
    yy, xx = np.mgrid[:128, :192]
    pixels = np.clip(100 + xx * 0.4 + 30 * np.sin(xx / 17) + 20 * np.cos(yy / 19), 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 20, 68, 82), fill=40)
    draw.ellipse((98, 24, 157, 90), fill=230)
    draw.line((14, 114, 174, 100), fill=30, width=3)
    return image.convert('RGB')


def sample_image(task, sample='chart'):
    """Create a repeatable degraded input for the selected restoration task."""
    image = clean_sample(sample)
    if task == 'sr':
        image = image.resize((image.width // 4, image.height // 4), Image.Resampling.BICUBIC)
    elif task == 'denoising':
        pixels = np.asarray(image.convert('L'), dtype=float)
        noise = np.random.default_rng(123).normal(0, 15, pixels.shape)
        image = Image.fromarray(np.clip(np.round(pixels + noise), 0, 255).astype(np.uint8))
    elif task == 'deblocking':
        buffer = BytesIO()
        image.convert('L').save(buffer, format='JPEG', quality=10)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            image = decoded.convert('L')
    else:
        raise ValueError(f'Unknown restoration task: {task}')
    return png_bytes(image)


def preview(image, size=(900, 460)):
    image = image.copy()
    image.thumbnail(size, Image.Resampling.LANCZOS)
    return image


def wipe_preview(before, after, split):
    after = preview(after).convert('RGB')
    before = before.convert('RGB').resize(after.size, Image.Resampling.NEAREST)
    cut = round(after.width * split / 100)
    combined = after.copy()
    if cut:
        combined.paste(before.crop((0, 0, cut, after.height)), (0, 0))
    if 0 < cut < after.width:
        ImageDraw.Draw(combined).line((cut, 0, cut, after.height), fill='#087f72', width=2)
    return combined


def download_name(name, task):
    # Treat uploaded names as display names, never as server filesystem paths.
    stem = Path(name.replace('\\', '/')).stem
    stem = ''.join(c for c in stem if c.isalnum() or c in '-_')[:100] or 'image'
    return f'{stem}_{task}.png'
