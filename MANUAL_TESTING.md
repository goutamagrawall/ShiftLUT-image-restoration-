# Test ShiftLUT on your computer

You do not need a frontend. `restore.py` reads one image and writes a restored PNG using the supplied lookup tables. No training, checkpoint download, or CNN-to-LUT conversion is needed. `demo.py` creates its own images to check all three tasks.

These helpers are separate from the original research scripts. They require Python 3.12 and were tested with Python 3.12.10 on Windows using CPU inference. Start with the installation instructions in [readme.md](readme.md#install-and-run).

## Interactive upload interface

The local interface lets you attach an image, choose a restoration, preview before/after side by side or with a wipe slider, and download a PNG. Choose **Try a sample** to select a test chart, portrait, cat, coffee, or camera scene using the picker or gallery buttons. Samples are bundled locally and adapted to each task: low-resolution color inputs for enlargement, noisy grayscale inputs for denoising, and JPEG-compressed grayscale inputs for cleanup. Gallery thumbnails show the clean sources. Credits are in `assets/samples/README.md`. All restoration runs on this computer using the supplied LUTs; uploaded images and results are kept in the browser session's server memory rather than written to the project folder. A refresh or server restart may end the session, so download any result you want to keep.

After creating the environment and installing `requirements-ui.txt`, start the interface from the project folder:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501). Keep the terminal open while using the interface; press Ctrl+C to stop the server. Alternatively, run `.\start_ui.ps1` if your PowerShell policy permits local scripts. The direct Python command above does not require changing that policy.

The included configuration disables file watching. Stop the server with Ctrl+C and rerun the command after editing Python files so changes to imported helper modules are loaded. On macOS/Linux, use `.venv/bin/python` wherever this guide uses `.\.venv\Scripts\python.exe`.

For a fresh Python 3.12 environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The interface accepts PNG, JPEG, WebP, and BMP files up to 20 MB. Processing defaults to Original size (up to 1 megapixel), so Enlarge 4× produces four times the uploaded width and height. Larger originals require an explicit preview-size selection. The 256-pixel and 512-pixel preview settings shrink larger uploads before restoration; this also reduces download dimensions and can produce a result smaller than the upload. The interface warns about downsizing and shows uploaded, processing, and output dimensions before you restore. On-screen previews fit the available space; the PNG download retains the stated output dimensions. The original upload is not overwritten. Denoising and JPEG cleanup produce grayscale output, matching the supplied pretrained configurations. Changing the image, task, model, or processing size clears the old result so that a download always matches the current selection.

Run the interface's automated checks with the same Python environment:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The checks exercise all three models through the interface, upload decoding, invalid-image handling, explicit resizing, result invalidation, and comparison controls.

## 1. Open PowerShell in the actual project folder

Use the cloned or extracted project folder containing `app.py`, `restore.py`, `net2LUT.py`, and `LUT_test`:

```powershell
cd ShiftLUT-image-restoration-
```

After completing the environment setup below, run the generated demo:

```powershell
.\.venv\Scripts\python.exe demo.py
Invoke-Item .\outputs\demo
```

All commands use the project-local `.venv` environment, so environment activation is optional.

## 2. Create a Python environment and install dependencies

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
```

Use Python 3.12 for the pinned setup. If `python --version` reports a different version and the Python launcher is installed, use `py -3.12 -m venv .venv` instead. The CPU dependency file downloads a CPU-only PyTorch build. The runner needs only PyTorch, NumPy, and Pillow; a GPU is optional. There is no need to activate the environment or change PowerShell execution policy.

The separate `requirements.txt` supports a normal PyTorch installation. `requirements-research.txt` adds dependencies for the original training/benchmark scripts, which still need dataset paths, checkpoint selections, and distributed GPU settings configured.

## 3. Test immediately without supplying an image

```powershell
.\.venv\Scripts\python.exe demo.py
Invoke-Item .\outputs\demo
```

The demo creates a clean test chart, makes low-resolution/noisy/JPEG-degraded versions, and runs all three LUT models. Open `sr_comparison.png`, `denoising_comparison.png`, and `deblocking_comparison.png` to inspect the clean reference, degraded input, and restored image. `results.json` records dimensions and PSNR scores. Re-running the demo replaces its generated files inside `outputs/demo`.

Successful execution means all three tasks finish and save their images. SR should enlarge 32×24 to 128×96. Denoising and deblocking should keep the input dimensions. On the generated chart, denoising and deblocking should improve PSNR; exact numeric values can vary with dependency and device behavior. These examples do not reproduce the paper's benchmark scores.

## 4. Test your own image

Replace the example input path below with an actual image. Quote any path containing spaces. Output directories are created automatically. Existing output files require `--overwrite`; the runner never allows the input or reference path as the output.

### Enlarge a low-resolution image by 4×

```powershell
.\.venv\Scripts\python.exe restore.py --task sr --input "C:\path\to\small_photo.jpg" --output ".\outputs\photo_4x.png"
Invoke-Item .\outputs\photo_4x.png
```

The default SR model uses seven extra stacks. For the smallest model, add `--sr-stacks 0`; the intermediate model is `--sr-stacks 1`. A 128×96 input should produce a 512×384 output. Color inputs remain RGB, and grayscale inputs remain grayscale.

### Remove noise

```powershell
.\.venv\Scripts\python.exe restore.py --task denoising --input "C:\path\to\noisy_photo.png" --output ".\outputs\denoised.png"
```

The supplied model was trained for Gaussian noise with sigma=15 on the 0–255 pixel scale. It is not tuned to every camera/noise level. This runner converts the input to grayscale to follow the pretrained model's evaluation convention. Output dimensions remain unchanged.

### Reduce JPEG compression artifacts

```powershell
.\.venv\Scripts\python.exe restore.py --task deblocking --input "C:\path\to\compressed_photo.jpg" --output ".\outputs\deblocked.png"
```

The supplied model targets JPEG quality factor 10. The runner converts the input to grayscale; output dimensions remain unchanged. Other JPEG quality levels may give different results. The PNG output avoids adding new JPEG compression damage.

Start with a small image, around 128–256 pixels per side, because the Python LUT implementation can be slow and memory-intensive on large photographs. The runner does not resize your input automatically. Motion blur and defocus removal are not supplied tasks.

## 5. Measure quality when you have a clean original

```powershell
.\.venv\Scripts\python.exe restore.py --task denoising --input ".\outputs\demo\denoising_input.png" --reference ".\outputs\demo\clean.png" --output ".\outputs\measured_denoising.png"
```

`--reference` prints full-image PSNR in dB; higher means closer to the reference. The reference must be aligned and exactly the restored output's dimensions. For SR, that means a clean image four times the input width and height. For denoising/deblocking, it means the same width and height as the input. Reference images use the same grayscale/RGB conversion as the task.

Without a clean reference, compare output and input visually. A script finishing successfully establishes execution, not restoration accuracy. This helper's full-image PSNR is not the paper's SR benchmark protocol, which uses luminance and boundary cropping. Printed inference time excludes model loading and is only a local single-run measurement.

## 6. Troubleshooting

- **`No module named torch`, `numpy`, or `PIL`:** run with the exact `.venv` Python executable above and install `requirements-cpu.txt` using that same executable.
- **Environment creation fails at `ensurepip`, or `.venv` has no pip:** bootstrap the fresh environment with your system pip:

  ```powershell
  python -m venv --without-pip .venv
  python -m pip --python .\.venv\Scripts\python.exe install -r requirements-cpu.txt
  ```

- **No matching distribution during installation:** confirm Python 3.12 and 64-bit Windows, and check access to PyPI and the PyTorch CPU package index.
- **Output already exists:** choose a new name or add `--overwrite`.
- **Missing model tables:** keep the complete `LUT_test/LUTs` directory from this checkout beside the scripts.
- **CUDA is unavailable:** keep the default CPU mode. `--device cuda` requires compatible GPU hardware and a separate CUDA-enabled PyTorch installation; the CPU requirements do not enable CUDA.
- **A color photo becomes grayscale:** this is intentional for the supplied denoising/deblocking configuration. SR retains color.
- **Input is merely blurry:** use an image degraded by one of the three supported tasks; the repository does not provide a general deblurring model.

To see all runner options:

```powershell
.\.venv\Scripts\python.exe restore.py --help
```

The original `eval_LUT.py` files are benchmark launchers with hardcoded datasets and distributed GPU settings. Use the new runner for a single-image manual test.
