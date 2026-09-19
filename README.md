# CardioExplain Vision

CardioExplain Vision is an educational research prototype for detecting the radiographic finding **cardiomegaly** in frontal chest X-rays. It uses transfer learning with ResNet18 and generates a Grad-CAM heatmap to show which image regions influenced a prediction.

It is **not a medical device**, does not diagnose heart disease, and must not be used for patient care.

## What changed from Version 0

The earlier tabular UCI Heart Disease prototype has been replaced by an image pipeline. The new system accepts chest X-rays, uses a convolutional neural network, evaluates patient-separated data, and displays Grad-CAM explanations.

## Computer requirements

- Inference: Windows, macOS, or Linux with 8 GB RAM; a CPU is sufficient.
- Small training experiment: 16 GB RAM and a recent CPU can work slowly.
- Recommended final training: NVIDIA GPU with at least 8 GB VRAM through Google Colab or Kaggle.
- Full CheXpert training is not recommended on a CPU-only computer.

## Dataset

Register for and download **CheXpert v1.0** from Stanford. The loader accepts
either the full-resolution release or `CheXpert-v1.0-small` when available:

https://stanfordmlgroup.github.io/competitions/chexpert/

Expected structure:

```text
<CHEXPERT_ROOT>/
├── train.csv
├── valid.csv
├── train/
└── valid/
```

The dataset may remain outside this repository. Pass its absolute extracted
directory to `--dataset-dir`; do not copy the full release into Git.

### Preparing Stanford's full-resolution batch release

The current full release is split into one metadata/validation batch and three
training-image batches. It can require more than 1 TB when fully extracted.
Create a compact 224x224 version directly from the ZIPs instead:

```bash
python prepare_chexpert_full.py \
  --source-dir /data/chexpertchestxrays-u20210408 \
  --output-dir /data/CheXpert-v1.0-224 \
  --workers 8
```

The command requires batches 1, 2, 3, and 4. It prefers an already extracted
batch directory, otherwise it streams images from the corresponding ZIP. A
rerun skips completed images, so interrupted preparation can be resumed.

The program uses frontal images with Cardiomegaly labels `0` or `1` and excludes uncertain (`-1`) or missing labels. Splits are performed by patient identifier to prevent one patient's images appearing in multiple splits. See `DATASET.md` before downloading or training.

## Setup on Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `python` is unavailable on PATH, install Python 3.11 or 3.12 from python.org. After the environment exists, use `.\.venv\Scripts\python.exe` directly.

## Validate the installation

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## One-command working demonstration

The following command creates generated images when needed, trains a one-epoch
CPU checkpoint, accepts one generated image as input, prints the returned JSON,
and saves both the JSON and Grad-CAM PNG:

```powershell
.\.venv\Scripts\python.exe demo.py
```

Subsequent runs reuse the generated-data checkpoint. Use `--retrain` to execute
the training stage again. This route exists only to prove that the software works;
the generated images and outputs have no medical meaning.

## Local smoke training

Use a limited sample on a CPU-only machine:

```powershell
.\.venv\Scripts\python.exe train.py --dataset-dir "C:\path\to\CheXpert-v1.0" --max-samples 2000 --epochs 2 --batch-size 8
```

This is a pipeline smoke test, not the final experiment.

## Recommended GPU training

```powershell
python train.py --dataset-dir /data/CheXpert-v1.0 --epochs 10 --batch-size 32 --workers 4 --train-full-backbone
```

The script automatically uses CUDA when available. It writes the selected threshold, checkpoint, metrics, plots, and split statistics to `artifacts/vision/`.

For a generated pipeline demonstration, explicitly record the non-medical dataset type:

```powershell
.\.venv\Scripts\python.exe tests\create_synthetic_chexpert.py --output datasets\synthetic-chexpert-demo --patients 60
.\.venv\Scripts\python.exe train.py --dataset-dir datasets\synthetic-chexpert-demo --output-dir artifacts\demo --epochs 1 --batch-size 8 --no-pretrained --dataset-kind synthetic_demo
```

Never present metrics from this generated dataset as clinical or research performance.

## Run the application

After a checkpoint has been trained or copied to `artifacts/vision/best_model.pt`:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501 and upload a de-identified frontal chest X-ray in PNG, JPG, or JPEG format.

## Run one image from the command line

The CLI takes an image path and returns JSON to the terminal. It also saves the JSON result and a Grad-CAM explanation image:

```powershell
.\.venv\Scripts\python.exe predict.py --image "C:\path\to\frontal-xray.jpg"
```

To use a checkpoint in another location:

```powershell
.\.venv\Scripts\python.exe predict.py `
  --image "C:\path\to\frontal-xray.jpg" `
  --checkpoint "C:\path\to\best_model.pt" `
  --output-dir artifacts\predictions
```

The JSON includes `probability`, `threshold`, `classification`, input/checkpoint paths, model test metrics, and the generated explanation path. A result above the threshold is a model flag only; it is not a clinical diagnosis.

## Student responsibilities

Before submission, verify the dataset agreement, understand every retained code path, run the final experiments, inspect errors, document AI assistance accurately, and avoid claims of diagnosis or clinical validity.
