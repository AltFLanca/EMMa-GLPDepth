
# EMMa-GLPDepth

Generate depth maps from EMMa product images using [GLPDepth](https://github.com/vinvino02/GLPDepth), a monocular depth estimation model.

Dataset access and augmentation storage are handled through [EMMa-Wrapper](https://github.com/AltFLanca/EMMa-Wrapper).

## Setup

Install the project dependencies using [uv](https://docs.astral.sh/uv/). From the repository root, run:

```bash
uv sync --locked
```

A prepared EMMa dataset is required. The dataset and GLPDepth model weights are not included in this repository.

## Repository Structure

- `src/test_single.ipynb` - Notebook for testing individual samples.
- `src/test_inference.ipynb` - Notebook for testing batched inference.
- `src/inference.py` - Generates depth maps using GLPDepth.
- `src/valid_output_filtering.py` - Produces a list of valid ASINs from the generated augmentations.
- `src/visualize_augmentation_output.ipynb` - Visualizes generated depth maps for inspection.

## Generate Augmentations

Only the first product image per ASIN is processed. Predicted depth is scaled by the Euclidean norm of the product's package dimensions (converted from inches to metres), so the selected ASINs must have package-dimension metadata.

Depth estimation requires an `.npy` file containing the ASINs to process, provided through the `--asins` argument, and a pretrained GLPDepth checkpoint, provided through `--ckpt`. A compatible checkpoint can be produced using [ShapeNet-GLPDepth-Train](https://github.com/AltFLanca/ShapeNet-GLPDepth-Train).

The inference and depth storage operations can be configured using the following arguments:

- `--ckpt` - Path to the GLPDepth checkpoint.
- `--batch-size` - Number of images processed per batch.
- `--max-depth` - Maximum depth used by the model.
- `--max-depth-eval` - Maximum predicted depth after clamping.
- `--min-depth-eval` - Minimum predicted depth after clamping.
- `--storage` - Depth map storage format (`float32`, `float16`, or `uint16`).

When using `uint16` storage, depth values are quantized to millimetres with a scale factor of 1000. The original depth values can be recovered by dividing the stored values by this factor.

Run inference with:

```bash
uv run src/inference.py \
  --dataset-root /path/to/EMMa \
  --asins /path/to/augmented_asins.npy \
  --output /path/to/EMMa-GLPDepth/corrected/glp_depth_uint16_2.63.h5 \
  --ckpt /path/to/models/glpdepth/corrected/epoch_25_model.ckpt \
  --max-depth 2.63 \
  --max-depth-eval 2.63 \
  --storage uint16
```

Generated depth maps are stored in an HDF5 file at the specified output path. Each record contains:

- `pred` - Predicted depth map.
- `scale` - Scale factor used to encode depth values when using `uint16` storage.

To inspect the generated depth maps, use [visualize_augmentation_output.ipynb](src/visualize_augmentation_output.ipynb).

**Note:** Inference automatically skips ASINs already present in the output file. When changing checkpoints or inference settings, use a new output path to avoid mixing results generated with different configurations.

## Validate Augmentations

The [valid_output_filtering.py](src/valid_output_filtering.py) script processes the generated augmentations and produces a `valid_asins.npy` file in the same directory as the input HDF5 file.

Run validation for each checkpoint output:

```bash
uv run src/valid_output_filtering.py \
  --dataset-root /path/to/EMMa \
  --augmentation-file /path/to/EMMa-GLPDepth/corrected/glp_depth_uint16_2.63.h5
```

Because the output is always named `valid_asins.npy`, validating multiple HDF5 files in the same directory overwrites the previous list.

**Note:** An ASIN is considered valid if its predicted depth map contains at least one finite, positive depth value. Depth maps that do not satisfy this criterion are excluded.