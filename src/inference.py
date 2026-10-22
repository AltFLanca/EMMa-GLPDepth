from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from torchvision.transforms import ToTensor
import io
from PIL import Image
import cv2
from typing import Any

from torch.utils.data import DataLoader
from emma import EmmaDataset
from emma.dataloader import EmmaMapDataset
from emma.dataloader import emma_collate
from emma.augment import HDF5Source
import h5py
import hdf5plugin

from models.model import GLPDepth
from collections import OrderedDict

import time
import utils.logging_ as logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GLP-Depth over EMMa and store metric predictions in "
            "an HDF5 augmentation file."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Directory containing the EMMa dataset.",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Path to the GLP-Depth checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HDF5 file. Defaults to <dataset_root>/glp_depth.h5.",
    )
    parser.add_argument(
        "--augmentation-name",
        default="glp_depth",
        help="Registered augmentation name.",
    )
    parser.add_argument(
        "--asins",
        type=Path,
        required=True,
        help="Path to a .npy file containing the ASINs to process.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "DataLoader workers."
        ),
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor used only when num_workers > 0.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally process only the first N pending ASINs.",
    )
    parser.add_argument(
        "--storage",
        choices=("float32", "float16", "uint16"),
        default="uint16",
        help=(
            "Storage mode for the metric depth map. float16 keeps the "
            "array directly; uint16 stores quantized depth plus a per-item "
            "scale for better compression."
        ),
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--max-depth-eval",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--min-depth-eval",
        type=float,
        default=1e-3,
    )
    return parser.parse_args()


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        print("Ready")
        return torch.device("cuda")
    print("Not Ready")
    return torch.device("cpu")


def load_asins(path: Path) -> list[str]:
    asins = np.load(path, allow_pickle=False)
    return [str(asin) for asin in asins.reshape(-1)]


def resize_image(image: np.ndarray) -> np.ndarray:
    border_size = (640 - 480) // 2
    border_color = [255, 255, 255]
    image = cv2.resize(image, (480, 480))
    image = cv2.copyMakeBorder(
        image,
        0,
        0,
        border_size,
        border_size,
        cv2.BORDER_CONSTANT,
        value=border_color,
    )
    return image


def clamp_pred(pred: torch.Tensor, min_num, max_num) -> torch.Tensor:
    pred = torch.nan_to_num(pred, nan=min_num, posinf=max_num, neginf=max_num)
    return pred.clamp_min(0.0)


def pad_to_batch_size(batch_tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    current_size = batch_tensor.shape[0]
    if current_size == batch_size:
        return batch_tensor

    pad_count = batch_size - current_size
    padding = batch_tensor[-1:].expand(pad_count, *batch_tensor.shape[1:])
    return torch.cat((batch_tensor, padding), dim=0)


def build_model(ckpt_path: Path, max_depth: float, device: torch.device) -> GLPDepth:
    model = GLPDepth(max_depth=max_depth, is_train=False).to(device)
    model_weight = torch.load(ckpt_path, map_location=device)
    if "module" in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model.load_state_dict(model_weight)
    model.eval()
    return model


def make_transform(ds: EmmaDataset):
    to_tensor = ToTensor()

    def my_transform(asin: str, rec: dict[str, Any]) -> dict[str, Any]:
        image_dict = ds.get_images(rec)
        _, first_img_bytes = next(iter(image_dict.items()))
        first_img = Image.open(io.BytesIO(first_img_bytes)).convert("RGB")
        first_img = np.asarray(first_img)
        first_img = resize_image(first_img)

        dimensions = np.asarray(ds.get_package_dimensions(rec), dtype=np.float32)
        dimensions = np.squeeze(dimensions) * np.float32(2.54 / 100.0)
        normalization = np.float32(np.linalg.norm(dimensions))

        return {
            "image": to_tensor(first_img),
            "normalization": torch.as_tensor(normalization),
            "asin": asin,
        }

    return my_transform


def make_writer(
    name: str,
    path: Path,
    compression: Any,
) -> HDF5Source:
    writer_kwargs: dict[str, Any] = {
        "mode": "a",
        "compression": compression,
        "shuffle": False,
        "fletcher32": False,
    }
    return HDF5Source(name, path, **writer_kwargs)


def prediction_record(pred: np.ndarray, storage: str) -> np.ndarray:
    if storage == "float32":
        stored_pred = pred.astype(np.float32, copy=False)
    elif storage == "float16":
        stored_pred = pred.astype(np.float16, copy=False)
    elif storage == "uint16":
        scale = np.float32(1000.0)
        stored_pred = (pred * scale).astype(np.uint16, copy=False)
    else:
        raise ValueError(f"Unknown storage mode: {storage!r}")

    fields: list[tuple[Any, ...]] = [
        ("pred", stored_pred.dtype, stored_pred.shape),
    ]
    if storage == "uint16":
        fields.append(("scale", np.float32))

    record = np.empty(1, dtype=np.dtype(fields))
    record["pred"][0] = stored_pred
    if storage == "uint16":
        record["scale"][0] = scale
    return record


def missing_asins(writer: HDF5Source, asins: list[str]) -> list[str]:
    existing = set(writer.keys())
    return [asin for asin in asins if asin not in existing]


def main() -> None:
    args = parse_args()
    device = resolve_device()

    dataset_root = args.dataset_root
    output_path = args.output or (dataset_root / f"{args.augmentation_name}.h5")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving augmentation to {output_path}")

    ds = EmmaDataset(dataset_root)

    asins = load_asins(args.asins)
    print(f"Candidate ASINs: {len(asins)}")

    compression = hdf5plugin.Blosc(
        cname="lz4hc",
        clevel=5,
        shuffle=hdf5plugin.Blosc.SHUFFLE,
    )

    writer = make_writer(
        args.augmentation_name,
        output_path,
        compression,
    )
    try:
        pending_asins = missing_asins(writer, asins)
        if args.limit is not None:
            pending_asins = pending_asins[: args.limit]
        print(f"Pending ASINs: {len(pending_asins)}")
        if not pending_asins:
            print("Nothing to do.")
            return

        loader_kwargs: dict[str, Any] = {
            "batch_size": args.batch_size,
            "shuffle": False,
            "collate_fn": emma_collate,
            "num_workers": args.num_workers,
            "pin_memory": device.type == "cuda",
        }
        if args.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = args.prefetch_factor

        loader = DataLoader(
            EmmaMapDataset(ds, pending_asins, transform=make_transform(ds)),
            **loader_kwargs,
        )

        model = build_model(
            args.ckpt,
            max_depth=args.max_depth,
            device=device,
        )

        start = time.time()
        for batch_idx, batch in enumerate(loader):
            input_rgbs = batch["image"].to(device)
            normalizations = batch["normalization"].to(device)
            asin_batch = list(batch["asin"])
            real_batch_size = len(asin_batch)

            with torch.inference_mode():
                pred = model(pad_to_batch_size(input_rgbs, args.batch_size))
            pred_depth = pred["pred_d"].squeeze()[:real_batch_size]

            for asin, depth_map, normalization in zip(asin_batch, pred_depth, normalizations, strict=True):
                depth_map = clamp_pred(depth_map * normalization, args.min_depth_eval, args.max_depth_eval)
                depth_map = depth_map.cpu().numpy()

                writer.save(asin, prediction_record(depth_map, args.storage))

            logging.progress_bar(batch_idx, len(loader), 1, 1)

        print(time.time() - start)
    finally:
        writer.close()

    print("Done")


if __name__ == "__main__":
    main()
