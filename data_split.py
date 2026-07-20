import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

try:
    import cv2
except ImportError:
    cv2 = None


IMG_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
}


def is_image(path):
    return path.is_file() and path.suffix.lower() in IMG_EXTENSIONS


def list_images(path):
    images = sorted([p for p in path.iterdir() if is_image(p)], key=lambda p: p.name)
    if not images:
        raise RuntimeError(f"No images found in {path}")
    return images


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def clean_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def resize_for_la(img, max_side):
    if max_side <= 0:
        return img.convert("RGB"), img.size

    rgb = img.convert("RGB")
    original_size = rgb.size
    work = rgb.copy()
    work.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return work, original_size


def luminance_estimation_with_cv2(img, max_side):
    sigma_list = [15, 60, 90]
    work, original_size = resize_for_la(img, max_side)
    scale = work.size[0] / original_size[0]
    img = np.uint8(np.array(work))
    illuminance = np.ones_like(img).astype(np.float32)
    for sigma in sigma_list:
        sigma = max(1, sigma * scale)
        illuminance_1 = np.log10(cv2.GaussianBlur(img, (0, 0), sigma) + 1e-8)
        illuminance_1 = np.clip(illuminance_1, 0, 255)
        illuminance = illuminance + illuminance_1
    illuminance = illuminance / 3
    denom = np.max(illuminance) - np.min(illuminance) + 1e-6
    light = (illuminance - np.min(illuminance)) / denom
    light = Image.fromarray(np.uint8(light * 255))
    light = light.resize(original_size, Image.Resampling.BILINEAR)
    return np.asarray(light)


def luminance_estimation_with_pil(img, max_side):
    sigma_list = [15, 60, 90]
    work, original_size = resize_for_la(img, max_side)
    scale = work.size[0] / original_size[0]

    illuminance = np.ones_like(np.asarray(work)).astype(np.float32)
    for sigma in sigma_list:
        radius = max(1, sigma * scale)
        blurred = np.asarray(work.filter(ImageFilter.GaussianBlur(radius=radius)))
        illuminance_1 = np.log10(blurred.astype(np.float32) + 1e-8)
        illuminance_1 = np.clip(illuminance_1, 0, 255)
        illuminance = illuminance + illuminance_1
    illuminance = illuminance / 3
    denom = np.max(illuminance) - np.min(illuminance) + 1e-6
    light = (illuminance - np.min(illuminance)) / denom
    light = Image.fromarray(np.uint8(light * 255))
    light = light.resize(original_size, Image.Resampling.BILINEAR)
    return np.asarray(light)


def luminance_estimation(img, max_side):
    if cv2 is not None:
        return luminance_estimation_with_cv2(img, max_side)
    return luminance_estimation_with_pil(img, max_side)


def copy_image(src, dst):
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def write_la(src, dst, max_side):
    ensure_dir(dst.parent)
    image = Image.open(src)
    la = Image.fromarray(luminance_estimation(image, max_side))
    la.save(dst)


def write_black_candidate(dst, size):
    ensure_dir(dst.parent)
    Image.new("RGB", size, (0, 0, 0)).save(dst)


def split_names(names, train_ratio, val_ratio, seed):
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Require 0 < train_ratio, 0 < val_ratio, and train_ratio + val_ratio < 1")

    rng = random.Random(seed)
    shuffled = list(names)
    rng.shuffle(shuffled)

    train_count = int(len(shuffled) * train_ratio)
    val_count = int(len(shuffled) * val_ratio)

    train_names = shuffled[:train_count]
    val_names = shuffled[train_count:train_count + val_count]
    test_names = shuffled[train_count + val_count:]

    return train_names, val_names, test_names


def log_progress(label, index, total):
    if index == 1 or index == total or index % 100 == 0:
        print(f"{label}: {index}/{total}", flush=True)


def prepare_paired_split(names, raw_dir, ref_dir, output_root, split_name, la_max_side):
    input_dir = output_root / split_name / "input"
    gt_dir = output_root / split_name / "GT"
    la_dir = output_root / split_name / "LA"

    total = len(names)
    for index, name in enumerate(names, start=1):
        raw_path = raw_dir / name
        ref_path = ref_dir / name
        copy_image(raw_path, input_dir / name)
        copy_image(ref_path, gt_dir / name)
        write_la(raw_path, la_dir / name, la_max_side)
        log_progress(split_name, index, total)


def prepare_test_split(names, raw_dir, ref_dir, output_root, benchmark_name, la_max_side):
    test_root = output_root / "test" / benchmark_name
    input_dir = test_root / "input"
    gt_dir = test_root / "GT"
    la_dir = test_root / "LA"

    total = len(names)
    for index, name in enumerate(names, start=1):
        raw_path = raw_dir / name
        copy_image(raw_path, input_dir / name)
        write_la(raw_path, la_dir / name, la_max_side)

        ref_path = ref_dir / name
        if ref_path.exists():
            copy_image(ref_path, gt_dir / name)
        log_progress(f"test/{benchmark_name}", index, total)


def prepare_unlabeled(images, output_root, candidate_size, la_max_side):
    input_dir = output_root / "unlabeled" / "input"
    la_dir = output_root / "unlabeled" / "LA"
    candidate_dir = output_root / "unlabeled" / "candidate"

    total = len(images)
    for index, src in enumerate(images, start=1):
        copy_image(src, input_dir / src.name)
        write_la(src, la_dir / src.name, la_max_side)
        write_black_candidate(candidate_dir / src.name, candidate_size)
        log_progress("unlabeled", index, total)


def validate_uieb(root):
    raw_dir = root / "raw-890"
    ref_dir = root / "reference-890"
    challenging_dir = root / "challenging-60"

    for path in [raw_dir, ref_dir, challenging_dir]:
        if not path.is_dir():
            raise RuntimeError(f"Missing required UIEB directory: {path}")

    raw_images = list_images(raw_dir)
    ref_images = list_images(ref_dir)
    raw_names = [p.name for p in raw_images]
    ref_names = [p.name for p in ref_images]
    if raw_names != ref_names:
        missing_ref = sorted(set(raw_names) - set(ref_names))[:10]
        missing_raw = sorted(set(ref_names) - set(raw_names))[:10]
        raise RuntimeError(
            "raw-890 and reference-890 filenames do not match. "
            f"missing_ref={missing_ref}, missing_raw={missing_raw}"
        )

    challenging_images = list_images(challenging_dir)
    return raw_dir, ref_dir, raw_names, challenging_images


def main():
    parser = argparse.ArgumentParser(
        description="Prepare UIEB for Semi-UIR train.py and test.py."
    )
    parser.add_argument(
        "--uieb-root",
        default="/data2/huangwei/EDANet/dataset/UIEB",
        type=Path,
        help="UIEB root containing raw-890, reference-890 and challenging-60.",
    )
    parser.add_argument(
        "--output-root",
        default=Path("./data"),
        type=Path,
        help="Semi-UIR data root consumed by train.py and test.py.",
    )
    parser.add_argument("--benchmark-name", default="benchmarkA")
    parser.add_argument("--train-ratio", default=0.8, type=float)
    parser.add_argument("--val-ratio", default=0.1, type=float)
    parser.add_argument("--seed", default=2022, type=int)
    parser.add_argument(
        "--candidate-size",
        default=256,
        type=int,
        help="Side length of the initial black reliable-bank candidate images.",
    )
    parser.add_argument(
        "--la-max-side",
        default=384,
        type=int,
        help="Long side used to estimate LA before resizing back. Use 0 for full resolution.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove generated output folders before preparing data.",
    )
    args = parser.parse_args()

    raw_dir, ref_dir, names, challenging_images = validate_uieb(args.uieb_root)
    train_names, val_names, test_names = split_names(
        names, args.train_ratio, args.val_ratio, args.seed
    )

    output_root = args.output_root
    generated_roots = [
        output_root / "labeled1",
        output_root / "unlabeled",
        output_root / "val",
        output_root / "test" / args.benchmark_name,
    ]
    for path in generated_roots:
        clean_dir(path) if args.clean else ensure_dir(path)

    prepare_paired_split(train_names, raw_dir, ref_dir, output_root, "labeled1", args.la_max_side)
    prepare_paired_split(val_names, raw_dir, ref_dir, output_root, "val", args.la_max_side)
    prepare_test_split(test_names, raw_dir, ref_dir, output_root, args.benchmark_name, args.la_max_side)
    prepare_unlabeled(
        challenging_images,
        output_root,
        (args.candidate_size, args.candidate_size),
        args.la_max_side,
    )

    print("UIEB preparation finished.")
    print(f"labeled1: {len(train_names)} paired images")
    print(f"val: {len(val_names)} paired images")
    print(f"test/{args.benchmark_name}: {len(test_names)} images")
    print(f"unlabeled: {len(challenging_images)} challenging images")
    print("")
    print("Next commands:")
    print("  python test.py")
    print("  python train.py")


if __name__ == "__main__":
    main()
