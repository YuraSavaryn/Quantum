"""
inference.py

Standalone script for inference of the fine-tuned EfficientLoFTR on an image pair.

Usage example:
    python inference.py \
        --image0 path/to/patch_a.png \
        --image1 path/to/patch_b.png \
        --ckpt /kaggle/working/checkpoints/eloftr_finetuned_epoch14.ckpt \
        --repo /kaggle/working/EfficientLoFTR \
        --output matches.png \
        --conf_thresh 0.2

Input images can be of any size — the script automatically resizes
them to a size that is a multiple of 32 (a requirement of the backbone model).
"""

import argparse
import sys
from copy import deepcopy

import cv2
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="EfficientLoFTR inference on an image pair")
    parser.add_argument("--image0", type=str, required=True, help="path to the first image")
    parser.add_argument("--image1", type=str, required=True, help="path to the second image")
    parser.add_argument("--ckpt", type=str, required=True, help="path to the .ckpt model")
    parser.add_argument("--repo", type=str, required=True,
                        help="path to the cloned EfficientLoFTR repository")
    parser.add_argument("--output", type=str, default="matches.png", help="where to save the visualization")
    parser.add_argument("--matches_out", type=str, default=None,
                         help="optional: .npz with the found match coordinates (mkpts0, mkpts1, mconf)")
    parser.add_argument("--img_size", type=int, default=512,
                        help="size to which images are resized (multiple of 32)")
    parser.add_argument("--conf_thresh", type=float, default=0.2,
                        help="confidence threshold for displaying matches")
    parser.add_argument("--max_matches", type=int, default=200, help="maximum number of lines on the visualization")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu, default is auto-detection")
    return parser.parse_args()


def load_model(ckpt_path: str, repo_path: str, device: str):
    sys.path.insert(0, repo_path)
    from src.loftr import LoFTR, full_default_cfg, reparameter

    cfg = deepcopy(full_default_cfg)
    model = LoFTR(config=cfg)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model = reparameter(model)
    return model.eval().to(device)


def load_image(path: str, img_size: int):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Can't read the image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    size = (img_size // 32) * 32
    img_resized = cv2.resize(img, (size, size))
    return img_resized


def run_inference(model, img0_rgb: np.ndarray, img1_rgb: np.ndarray, device: str):
    gray0 = cv2.cvtColor(img0_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    batch = {
        "image0": torch.from_numpy(gray0)[None, None].to(device),
        "image1": torch.from_numpy(gray1)[None, None].to(device),
    }
    with torch.no_grad():
        model(batch)

    mkpts0 = batch["mkpts0_f"].cpu().numpy()
    mkpts1 = batch["mkpts1_f"].cpu().numpy()
    mconf = batch["mconf"].cpu().numpy()
    return mkpts0, mkpts1, mconf


def draw_and_save(img0, img1, mkpts0, mkpts1, mconf, output_path, conf_thresh, max_matches):
    h, w = img0.shape[:2]
    canvas = np.hstack([img0, img1]).copy()

    keep = mconf > conf_thresh
    idx = np.where(keep)[0]
    if len(idx) > max_matches:
        idx = np.random.choice(idx, max_matches, replace=False)

    cmap = cv2.applyColorMap(
        (mconf[idx] * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET
    ).reshape(-1, 3)

    for k, i in enumerate(idx):
        x0, y0 = mkpts0[i]
        x1, y1 = mkpts1[i]
        color = tuple(int(c) for c in cmap[k])
        pt0 = (int(x0), int(y0))
        pt1 = (int(x1 + w), int(y1))
        cv2.line(canvas, pt0, pt1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt0, 2, color, -1)
        cv2.circle(canvas, pt1, 2, color, -1)

    cv2.putText(canvas, f"matches: {len(idx)}/{len(mkpts0)} (thresh={conf_thresh})",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"Visualizations are saved: {output_path}")


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.ckpt, args.repo, device)
    img0 = load_image(args.image0, args.img_size)
    img1 = load_image(args.image1, args.img_size)

    mkpts0, mkpts1, mconf = run_inference(model, img0, img1, device)
    print(f"Found {len(mkpts0)} matches "
          f"(Average confidence: {mconf.mean():.3f}, "
          f"Above the threshold {args.conf_thresh}: {(mconf > args.conf_thresh).sum()})")

    draw_and_save(img0, img1, mkpts0, mkpts1, mconf,
                  args.output, args.conf_thresh, args.max_matches)

    if args.matches_out:
        np.savez_compressed(args.matches_out, mkpts0=mkpts0, mkpts1=mkpts1, mconf=mconf)
        print(f"Coordinates of matches are saved: {args.matches_out}")


if __name__ == "__main__":
    main()