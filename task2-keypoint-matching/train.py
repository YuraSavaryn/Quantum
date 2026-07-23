import os
import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune EfficientLoFTR")

    parser.add_argument("--repo_path", type=str, required=True,
                        help="Path to the EfficientLoFTR repository")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the directory containing training .npz files")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to the initial model weights (.ckpt)")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save the finetuned weights")

    parser.add_argument("--img_size", type=int, default=512,
                        help="Image size (must be a multiple of 32)")
    parser.add_argument("--coarse_scale", type=int, default=8,
                        help="Coarse scale factor")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="Learning rate")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=3,
                        help="Number of epochs to keep the backbone frozen")

    return parser.parse_args()


class SatellitePairDataset(Dataset):
    """Reads .npz pairs (image0, image1, homography) from prepare_dataset.py."""

    def __init__(self, data_dir: Path, img_size: int):
        self.files = sorted(glob.glob(str(data_dir / "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")
        self.img_size = img_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        img0, img1, H = data["image0"], data["image1"], data["homography"]

        scale = self.img_size / img0.shape[0]
        if img0.shape[0] != self.img_size:
            img0 = cv2.resize(img0, (self.img_size, self.img_size))
            img1 = cv2.resize(img1, (self.img_size, self.img_size))
            S = np.diag([scale, scale, 1.0]).astype(np.float32)
            H = S @ H @ np.linalg.inv(S)

        gray0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        return {
            "image0": torch.from_numpy(gray0)[None],
            "image1": torch.from_numpy(gray1)[None],
            "homography": torch.from_numpy(H.astype(np.float32)),
        }


def compute_supervision_coarse_homography(batch: dict, coarse_scale: int):
    """
    Builds a coarse-level GT correspondence matrix directly from the homography,
    without depth/pose. The logic is equivalent to spvs_coarse() from the original
    repo, but the warp is performed via H instead of unproject-transform-project.

    Adds to batch:
        conf_matrix_gt : (B, Lc, Sc) binary GT correspondence matrix
        spv_b_ids, spv_i_ids, spv_j_ids : positive pair indices (for loss)
    """
    image0, homography = batch["image0"], batch["homography"]
    B, _, H_dim, W_dim = image0.shape
    hc, wc = H_dim // coarse_scale, W_dim // coarse_scale
    device = image0.device

    yy, xx = torch.meshgrid(
        torch.arange(hc, device=device), torch.arange(wc, device=device), indexing="ij"
    )
    grid0 = torch.stack([xx, yy], dim=-1).float() * coarse_scale + coarse_scale / 2
    grid0 = grid0.reshape(-1, 2)

    b_ids_list, i_ids_list, j_ids_list = [], [], []
    for b in range(B):
        Hb = homography[b]
        pts0_h = torch.cat([grid0, torch.ones(grid0.shape[0], 1, device=device)], dim=1)
        pts1_h = (Hb @ pts0_h.T).T
        pts1 = pts1_h[:, :2] / pts1_h[:, 2:3].clamp(min=1e-6)

        j_col = (pts1[:, 0] / coarse_scale).round().long()
        j_row = (pts1[:, 1] / coarse_scale).round().long()
        valid = (j_col >= 0) & (j_col < wc) & (j_row >= 0) & (j_row < hc)

        i_ids = torch.nonzero(valid, as_tuple=False).squeeze(1)
        j_ids = j_row[valid] * wc + j_col[valid]
        b_ids = torch.full_like(i_ids, b)

        b_ids_list.append(b_ids)
        i_ids_list.append(i_ids)
        j_ids_list.append(j_ids)

    batch["spv_b_ids"] = torch.cat(b_ids_list)
    batch["spv_i_ids"] = torch.cat(i_ids_list)
    batch["spv_j_ids"] = torch.cat(j_ids_list)
    batch["hc"], batch["wc"] = hc, wc
    return batch


def coarse_focal_loss(conf_matrix: torch.Tensor, batch: dict, alpha=0.25, gamma=2.0):
    """
    Focal loss on positive GT pairs for coarse matching (analogous to coarse loss
    from LoFTR, simplified - without explicit negative mining, because dual-softmax already
    pushes non-matches to zero).
    """
    b_ids, i_ids, j_ids = batch["spv_b_ids"], batch["spv_i_ids"], batch["spv_j_ids"]
    if len(b_ids) == 0:
        return torch.tensor(0.0, device=conf_matrix.device, requires_grad=True)

    conf = conf_matrix[b_ids, i_ids, j_ids].clamp(min=1e-6, max=1 - 1e-6)
    loss = -alpha * (1 - conf) ** gamma * conf.log()
    return loss.mean()


def compute_supervision_fine_homography(batch, coarse_scale):
    """
    Computes sub-pixel offset (ground truth) for fine matching
    using the homography matrix.
    """
    b_ids = batch["spv_b_ids"]
    i_ids = batch["spv_i_ids"]
    j_ids = batch["spv_j_ids"]
    homography = batch["homography"]
    wc = batch["wc"]

    if len(b_ids) == 0:
        return batch

    i_col = i_ids % wc
    i_row = i_ids // wc
    pts0_x = (i_col.float() * coarse_scale) + (coarse_scale / 2.0)
    pts0_y = (i_row.float() * coarse_scale) + (coarse_scale / 2.0)

    pts0 = torch.stack([pts0_x, pts0_y, torch.ones_like(pts0_x)], dim=1)

    H = homography[b_ids]
    pts1_h = torch.bmm(H, pts0.unsqueeze(2)).squeeze(2)
    pts1_gt_x = pts1_h[:, 0] / pts1_h[:, 2].clamp(min=1e-6)
    pts1_gt_y = pts1_h[:, 1] / pts1_h[:, 2].clamp(min=1e-6)

    j_col = j_ids % wc
    j_row = j_ids // wc
    j_center_x = (j_col.float() * coarse_scale) + (coarse_scale / 2.0)
    j_center_y = (j_row.float() * coarse_scale) + (coarse_scale / 2.0)

    offset_x = pts1_gt_x - j_center_x
    offset_y = pts1_gt_y - j_center_y

    valid_mask = (offset_x.abs() <= coarse_scale / 2) & (offset_y.abs() <= coarse_scale / 2)

    batch["expt_coords_gt"] = torch.stack([offset_x, offset_y], dim=1)[valid_mask]
    batch["fine_valid_mask"] = valid_mask

    return batch


def fine_l2_loss(batch):
    """
    Simple L2 (MSE) loss for the fine matching module.
    """
    if "expt_coords_gt" not in batch or len(batch["expt_coords_gt"]) == 0:
        return torch.tensor(0.0, device=batch["image0"].device, requires_grad=True)

    valid_mask = batch["fine_valid_mask"]

    if "expected_coords" not in batch:
        return torch.tensor(0.0, device=batch["image0"].device, requires_grad=True)

    pred_offsets = batch["expected_coords"][valid_mask]
    gt_offsets = batch["expt_coords_gt"]

    loss = torch.nn.functional.mse_loss(pred_offsets, gt_offsets)

    return loss


def build_model(ckpt_path, device, full_default_cfg, LoFTR):
    from copy import deepcopy

    cfg = deepcopy(full_default_cfg)
    model = LoFTR(config=cfg)

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state_dict, strict=False)

    return model.to(device)


def set_backbone_trainable(model, trainable: bool):
    for name, module in model.named_modules():
        if "backbone" in name:
            for param in module.parameters(recurse=False):
                param.requires_grad = trainable
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()


def main():
    args = parse_args()

    if args.repo_path not in sys.path:
        sys.path.insert(0, args.repo_path)
    os.chdir(args.repo_path)

    try:
        from src.loftr import LoFTR, full_default_cfg
    except ImportError:
        raise ImportError(f"Could not import LoFTR from {args.repo_path}. Ensure the path is correct.")

    print("Import successful!")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data_dir = Path(args.data_dir)
    dataset = SatellitePairDataset(train_data_dir, img_size=args.img_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    model = build_model(args.ckpt_path, device, full_default_cfg, LoFTR)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        set_backbone_trainable(model, trainable=epoch >= args.freeze_backbone_epochs)
        model.train()

        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()

        epoch_loss = 0.0
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            batch = compute_supervision_coarse_homography(batch, coarse_scale=args.coarse_scale)
            batch = compute_supervision_fine_homography(batch, coarse_scale=args.coarse_scale)

            model(batch)

            loss_coarse = coarse_focal_loss(batch["conf_matrix"], batch)
            loss_fine = fine_l2_loss(batch)

            loss = loss_coarse + 0.5 * loss_fine

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            if step % 10 == 0:
                print(f"Epoch {epoch} Step {step}/{len(loader)} Loss {loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(loader)

        save_path = output_dir / f"eloftr_finetuned_epoch_{epoch}.ckpt"
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_loss,
        }, save_path)

        print(f"Weights saved to: {save_path}")
        print(f"=== Epoch {epoch} finished, avg loss = {avg_loss:.4f} ===")


if __name__ == "__main__":
    main()