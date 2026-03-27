
import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from PIL import Image
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from src.training.unetpp import MBENUNetPlusPlus

from src.utils.extract import (extract_prnu, 
                               extract_frequency, 
                               extract_illumination)




def _to_gray_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def _norm_tensor(arr: np.ndarray) -> torch.Tensor:
    arr = arr.astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).unsqueeze(0)

def preprocess(rgb_array: np.ndarray, img_size, sigma, window_size):
    h, w = img_size, img_size

    prnu_raw = extract_prnu(rgb_array)
    freq_raw = extract_frequency(rgb_array)
    illu_raw = extract_illumination(rgb_array, sigma, window_size)

    def _resize(arr):
        img = Image.fromarray(_to_gray_uint8(arr)).resize((w, h), Image.BILINEAR)
        return np.array(img, dtype=np.uint8)

    prnu = _resize(prnu_raw)
    freq = _resize(freq_raw)
    illu = _resize(illu_raw)

    prnu_t = _norm_tensor(prnu)
    illu_t = _norm_tensor(illu)
    freq_t = _norm_tensor(freq)

    fused_t = torch.cat([prnu_t, illu_t, freq_t], dim=0)

    return (
        prnu_t.unsqueeze(0),   # (1, 1, H, W)
        illu_t.unsqueeze(0),   # (1, 1, H, W)
        freq_t.unsqueeze(0),   # (1, 1, H, W)
        fused_t.unsqueeze(0),  # (1, 3, H, W)
    )

def load_model(
    safetensors_path: str,
    device: torch.device,
    from_hub: bool = False,
    hub_repo: str = None,
    hub_filename: str = "model.safetensors",
    hub_token: str = None,
) -> MBENUNetPlusPlus:
    
    if from_hub:
        if not hub_repo:
            raise ValueError("hub_repo must be provided when from_hub=True.")
        print(f"Downloading model from HuggingFace: {hub_repo}/{hub_filename} ...")
        safetensors_path = hf_hub_download(
            repo_id   =hub_repo,
            filename  =hub_filename,
            token     =hub_token,
        )
        print(f"Downloaded to: {safetensors_path}")

    model = MBENUNetPlusPlus(mben_out_ch=64)
    state_dict = load_file(safetensors_path, device=str(device))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded from: {safetensors_path}")
    return model

def predict(
    model: MBENUNetPlusPlus,
    image_path: str,
    image_size: int,
    device: torch.device,
    threshold: float = 0.5,
    sigma: int = 80,
    window_size: int = 31
) -> dict:
    
    rgb_array = np.array(Image.open(image_path).convert("RGB"))

    prnu_t, illu_t, freq_t, fused_t = preprocess(rgb_array, image_size, sigma, window_size)
    prnu_t  = prnu_t.to(device)
    illu_t  = illu_t.to(device)
    freq_t  = freq_t.to(device)
    fused_t = fused_t.to(device)

    feature_dict = {
        "prnu": prnu_t,
        "illumination": illu_t,
        "frequency": freq_t,
    }

    with torch.no_grad():
        logits = model(feature_dict, fused_t)   # (1, 1, H, W)

    prob_map    = torch.sigmoid(logits).squeeze().cpu().numpy()          # (H, W)
    binary_mask = (prob_map >= threshold).astype(np.uint8) * 255        # (H, W)

    return {
        "prob_map":    prob_map,
        "binary_mask": binary_mask,
        "original_rgb": rgb_array,
    }

def visualize_prediction(
    result: dict,
    output_path: str = "overlay_output.png",
    colormap: str = "jet",
    overlay_alpha: float = 0.55,
    only_masked: bool = True,
) -> None:

    prob_map    = result["prob_map"]        # (H, W) float32
    binary_mask = result["binary_mask"]     # (H, W) uint8
    rgb_array   = result["original_rgb"]    # (H, W, 3) uint8

    H, W = prob_map.shape

    orig_resized = np.array(
        Image.fromarray(rgb_array).resize((W, H), Image.BILINEAR),
        dtype=np.uint8,
    )

    # FIX 1 — Normalize colormap to masked region's actual min-max range
    #          so the gradient is visible even when confidence is uniformly high
    cmap = cm.get_cmap(colormap)
    if only_masked and binary_mask.any():
        masked_vals = prob_map[binary_mask > 0]
        masked_min  = masked_vals.min()
        masked_max  = masked_vals.max()
    else:
        masked_min, masked_max = 0.0, 1.0

    norm_map     = (prob_map - masked_min) / (masked_max - masked_min + 1e-8)
    heatmap_rgba = (cmap(norm_map) * 255).astype(np.uint8)
    heatmap_rgb  = heatmap_rgba[:, :, :3]

    if only_masked:
        mask_bool = binary_mask > 0
        heatmap_rgb[~mask_bool] = 0

    orig_f    = orig_resized.astype(np.float32)
    heatmap_f = heatmap_rgb.astype(np.float32)

    if only_masked:
        mask_3ch  = np.stack([mask_bool] * 3, axis=-1)
        blended_f = np.where(
            mask_3ch,
            orig_f * (1 - overlay_alpha) + heatmap_f * overlay_alpha,
            orig_f,
        )
    else:
        blended_f = orig_f * (1 - overlay_alpha) + heatmap_f * overlay_alpha

    blended = np.clip(blended_f, 0, 255).astype(np.uint8)

    # FIX 2 — Wider figure to prevent colorbar from clipping the overlay panel
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor("#1a1a1a")

    for ax, img, title in zip(
        axes,
        [orig_resized, heatmap_rgb, blended],
        ["Original Image", "Confidence Heatmap", "Overlay"],
    ):
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=12, fontweight="bold")
        ax.axis("off")

    # FIX 3 — Attach colorbar only to the last axis and reflect actual min-max range
    sm   = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=masked_min, vmax=masked_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("Manipulation Confidence", color="white", fontsize=11, labelpad=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    if binary_mask.any():
        masked_conf = prob_map[binary_mask > 0]
        stats_text  = (
            f"Mean Confidence: {masked_conf.mean():.3f}  |  "
            f"Max Confidence: {masked_conf.max():.3f}  |  "
            f"Coverage: {(binary_mask > 0).mean() * 100:.2f}%"
        )
        fig.text(
            0.5, 0.01, stats_text, ha="center", color="white", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="#333", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Overlay saved to: {output_path}")