import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from PIL import Image
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from src.training.unetpp import MBENUNetPlusPlus
from src.training.attention_unet import MBENAttentionUNet

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
    device: torch.device,
    unetpp_path: str = None,
    attunet_path: str = None,
    from_hub: bool = False,
    unetpp_hub_repo: str = None,
    attunet_hub_repo: str = None,
    hub_filename: str = "model.safetensors",
    hub_token: str = None,
) -> tuple[MBENUNetPlusPlus, MBENAttUNet]:
    """
    Load both the MBEN U-Net++ and MBEN Attention U-Net models.

    Supports loading from local safetensors paths or from HuggingFace Hub.
    Returns a (unetpp_model, attunet_model) tuple, both set to eval mode.

    Args:
        device:           Torch device to load models onto.
        unetpp_path:      Local path to the U-Net++ safetensors file.
        attunet_path:     Local path to the Attention U-Net safetensors file.
        from_hub:         If True, download weights from HuggingFace Hub.
        unetpp_hub_repo:  HuggingFace repo ID for U-Net++ (e.g. "user/mben-unetpp").
        attunet_hub_repo: HuggingFace repo ID for Attention U-Net (e.g. "user/mben-attunet").
        hub_filename:     Filename to download from each Hub repo (default: "model.safetensors").
        hub_token:        Optional HuggingFace read token for private repos.

    Returns:
        Tuple of (MBENUNetPlusPlus, MBENAttUNet), both in eval mode on `device`.
    """

    def _load(model_cls, local_path, hub_repo, mben_out_ch=64):
        if from_hub:
            if not hub_repo:
                raise ValueError(
                    f"hub_repo must be provided for {model_cls.__name__} when from_hub=True."
                )
            print(f"Downloading {model_cls.__name__} from HuggingFace: {hub_repo}/{hub_filename} ...")
            local_path = hf_hub_download(
                repo_id=hub_repo,
                filename=hub_filename,
                token=hub_token,
            )
            print(f"Downloaded to: {local_path}")

        model = model_cls(mben_out_ch=mben_out_ch)
        state_dict = load_file(local_path, device=str(device))
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"{model_cls.__name__} loaded from: {local_path}")
        return model

    unetpp  = _load(MBENUNetPlusPlus, unetpp_path,  unetpp_hub_repo)
    attunet = _load(MBENAttUNet,      attunet_path, attunet_hub_repo)

    return unetpp, attunet


def predict(
    unetpp: MBENUNetPlusPlus,
    attunet: MBENAttUNet,
    image_path: str,
    image_size: int,
    device: torch.device,
    threshold: float = 0.5,
    alpha: float = 0.5,
    sigma: int = 80,
    window_size: int = 31,
) -> dict:
    """
    Run ensemble inference using the weighted combination formula:
        Y = alpha * UNetPP(X) + beta * AttUNet(X),  alpha + beta = 1

    Both models receive the same forensic feature inputs (PRNU, illumination,
    frequency) extracted from the input image. Their output probability maps
    are combined as a weighted sum before thresholding into a binary mask.

    Args:
        unetpp:       Loaded MBENUNetPlusPlus model (eval mode).
        attunet:      Loaded MBENAttUNet model (eval mode).
        image_path:   Path to the input image file.
        image_size:   Spatial resolution to resize inputs to (H = W = image_size).
        device:       Torch device for inference.
        threshold:    Binarization threshold applied to the ensemble probability map.
        alpha:        Weight for U-Net++ output; Attention U-Net receives (1 - alpha).
                      Must be in [0, 1]. Default is 0.5 (equal weighting).
        sigma:        Gaussian sigma for illumination extraction.
        window_size:  Window size for illumination extraction.

    Returns:
        dict with keys:
            "prob_map"      — ensemble probability map, shape (H, W), float32
            "binary_mask"   — thresholded mask, shape (H, W), uint8 (0 or 255)
            "original_rgb"  — original image as numpy array, shape (H0, W0, 3), uint8
            "prob_map_unetpp"   — raw U-Net++ probability map, shape (H, W)
            "prob_map_attunet"  — raw Attention U-Net probability map, shape (H, W)
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
    beta = 1.0 - alpha

    rgb_array = np.array(Image.open(image_path).convert("RGB"))
    prnu_t, illu_t, freq_t, fused_t = preprocess(rgb_array, image_size, sigma, window_size)

    prnu_t  = prnu_t.to(device)
    illu_t  = illu_t.to(device)
    freq_t  = freq_t.to(device)
    fused_t = fused_t.to(device)

    feature_dict = {
        "prnu":        prnu_t,
        "illumination": illu_t,
        "frequency":   freq_t,
    }

    with torch.no_grad():
        logits_unetpp  = unetpp(feature_dict, fused_t)   # (1, 1, H, W)
        logits_attunet = attunet(feature_dict, fused_t)  # (1, 1, H, W)

    # Convert logits to probability maps
    prob_unetpp  = torch.sigmoid(logits_unetpp).squeeze().cpu().numpy()   # (H, W)
    prob_attunet = torch.sigmoid(logits_attunet).squeeze().cpu().numpy()  # (H, W)

    # Weighted ensemble: Y = alpha * UNetPP(X) + beta * AttUNet(X)
    prob_ensemble = alpha * prob_unetpp + beta * prob_attunet             # (H, W)

    binary_mask = (prob_ensemble >= threshold).astype(np.uint8) * 255    # (H, W)

    return {
        "prob_map":         prob_ensemble,
        "binary_mask":      binary_mask,
        "original_rgb":     rgb_array,
        "prob_map_unetpp":  prob_unetpp,
        "prob_map_attunet": prob_attunet,
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