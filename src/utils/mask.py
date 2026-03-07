import os
import zipfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image


SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
)

CHANNEL_MODES: dict[str, str] = {
    "L (grayscale)": "L",
    "RGB": "RGB",
    "RGBA": "RGBA",
}

FORMAT_EXTENSIONS: dict[str, str] = {
    "PNG": ".png",
    "JPEG": ".jpg",
}

@dataclass
class MaskConfig:
    """All user-facing settings for the mask generation pipeline."""

    input_folder: str = "/content/images"
    output_folder: str = "/content/masks"
    output_format: str = "PNG"      
    mask_suffix: str = "_mask"
    channels: str = "L (grayscale)"  

    mask_mode: str = field(init=False)

    def __post_init__(self) -> None:
        if self.channels not in CHANNEL_MODES:
            raise ValueError(
                f"Invalid channels '{self.channels}'. "
                f"Choose from: {list(CHANNEL_MODES)}"
            )
        if self.output_format not in ("PNG", "JPEG", "Same as input"):
            raise ValueError(
                f"Invalid output_format '{self.output_format}'. "
                "Choose 'PNG', 'JPEG', or 'Same as input'."
            )
        self.mask_mode = CHANNEL_MODES[self.channels]


@dataclass
class MaskResult:
    """Outcome of a single mask-generation attempt."""

    source_path: Path
    output_path: Optional[Path] = None
    width: int = 0
    height: int = 0
    success: bool = False
    error: str = ""

    def __str__(self) -> str:
        if self.success:
            return (
                f"{self.source_path.name:<40s} → "
                f"{self.output_path.name}  [{self.width}×{self.height}]"
            )
        return f"{self.source_path.name}: {self.error}"


def collect_images(folder: str) -> list[Path]:
    """Return all supported image paths inside *folder* (non-recursive)."""
    return sorted(
        p for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def resolve_output_path(
    source: Path,
    config: MaskConfig,
) -> tuple[Path, Optional[str]]:
    """
    Return (output_path, save_format) for a given source image.

    save_format is the Pillow format string (e.g. 'PNG') or None when Pillow
    should infer the format from the file extension.
    """
    if config.output_format == "Same as input":
        ext = source.suffix.lower()
        save_fmt = None
    elif config.output_format == "JPEG":
        ext = ".jpg"
        save_fmt = "JPEG"
    else:
        ext = ".png"
        save_fmt = "PNG"

    out_name = source.stem + config.mask_suffix + ext
    out_path = Path(config.output_folder) / out_name
    return out_path, save_fmt


def create_black_mask(size: tuple[int, int], mode: str) -> Image.Image:
    """Create a pure-black Pillow image of the given *size* and *mode*."""
    return Image.new(mode, size, color=0)


def generate_mask(source: Path, config: MaskConfig) -> MaskResult:
    """
    Generate a black mask for a single *source* image.

    Returns a :class:`MaskResult` describing the outcome.
    """
    result = MaskResult(source_path=source)
    try:
        with Image.open(source) as img:
            w, h = img.size

        result.width, result.height = w, h

        mask_mode = config.mask_mode
        mask = create_black_mask((w, h), mask_mode)

        out_path, save_fmt = resolve_output_path(source, config)

        # JPEG does not support alpha — silently downgrade
        if save_fmt == "JPEG" and mask_mode == "RGBA":
            mask = mask.convert("RGB")

        save_kwargs = {"format": save_fmt} if save_fmt else {}
        mask.save(out_path, **save_kwargs)

        result.output_path = out_path
        result.success = True

    except Exception as exc:  # noqa: BLE001
        result.error = str(exc)

    return result


class MaskGenerator:
    """
    High-level orchestrator that runs the full mask-generation pipeline.

    Usage
    -----
    >>> cfg = MaskConfig(input_folder="/content/images")
    >>> gen = MaskGenerator(cfg)
    >>> results = gen.run()
    """

    def __init__(self, config: MaskConfig) -> None:
        self.config = config
        self.results: list[MaskResult] = []

    def run(self) -> list[MaskResult]:
        """Discover images, generate masks, print a summary, return results."""
        os.makedirs(self.config.output_folder, exist_ok=True)

        image_paths = collect_images(self.config.input_folder)
        if not image_paths:
            print(f"No supported images found in '{self.config.input_folder}'.")
            return []

        print(f"Found {len(image_paths)} image(s). Generating masks…\n")

        self.results = [generate_mask(p, self.config) for p in image_paths]

        for r in self.results:
            print(r)

        self._print_summary()
        return self.results

    def successful_masks(self) -> list[Path]:
        """Return output paths for every successfully generated mask."""
        return [r.output_path for r in self.results if r.success and r.output_path]

    def failed_sources(self) -> list[Path]:
        """Return source paths for every failed mask generation."""
        return [r.source_path for r in self.results if not r.success]

    def zip_masks(self, zip_path: str = "/content/masks.zip") -> Path:
        """
        Bundle all successful masks into a ZIP archive.

        Parameters
        ----------
        zip_path:
            Destination path for the ZIP file.

        Returns
        -------
        Path
            The path of the created ZIP file.
        """
        masks = self.successful_masks()
        if not masks:
            raise RuntimeError("No masks available to zip.")

        zip_path = Path(zip_path)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for mask_path in masks:
                zf.write(mask_path, arcname=mask_path.name)

        print(f"Zipped {len(masks)} mask(s) → {zip_path}")
        return zip_path

    def _print_summary(self) -> None:
        succeeded = sum(1 for r in self.results if r.success)
        failed = len(self.results) - succeeded
        print(f"\n{'=' * 55}")
        print(f"Done!  {succeeded} mask(s) saved to '{self.config.output_folder}'")
        if failed:
            print(f"Skipped {failed} file(s) due to errors.")

