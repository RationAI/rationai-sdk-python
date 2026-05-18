from dataclasses import dataclass
from os import PathLike


@dataclass
class SlideCheckConfig:
    """Configuration for slide quality checks.

    Attributes:
        check_residual: Enable residual artifacts detection.
        check_folding: Enable folding artifacts detection.
        check_blur: Enable blur artifacts detection.
        wb_correction: Enable white balance correction.
        mask_dir: Optional directory with pre-computed tissue masks.
        store_masks_at_original_resolution: Whether resolution
            of the stored masks matches resolution of the level
            the WSI was samples from. If False, the masks are sub-sampled.
    """

    check_residual: bool = True
    check_folding: bool = True
    check_blur: bool = True
    wb_correction: bool = False
    mask_dir: PathLike[str] | str | None = None
    store_masks_at_original_resolution: bool = False


@dataclass
class SlideCheckResult:
    wsi_path: PathLike[str] | str
    xopat_url: str | None = None
    error: str | None = None
    success: bool = False
