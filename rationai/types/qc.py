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
        subsample_masks: Whether to sub-sample the produced masks.
            If False, the masks will be stored at the same resolution
            as the input WSI is processed at.
        tissue_mask_dir: Optional directory with pre-computed tissue masks.
    """

    check_residual: bool = True
    check_folding: bool = True
    check_blur: bool = True
    wb_correction: bool = False
    subsample_masks: bool = True
    tissue_mask_dir: PathLike[str] | str | None = None


@dataclass
class SlideCheckResult:
    wsi_path: PathLike[str] | str
    xopat_url: str | None = None
    error: str | None = None
    success: bool = False
