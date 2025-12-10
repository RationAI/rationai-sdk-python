from dataclasses import dataclass
from os import PathLike


@dataclass
class SlideCheckConfig:
    """Configuration for slide quality checks.

    Attributes:
        mask_level: Pyramid level for mask generation.
        sample_level: Pyramid level for sampling.
        check_residual: Enable residual tissue detection.
        check_folding: Enable folding artifact detection.
        check_focus: Enable focus quality assessment.
        wb_correction: Enable white balance correction.
    """

    mask_level: int = 3
    sample_level: int = 1
    check_residual: bool = True
    check_folding: bool = True
    check_focus: bool = True
    wb_correction: bool = False


@dataclass
class SlideCheckResult:
    wsi_path: PathLike[str] | str
    xopat_url: str | None = None
    error: str | None = None
    success: bool = False
