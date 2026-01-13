from httpx._client import UseClientDefault
from httpx._types import TimeoutTypes

from rationai._resource import APIResource, AsyncAPIResource


class Slide(APIResource):
    def heatmap(
        self,
        model: str,
        slide_path: str,
        tissue_mask_path: str,
        output_path: str,
        stride_fraction: float = 0.5,
        output_bigtiff_tile_height: int = 512,
        output_bigtiff_tile_width: int = 512,
        timeout: TimeoutTypes | UseClientDefault = 1000,
    ) -> str:
        """Creates a heatmap for a given slide using the specified model.

        Args:
            model: The model identifier to use for heatmap generation.
            slide_path: The path to the slide image.
            tissue_mask_path: The path to the tissue mask image.
            output_path: The path where the output heatmap will be saved. This includes the filename.
            stride_fraction: The fraction of the tile size to use as stride.
            output_bigtiff_tile_height: The tile height of the generated big-tiff heatmap.
            output_bigtiff_tile_width: The tile width of the generated big-tiff heatmap.
            timeout: Optional timeout for the request.

        Returns:
            str: The path to the generated heatmap. Should match the output_path provided.
        """
        response = self._post(
            "heatmap-builder",
            params={
                "model_id": model,
                "slide_path": slide_path,
                "tissue_mask_path": tissue_mask_path,
                "output_path": output_path,
                "stride_fraction": stride_fraction,
                "output_bigtiff_tile_height": output_bigtiff_tile_height,
                "output_bigtiff_tile_width": output_bigtiff_tile_width,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.text


class AsyncSlide(AsyncAPIResource):
    async def heatmap(
        self,
        model: str,
        slide_path: str,
        tissue_mask_path: str,
        output_path: str,
        stride_fraction: float = 0.5,
        output_bigtiff_tile_height: int = 512,
        output_bigtiff_tile_width: int = 512,
        timeout: TimeoutTypes | UseClientDefault = 1000,
    ) -> str:
        """Creates a heatmap for a given slide using the specified model.

        Args:
            model: The model identifier to use for heatmap generation.
            slide_path: The path to the slide image.
            tissue_mask_path: The path to the tissue mask image.
            output_path: The path where the output heatmap will be saved. This includes the filename.
            stride_fraction: The fraction of the tile size to use as stride.
            output_bigtiff_tile_height: The tile height of the generated big-tiff heatmap.
            output_bigtiff_tile_width: The tile width of the generated big-tiff heatmap.
            timeout: Optional timeout for the request.

        Returns:
            str: The path to the generated heatmap. Should match the output_path provided.
        """
        response = await self._post(
            "heatmap-builder",
            params={
                "model_id": model,
                "slide_path": slide_path,
                "tissue_mask_path": tissue_mask_path,
                "output_path": output_path,
                "stride_fraction": stride_fraction,
                "output_bigtiff_tile_height": output_bigtiff_tile_height,
                "output_bigtiff_tile_width": output_bigtiff_tile_width,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.text
