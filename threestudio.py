import torch

from torch import Tensor
from typing import Callable, Literal
from backend.loader.decorator import KatzukiNode
from backend.nodes.builtin import BaseNode

from kokikit.nerf import (
    AccNeRF as _AccNeRF,
    NeRFAccSampler,
    ThreestudioAccSampler as _ThreestudioAccSampler,
    NeRFAccField as _NeRFAccField,
    SHMLPBackground as _SHMLPBackground,
)

from kokikit.nerf import DensityInit

from KatUIDiffusionBasics.basic import Sigmoid, SoftPlus
from .nerf import Magic3DDensityInit


class ThreestudioAccSampler(BaseNode):

    @KatzukiNode(node_type="nerf.sampler.threestudio_acc_sampler")
    def __init__(self) -> None:
        pass

    def execute(
        self,
        aabb_scale: int = 1,
        resolution: int = 32,
        levels: int = 1,
        grid_prune: bool = True,
        prune_alpha_threshold: bool = True,
        stratified: bool = False,
        num_samples_per_ray: int = 512,
        radius: float = 1.0,
    ) -> _ThreestudioAccSampler:
        return _ThreestudioAccSampler(
            aabb_scale=aabb_scale,
            resolution=resolution,
            levels=levels,
            grid_prune=grid_prune,
            prune_alpha_threshold=prune_alpha_threshold,
            stratified=stratified,
            num_samples_per_ray=num_samples_per_ray,
            radius=radius,
        )


class NeRFAccField(BaseNode):

    @KatzukiNode(node_type="nerf.field.nerf_acc_field")
    def __init__(self) -> None:
        pass

    def execute(
            self,
            density_init: DensityInit = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
            density_activation: Callable = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
            color_activation: Callable = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
            interpolation: Literal['Nearest', 'Linear', 'Smoothstep'] = 'Smoothstep',
            bbox: Tensor = torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32),
    ) -> _NeRFAccField:
        if density_init is None:
            density_init = Magic3DDensityInit().execute()
        if density_activation is None:
            density_activation = SoftPlus().execute()
        if color_activation is None:
            color_activation = Sigmoid().execute()

        return _NeRFAccField(
            density_init=density_init,
            density_activation=density_activation,
            color_activation=color_activation,
            interpolation=interpolation,
            bbox=bbox,
        )


class SHMLPBackground(BaseNode):

    @KatzukiNode(node_type="nerf.field.sh_mlp_background")
    def __init__(self) -> None:
        pass

    def execute(
        self,
        color_activation: Callable = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
        random_background_probability: float = 0.0,
        same_random_across_batch: bool = True,
    ) -> _SHMLPBackground:
        if color_activation is None:
            color_activation = Sigmoid().execute()

        return _SHMLPBackground(
            color_activation=color_activation,
            random_background_probability=random_background_probability,
            same_random_across_batch=same_random_across_batch,
        )


class AccNeRF(BaseNode):

    @KatzukiNode(node_type="nerf.model.acc_nerf")
    def __init__(self) -> None:
        pass

    def execute(
            self,
            sampler: NeRFAccSampler = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
            nerf_field: _NeRFAccField = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
            background_field: _SHMLPBackground = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
            latent_dreamfusion: bool = False,
            degree_latent: int = 3,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cuda"),
    ) -> _AccNeRF:
        if sampler is None:
            sampler = ThreestudioAccSampler().execute()
        if nerf_field is None:
            nerf_field = NeRFAccField().execute()
        if background_field is None:
            background_field = SHMLPBackground().execute()

        return _AccNeRF(
            latent_dreamfusion=latent_dreamfusion,
            degree_latent=degree_latent,
            sampler=sampler,
            nerf_field=nerf_field,
            background_field=background_field,
            dtype=dtype,
            device=device,
        ).to_device(device) # WARNING: better way to do to_device?
