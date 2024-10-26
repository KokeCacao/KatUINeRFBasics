import os
import time
import random
import torch
import warnings
import numpy as np
import base64
import io
import asyncio

from PIL import Image
from torch import Tensor
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from tqdm import tqdm
from typing import Any, Dict, Tuple, Union, Callable, Literal, List, Optional, TypedDict
from backend.loader.decorator import KatzukiNode
from backend import variable
from backend.nodes.builtin import BaseNode
from backend.sio import sio
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from transformers import CLIPVisionModelWithProjection

from kokikit.models import CCProjection, MultiViewUNetWrapperModel, convert_opengl_to_blender
from kokikit.dataset import NeRFDataset as _NeRFDataset, ImageDataset as _ImageDataset
from kokikit.dmtet import DMTet as _DMTet, Gaussian as _Gaussian, GaussianModel as _GaussianModel
from kokikit.diffusion.diffusion_utils import predict_noise_sd, predict_noise_z123, predict_noise_mvdream
from kokikit.utils.utils import SpecifyGradient
from kokikit.nerf import (
    trunc_exp,
    FieldBase,
    NeRFField,
    Collider,
    Sampler,
    Siren,
    Embed,
    DensityInit,
    FARTNeRFContraction as _FARTNeRFContraction,
    Renderer,
    ImageRenderer,
    NeRFRenderer,
    Regularization,
    TimeScheduler,
    WeightScheduler,
    NeRF as _NeRF,
    AccNeRF as _AccNeRF,
    NearFarCollider as _NearFarCollider,
    FARTNeRFField as _FARTNeRFField,
    # inits
    GaussianBlobDensityInit as _GaussianBlobDensityInit,
    DreamfusionDensityInit as _DreamfusionDensityInit,
    Magic3DDensityInit as _Magic3DDensityInit,
    EmptyDensityInit as _EmptyDensityInit,
    HashMLPDensityField as _HashMLPDensityField,
    # samplers
    UniformSampler as _UniformSampler,
    ProposalNetworkSampler as _ProposalNetworkSampler,
    # regularizations
    EntropyRegularization as _EntropyRegularization,
    ContrastRegularization as _ContrastRegularization,
    MeanRegularization as _MeanRegularization,
    L2DerivativeRegularization as _L2DerivativeRegularization,
    # renderers
    IdentityRenderer as _IdentityRenderer,
    RGBRenderer as _RGBRenderer,
    RGBMaxRenderer as _RGBMaxRenderer,
    NormalAlignmentRenderer as _NormalAlignmentRenderer,
    NormalRenderer as _NormalRenderer,
    LambertianRenderer as _LambertianRenderer,
    TexturelessRenderer as _TexturelessRenderer,
    AccumulationRenderer as _AccumulationRenderer,
    DepthRenderer as _DepthRenderer,
    DeltaDepthRenderer as _DeltaDepthRenderer,
    GeometryRenderer as _GeometryRenderer,
    DebugImageRenderer as _DebugImageRenderer,
    # schedulers
    RandomTimeScheduler as _RandomTimeScheduler,
    RandomDecayTimeScheduler as _RandomDecayTimeScheduler,
    LinearTimeScheduler as _LinearTimeScheduler,
    DreamTimeTimeScheduler as _DreamTimeTimeScheduler,
    DreamTimeMaxTimeScheduler as _DreamTimeMaxTimeScheduler,
    DreamTimeGaussianTimeScheduler as _DreamTimeGaussianTimeScheduler,
    ConstantWeightScheduler as _ConstantWeightScheduler,
    SDSWeightScheduler as _SDSWeightScheduler,
    FantasiaWeightScheduler as _FantasiaWeightScheduler,
    DiffusionSchedule as _DiffusionSchedule,
)

from .z123 import Z123ImagePartialEmbedding, Z123Embedding
from KatUIDiffusionBasics.util import should_update

DIRECTORY = os.path.dirname(os.path.abspath(__file__))


class FARTNeRFContraction(BaseNode):

    @KatzukiNode(node_type="nerf.field.contraction.fart_nerf_contraction")
    def __init__(self) -> None:
        pass

    # TODO: contraction isn't used in code now
    def execute(self, setting: Literal["NO_CONTRACTION", "L2_CONTRACTION", "LINF_CONTRACTION"] = "LINF_CONTRACTION") -> _FARTNeRFContraction:
        if setting == "NO_CONTRACTION":
            return _FARTNeRFContraction.NO_CONTRACTION
        elif setting == "L2_CONTRACTION":
            return _FARTNeRFContraction.L2_CONTRACTION
        elif setting == "LINF_CONTRACTION":
            return _FARTNeRFContraction.LINF_CONTRACTION
        else:
            raise ValueError(f"Invalid setting: {setting}")


class NearFarCollider(BaseNode):

    @KatzukiNode(node_type="nerf.dataset.near_far_collider")
    def __init__(self) -> None:
        pass

    def execute(
        self,
        near_plane: float = 1.0,
        far_plane: float = 2 * (1.0 + np.sqrt(3.0)) - 1.0,
        **kwargs: Dict[str, Any],
    ) -> _NearFarCollider:
        """An object that specify the bounary of the scene.

        Args:
            near_plane (float, optional): Near plane clipping of the camera. Defaults to 1.0.
            far_plane (float, optional): Far plane clipping of the camera. Defaults to 2*(1.0 + np.sqrt(3.0))-1.0.

        Returns:
            _NearFarCollider: An object that specify the bounary of the scene.
        """
        return _NearFarCollider(
            near_plane=near_plane,
            far_plane=far_plane,
            **kwargs,
        )


class NeRFDataset(BaseNode):

    @KatzukiNode(node_type="nerf.dataset.nerf_dataset")
    def __init__(self) -> None:
        pass

    def execute(
            self,
            collider: Collider = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
            radius_min: float = 1.0 + np.sqrt(3.0),
            radius_max: float = 1.0 + np.sqrt(3.0),
            theta_min: float = (90 - 45),
            theta_max: float = (90 + 15),
            phi_min: float = -180,
            phi_max: float = 180,
            uniform_sphere_rate: float = 0.0,
            fov_y_min: float = 49.1,
            fov_y_max: float = 49.1,
            device: torch.device = torch.device("cuda"),
    ) -> _NeRFDataset:
        """An object that generate rays for 3D NeRF.

        Args:
            collider (Collider): An object that specify the bounary of the scene.
            radius_min (float, optional): Camera radius from center of the scene. Defaults to 1.0+np.sqrt(3.0).
            radius_max (float, optional): camera radius from center of the scene. Defaults to 1.0+np.sqrt(3.0).
            theta_min (float, optional): Camera theta around the center. Defaults to (90 - 45)/180*np.pi.
            theta_max (float, optional): Camera theta around the center. Defaults to (90 + 15)/180*np.pi.
            phi_min (float, optional): Camera phi around the center. Defaults to -180/180*np.pi.
            phi_max (float, optional): Camera phi around the center. Defaults to 180/180*np.pi.
            uniform_sphere_rate (float, optional): The probability to sample from uniform sphere instead of using min-max above. Defaults to 0.0.
            fov_y_min (float, optional): Field of view of the camera. Defaults to 49.1/180*np.pi.
            fov_y_max (float, optional): Field of view of the camera. Defaults to 49.1/180*np.pi.
            device (torch.device, optional): CPU or GPU where tensor stores. Defaults to torch.device("cuda").

        Returns:
            _NeRFDataset: An object that generate rays for 3D NeRF.
        """
        if collider is None:
            collider = NearFarCollider().execute()

        theta_min = theta_min * np.pi / 180
        theta_max = theta_max * np.pi / 180
        phi_min = phi_min * np.pi / 180
        phi_max = phi_max * np.pi / 180
        fov_y_min = fov_y_min * np.pi / 180
        fov_y_max = fov_y_max * np.pi / 180

        return _NeRFDataset(
            collider=collider,
            radius_min=radius_min,
            radius_max=radius_max,
            theta_min=theta_min,
            theta_max=theta_max,
            phi_min=phi_min,
            phi_max=phi_max,
            uniform_sphere_rate=uniform_sphere_rate,
            fov_y_min=fov_y_min,
            fov_y_max=fov_y_max,
            device=device,
        )


class ImageDataset(BaseNode):

    @KatzukiNode(node_type="nerf.dataset.image_dataset")
    def __init__(self) -> None:
        pass

    def execute(
            self,
            device: torch.device = torch.device("cuda"),
    ) -> _ImageDataset:
        """An object that generate rays for 2D representation.

        Args:
            device (torch.device, optional): CPU or GPU where tensor stores. Defaults to torch.device("cuda").

        Returns:
            _ImageDataset: An object that generate rays for 2D representation.
        """
        return _ImageDataset(device=device)


class SirenModel(BaseNode):

    @KatzukiNode(node_type="nerf.model.siren_model")
    def __init__(self) -> None:
        pass

    def execute(
            self,
            latent_dreamfusion: bool = False,
            degree_latent: int = 3,
            hidden_features: int = 256,
            hidden_layers: int = 3,
            color_activation: torch.nn.Module = torch.nn.Sigmoid(),
            outermost_linear: bool = True,
            first_omega_0: int = 30,
            hidden_omega_0: int = 30,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cuda"),
    ) -> Siren:
        """Siren neural network for representing 2D scene with sinusoidal encoding.

        Args:
            latent_dreamfusion (bool, optional): Whether to use latent scene representation. Defaults to False.
            degree_latent (int, optional): If latent_dreamfusion, then 4 else 3. Defaults to 3.
            hidden_features (int, optional): Settings of Siren. Defaults to 256.
            hidden_layers (int, optional): Settings of Siren. Defaults to 3.
            color_activation (torch.nn.Module, optional): Activation function for color. Defaults to torch.nn.Sigmoid().
            outermost_linear (bool, optional): Settings of Siren. Defaults to True.
            first_omega_0 (int, optional): Settings of Siren. Defaults to 30.
            hidden_omega_0 (int, optional): Settings of Siren. Defaults to 30.
            dtype (torch.dtype, optional): Datatype of tensor. Defaults to torch.float32.
            device (torch.device, optional): CPU or GPU where tensor stores. Defaults to torch.device("cuda").

        Returns:
            Siren: _description_
        """
        return Siren(
            latent_dreamfusion=latent_dreamfusion,
            degree_latent=degree_latent,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            color_activation=color_activation,
            outermost_linear=outermost_linear,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
            dtype=dtype,
            device=device,
        )


class EmbedModel(BaseNode):

    @KatzukiNode(node_type="nerf.model.embed_model")
    def __init__(self) -> None:
        pass

    def execute(
            self,
            embedding_h: int,
            embedding_w: int,
            latent_dreamfusion: bool = False,
            degree_latent: int = 3,
            color_activation: torch.nn.Module = torch.nn.Sigmoid(),
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cuda"),
    ) -> Embed:
        return Embed(
            latent_dreamfusion=latent_dreamfusion,
            embedding_h=embedding_h,
            embedding_w=embedding_w,
            degree_latent=degree_latent,
            color_activation=color_activation,
            dtype=dtype,
            device=device,
        )


class GaussianBlobDensityInit(BaseNode):

    @KatzukiNode(node_type="nerf.init.gaussian_blob_density_init")
    def __init__(self) -> None:
        pass

    def execute(
        self,
        blob_sharpness: float = 0.8,
        blob_density: float = 13.0,
        additive_density: float = -8.0,
        norm_order: float = 2,
    ) -> _GaussianBlobDensityInit:
        return _GaussianBlobDensityInit(
            blob_sharpness=blob_sharpness,
            blob_density=blob_density,
            additive_density=additive_density,
            norm_order=norm_order,
        )


class DreamfusionDensityInit(BaseNode):

    @KatzukiNode(node_type="nerf.init.dreamfusion_density_init")
    def __init__(self) -> None:
        pass

    def execute(
        self,
        blob_density: float = 8.0,
        blob_radius: float = 0.2,
        additive_density: float = -2.0,
    ) -> _DreamfusionDensityInit:
        return _DreamfusionDensityInit(
            blob_density=blob_density,
            blob_radius=blob_radius,
            additive_density=additive_density,
        )


class Magic3DDensityInit(BaseNode):

    @KatzukiNode(node_type="nerf.init.magic_3d_density_init")
    def __init__(self) -> None:
        pass

    def execute(self, density_blob_scale: float = 10.0, density_blob_std: float = 0.5) -> _Magic3DDensityInit:
        return _Magic3DDensityInit(
            density_blob_scale=density_blob_scale,
            density_blob_std=density_blob_std,
        )


class EmptyDensityInit(BaseNode):

    @KatzukiNode(node_type="nerf.init.empty_density_init")
    def __init__(self) -> None:
        pass

    def execute(self, additive_density: float = 0.0) -> _EmptyDensityInit:
        return _EmptyDensityInit(additive_density=additive_density)


class FARTNeRFField(BaseNode):

    @KatzukiNode(node_type="nerf.field.fart_nerf_field")
    def __init__(self) -> None:
        pass

    def execute(
            self,
            aabb_scale: float = 1.0,
            grid_dims: Tuple[int, ...] = (2048, 2048, 2048), # if only cube supported, will use 1st element
            max_sh_order: int = 0,
            color_activation: Union[torch.nn.Module, Callable[..., Any]] = torch.nn.Sigmoid(),
            density_activation: Union[torch.nn.Module, Callable[..., Any]] = trunc_exp,
            contraction: Literal["No", "L2", "LInf"] = "No",
            density_init: DensityInit = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
            background_density_init: float = -24.0,
            interpolation: Literal["Nearest", "Linear", "Smoothstep"] = "Smoothstep",
            color_degree: int = 3,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cuda"),
    ) -> _FARTNeRFField:
        if density_init is None:
            density_init = GaussianBlobDensityInit().execute()

        if contraction == "No":
            contraction_enum = _FARTNeRFContraction.NO_CONTRACTION
        elif contraction == "L2":
            contraction_enum = _FARTNeRFContraction.L2_CONTRACTION
        elif contraction == "LInf":
            contraction_enum = _FARTNeRFContraction.LINF_CONTRACTION
        else:
            raise ValueError(f"Invalid contraction: {contraction}")

        return _FARTNeRFField(
            aabb_scale=aabb_scale,
            grid_dims=grid_dims,
            max_sh_order=max_sh_order,
            color_activation=color_activation,
            density_activation=density_activation,
            contraction=contraction_enum,
            density_init=density_init,
            background_density_init=background_density_init,
            interpolation=interpolation,
            color_degree=color_degree,
            dtype=dtype,
            device=device,
        )


class HashMLPDensityField(BaseNode):

    @KatzukiNode(node_type="nerf.field.hash_mlp_density_field")
    def __init__(self) -> None:
        pass

    def execute(
            self,
            num_layers: int = 2,
            hidden_dim: int = 16,
            contraction: _FARTNeRFContraction = _FARTNeRFContraction.NO_CONTRACTION,
            aabb_scale: float = 1.0,
            density_activation: Union[torch.nn.Module, Callable[..., Any]] = trunc_exp,
            use_linear: bool = False,
            num_levels: int = 5,
            max_res: int = 128,
            base_res: int = 16,
            log2_hashmap_size: int = 17,
            features_per_level: int = 2,
            background_density_init: float = -24.0,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cuda"),
    ) -> _HashMLPDensityField:
        return _HashMLPDensityField(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            contraction=contraction,
            aabb_scale=aabb_scale,
            density_activation=density_activation,
            use_linear=use_linear,
            num_levels=num_levels,
            max_res=max_res,
            base_res=base_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            background_density_init=background_density_init,
            dtype=dtype,
            device=device,
        )


class UniformSampler(BaseNode):

    @KatzukiNode(node_type="nerf.sampler.uniform_sampler")
    def __init__(self) -> None:
        pass

    def execute(
        self,
        stratified: bool = True,
        single_jitter: bool = False,
    ) -> _UniformSampler:
        return _UniformSampler(
            stratified=stratified,
            single_jitter=single_jitter,
        )


class ProposalNetworkSampler(BaseNode):

    @KatzukiNode(node_type="nerf.sampler.proposal_network_sampler")
    def __init__(self) -> None:
        pass

    def execute(
            self,
            proposal_networks: List[NeRFField],
            stratified: bool = True,
            accumulator: Literal["cumsum", "cumprod"] = "cumsum",
            num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96),
            proposal_weights_anneal_max_num_iters: int = 1000,
            proposal_weights_anneal_slope: float = 10,
            single_jitter: bool = False,
            update_sched: Callable[[int], int] = lambda step: int(np.clip(
                a=np.interp(x=step, xp=[0, 5000], fp=[0, 5]), # scale n from 1 to 5 over 5000 warmup steps # type: ignore
                a_min=1,
                a_max=5, # sample every 5 steps after warmup
            )),
    ) -> _ProposalNetworkSampler:
        return _ProposalNetworkSampler(
            stratified=stratified,
            accumulator=accumulator,
            num_proposal_samples_per_ray=num_proposal_samples_per_ray,
            proposal_networks=proposal_networks,
            proposal_weights_anneal_max_num_iters=proposal_weights_anneal_max_num_iters,
            proposal_weights_anneal_slope=proposal_weights_anneal_slope,
            single_jitter=single_jitter,
            update_sched=update_sched,
        )


class NeRF(BaseNode):

    @KatzukiNode(node_type="nerf.model.nerf")
    def __init__(self) -> None:
        pass

    def execute(
            self,
            latent_dreamfusion: bool = False,
            degree_latent: int = 3,
            num_samples: int = 128,
            sampler: Sampler = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
            nerf_field: NeRFField = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
            light_delta: Optional[float] = 0.5,
            accumulator: Literal["cumsum", "cumprod"] = "cumsum",
            ray_samples_chunk_size: int = (512 * 512 * 32) // 128,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cuda"),
    ) -> _NeRF:
        if nerf_field is None:
            nerf_field = FARTNeRFField().execute()
        contraction = nerf_field.contraction
        density_activation = nerf_field.density_activation
        aabb_scale = nerf_field.aabb_scale
        if sampler is None:
            sampler = ProposalNetworkSampler().execute([HashMLPDensityField().execute(
                contraction=contraction,
                density_activation=density_activation,
                aabb_scale=aabb_scale,
            ), HashMLPDensityField().execute(
                contraction=contraction,
                density_activation=density_activation,
                aabb_scale=aabb_scale,
            )])

        return _NeRF(
            latent_dreamfusion=latent_dreamfusion,
            degree_latent=degree_latent,
            num_samples=num_samples,
            sampler=sampler,
            nerf_field=nerf_field,
            light_delta=light_delta,
            accumulator=accumulator,
            ray_samples_chunk_size=ray_samples_chunk_size,
            dtype=dtype,
            device=device,
        ).to_device(device) # WARNING: better way to do to_device?


class GetParameterGroups(BaseNode):

    @KatzukiNode(node_type="nerf.fn.get_parameter_groups")
    def __init__(self) -> None:
        pass

    def execute(self, module: torch.nn.Module, lr: float) -> List[Dict[str, Any]]:
        if hasattr(module, "parameter_groups"):
            return module.parameter_groups(lr=lr)
        else:
            return [{"params": module.parameters(), "lr": lr}]


class DMTet(BaseNode):

    @KatzukiNode(node_type="nerf.model.dmtet")
    def __init__(self) -> None:
        pass

    def execute(
            self,
            latent_dreamfusion: bool = False,
            degree_latent: int = 3,
            nerf_field: NeRFField = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
            nerf_field_state_dict: str = "",
            tet_grid_size: Literal[32, 64, 128] = 128,
            tet_density_threshold: float = 12.0,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cuda"),
            tet_npz_file: str = "128_tets.npz",
            raster_method: Literal['cuda', 'opengl'] = 'opengl',
    ) -> _DMTet:
        if nerf_field is None:
            nerf_field = FARTNeRFField().execute()

        return _DMTet(
            latent_dreamfusion=latent_dreamfusion,
            degree_latent=degree_latent,
            nerf_field=nerf_field,
            nerf_field_state_dict=nerf_field_state_dict,
            tet_grid_size=tet_grid_size,
            tet_density_threshold=tet_density_threshold,
            dtype=dtype,
            device=device,
            tet_npz_path=os.path.join(DIRECTORY, tet_npz_file),
            raster_method=raster_method,
        )


class GaussianModel(BaseNode):

    @KatzukiNode(node_type="nerf.model.gaussian_model")
    def __init__(self) -> None:
        pass

    class ReturnDict(TypedDict):
        gaussian_model: _GaussianModel
        optimizer: torch.optim.Optimizer

    def execute(
        self,
        sh_degree: int = 0,
        percent_dense: float = 0.1,
        points: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        num_pts: int = 5000,
        radius: float = 0.5,
        position_lr_init: float = 0.001, # TODO: try scheduler, and different value for different tasks
        feature_lr: float = 0.01,
        opacity_lr: float = 0.05,
        scaling_lr: float = 0.005,
        rotation_lr: float = 0.005,
    ) -> ReturnDict:
        gaussian_model = _GaussianModel(
            sh_degree=sh_degree,
            percent_dense=percent_dense,
            points=points,
            colors=colors,
            num_pts=num_pts,
            radius=radius,
            position_lr_init=position_lr_init,
            feature_lr=feature_lr,
            opacity_lr=opacity_lr,
            scaling_lr=scaling_lr,
            rotation_lr=rotation_lr,
        )
        gaussian_model.active_sh_degree = sh_degree # no progressive sh-level
        optimizer = gaussian_model.optimizer
        return self.ReturnDict(gaussian_model=gaussian_model, optimizer=optimizer)


class Gaussian(BaseNode):

    @KatzukiNode(node_type="nerf.model.gaussian")
    def __init__(self) -> None:
        pass

    def execute(
            self,
            gaussian_model: _GaussianModel,
            degree_latent: int = 3,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cuda"),
    ) -> _Gaussian:

        gaussian = _Gaussian(
            latent_dreamfusion=False,
            gaussian_model=gaussian_model,
            degree_latent=degree_latent,
            dtype=dtype,
            device=device,
        )
        return gaussian


class EntropyRegularization(BaseNode):

    @KatzukiNode(node_type="nerf.regularization.entropy_regularization")
    def __init__(self) -> None:
        pass

    def execute(self, weight: float = 1.0) -> _EntropyRegularization:
        return _EntropyRegularization(weight=weight)


class ContrastRegularization(BaseNode):

    @KatzukiNode(node_type="nerf.regularization.contrast_regularization")
    def __init__(self) -> None:
        pass

    def execute(self, weight: float = 1.0) -> _ContrastRegularization:
        return _ContrastRegularization(weight=weight)


class MeanRegularization(BaseNode):

    @KatzukiNode(node_type="nerf.regularization.mean_regularization")
    def __init__(self) -> None:
        pass

    def execute(self, weight: float = 1.0) -> _MeanRegularization:
        return _MeanRegularization(weight=weight)


class L2DerivativeRegularization(BaseNode):

    @KatzukiNode(node_type="nerf.regularization.l2_derivative_regularization")
    def __init__(self) -> None:
        pass

    def execute(self, weight: float = 1.0, threshold: float = 0.1) -> _L2DerivativeRegularization:
        return _L2DerivativeRegularization(weight=weight, threshold=threshold)


class IdentityRenderer(BaseNode):

    @KatzukiNode(node_type="nerf.renderer.identity_renderer")
    def __init__(self) -> None:
        pass

    def execute(
        self,
        need_decode: bool = False,
        reg: Optional[Regularization] = None,
        **kwargs: Dict[str, Any],
    ) -> _IdentityRenderer:
        return _IdentityRenderer(
            need_decode=need_decode,
            reg=reg,
            **kwargs,
        )


class DebugImageRenderer(BaseNode):

    @KatzukiNode(node_type="nerf.renderer.debug_image_renderer", hidden=True)
    def __init__(self) -> None:
        pass

    def execute(
        self,
        need_decode: bool = False,
        reg: Optional[Regularization] = None,
        **kwargs: Dict[str, Any],
    ) -> _DebugImageRenderer:
        return _DebugImageRenderer(
            need_decode=need_decode,
            reg=reg,
            **kwargs,
        )


class GeometryRenderer(BaseNode):

    @KatzukiNode(node_type="nerf.renderer.geometry_renderer", hidden=True)
    def __init__(self) -> None:
        pass

    def execute(
        self,
        need_decode: bool = False,
        reg: Optional[Regularization] = None,
        **kwargs: Dict[str, Any],
    ) -> _GeometryRenderer:
        return _GeometryRenderer(
            need_decode=need_decode,
            reg=reg,
            **kwargs,
        )


class AccumulationRenderer(BaseNode):

    @KatzukiNode(node_type="nerf.renderer.accumulation_renderer")
    def __init__(self) -> None:
        pass

    def execute(
        self,
        need_decode: bool = False,
        reg: Optional[Regularization] = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
        **kwargs: Dict[str, Any],
    ) -> _AccumulationRenderer:
        if reg is None:
            reg = EntropyRegularization().execute(weight=1.0)

        return _AccumulationRenderer(
            need_decode=need_decode,
            reg=reg,
            **kwargs,
        )


class DepthRenderer(BaseNode):

    @KatzukiNode(node_type="nerf.renderer.depth_renderer")
    def __init__(self) -> None:
        pass

    def execute(
        self,
        method: Literal["median", "expected"] = "expected",
        need_decode: bool = False,
        reg: Optional[Regularization] = None,
        **kwargs: Dict[str, Any],
    ) -> _DepthRenderer:
        return _DepthRenderer(
            method=method,
            need_decode=need_decode,
            reg=reg,
            **kwargs,
        )


class DeltaDepthRenderer(BaseNode):

    @KatzukiNode(node_type="nerf.renderer.delta_delth_renderer")
    def __init__(self) -> None:
        pass

    def execute(
        self,
        need_decode: bool = False,
        reg: Optional[Regularization] = None,
        **kwargs: Dict[str, Any],
    ) -> _DeltaDepthRenderer:
        return _DeltaDepthRenderer(
            method="median",
            need_decode=need_decode,
            reg=reg,
            **kwargs,
        )


class NormalAlignmentRenderer(BaseNode):

    @KatzukiNode(node_type="nerf.renderer.normal_alignment_renderer")
    def __init__(self) -> None:
        pass

    def execute(
        self,
        need_decode: bool = False,
        reg: Optional[Regularization] = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
        **kwargs: Dict[str, Any],
    ) -> _NormalAlignmentRenderer:
        if reg is None:
            reg = MeanRegularization().execute(weight=0.01)

        return _NormalAlignmentRenderer(
            need_decode=need_decode,
            reg=reg,
            **kwargs,
        )


class RGBRenderer(BaseNode):

    @KatzukiNode(node_type="nerf.renderer.rgb_renderer")
    def __init__(self) -> None:
        pass

    def execute(
        self,
        background_color: Union[Literal["random", "last_sample", "black"], Tensor, torch.nn.Parameter],
        need_decode: bool = True,
        reg: Optional[Regularization] = None,
        **kwargs: Dict[str, Any],
    ) -> _RGBRenderer:
        return _RGBRenderer(
            background_color=background_color,
            need_decode=need_decode,
            reg=reg,
            **kwargs,
        )


class RGBMaxRenderer(BaseNode):

    @KatzukiNode(node_type="nerf.renderer.rgb_max_renderer")
    def __init__(self) -> None:
        pass

    def execute(
        self,
        background_color: Union[Literal["random", "last_sample", "black"], Tensor, torch.nn.Parameter],
        need_decode: bool = True,
        reg: Optional[Regularization] = None,
        **kwargs: Dict[str, Any],
    ) -> _RGBRenderer:
        return _RGBMaxRenderer(
            background_color=background_color,
            need_decode=need_decode,
            reg=reg,
            **kwargs,
        )


class NormalRenderer(BaseNode):

    @KatzukiNode(node_type="nerf.renderer.normal_renderer")
    def __init__(self) -> None:
        pass

    def execute(
        self,
        need_decode: bool = False,
        reg: Optional[Regularization] = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
        **kwargs: Dict[str, Any],
    ) -> _NormalRenderer:
        if reg is None:
            reg = L2DerivativeRegularization().execute(weight=10.0, threshold=0.1)

        return _NormalRenderer(
            background_color="black",
            need_decode=need_decode,
            reg=reg,
            **kwargs,
        )


class LambertianRenderer(BaseNode):

    @KatzukiNode(node_type="nerf.renderer.lambertian_renderer")
    def __init__(self) -> None:
        pass

    def execute(
        self,
        background_color: Union[Literal["random", "last_sample", "black"], Tensor, torch.nn.Parameter],
        ambient: float = 0.5,
        need_decode: bool = True,
        reg: Optional[Regularization] = None,
        **kwargs: Dict[str, Any],
    ) -> _LambertianRenderer:
        return _LambertianRenderer(
            ambient=ambient,
            background_color=background_color,
            need_decode=need_decode,
            reg=reg,
            **kwargs,
        )


class TexturelessRenderer(BaseNode):

    @KatzukiNode(node_type="nerf.renderer.textureless_renderer")
    def __init__(self) -> None:
        pass

    def execute(
            self,
            background_color: Union[Literal["random", "last_sample", "black"], Tensor, torch.nn.Parameter],
            const_color: Tensor = torch.tensor([1.0]),
            ambient: float = 0.5,
            need_decode: bool = True,
            reg: Optional[Regularization] = None,
            **kwargs: Dict[str, Any],
    ) -> _TexturelessRenderer:
        return _TexturelessRenderer(
            const_color=const_color,
            ambient=ambient,
            background_color=background_color,
            need_decode=need_decode,
            reg=reg,
            **kwargs,
        )


class RandomTimeScheduler(BaseNode):

    @KatzukiNode(node_type="nerf.time_scheduler.random_time_scheduler")
    def __init__(self) -> None:
        pass

    def execute(
        self,
        scheduler_scale: int = 1000,
        t_min: float = 0.02,
        t_max: float = 0.98,
    ) -> _RandomTimeScheduler:
        return _RandomTimeScheduler(
            scheduler_scale=scheduler_scale,
            t_min=t_min,
            t_max=t_max,
        )


class RandomDecayTimeScheduler(BaseNode):

    @KatzukiNode(node_type="nerf.time_scheduler.random_decay_time_scheduler")
    def __init__(self) -> None:
        pass

    def execute(
        self,
        scheduler_scale: int = 1000,
        t_min_start: float = 0.975, # 0.2
        t_max_start: float = 0.985, # 0.98
        t_min_end: float = 0.02, # 0.2
        t_max_end: float = 0.50, # 0.5
        t_start_at: float = 0.00,
        t_end_at: float = 0.80,
    ) -> _RandomDecayTimeScheduler:
        return _RandomDecayTimeScheduler(
            scheduler_scale=scheduler_scale,
            t_min_start=t_min_start,
            t_max_start=t_max_start,
            t_min_end=t_min_end,
            t_max_end=t_max_end,
            t_start_at=t_start_at,
            t_end_at=t_end_at,
        )


class LinearTimeScheduler(BaseNode):

    @KatzukiNode(node_type="nerf.time_scheduler.linear_time_scheduler")
    def __init__(self) -> None:
        pass

    def execute(
        self,
        scheduler_scale: int = 1000,
        t_min: float = 0.5,
        t_max: float = 0.98,
    ) -> _LinearTimeScheduler:
        return _LinearTimeScheduler(
            scheduler_scale=scheduler_scale,
            t_min=t_min,
            t_max=t_max,
        )


class DreamTimeTimeScheduler(BaseNode):

    @KatzukiNode(node_type="nerf.time_scheduler.dream_time_time_scheduler")
    def __init__(self) -> None:
        pass

    def execute(self, scheduler_scale: int = 1000) -> _DreamTimeTimeScheduler:
        return _DreamTimeTimeScheduler(scheduler_scale=scheduler_scale)


class DreamTimeMaxTimeScheduler(BaseNode):

    @KatzukiNode(node_type="nerf.time_scheduler.dream_time_max_time_scheduler")
    def __init__(self) -> None:
        pass

    def execute(self, scheduler_scale: int = 1000, t_min: float = 0.02, t_max: float = 0.98) -> _DreamTimeMaxTimeScheduler:
        return _DreamTimeMaxTimeScheduler(
            scheduler_scale=scheduler_scale,
            t_min=t_min,
            t_max=t_max,
        )


class DreamTimeGaussianTimeScheduler(BaseNode):

    @KatzukiNode(node_type="nerf.time_scheduler.dream_time_gaussian_time_scheduler")
    def __init__(self) -> None:
        pass

    def execute(self, scheduler_scale: int = 1000, t_min: float = 0.02, t_max: float = 0.98, deviation: float = 0.01) -> _DreamTimeGaussianTimeScheduler:
        return _DreamTimeGaussianTimeScheduler(
            scheduler_scale=scheduler_scale,
            t_min=t_min,
            t_max=t_max,
            deviation=deviation,
        )


class ConstantWeightScheduler(BaseNode):

    @KatzukiNode(node_type="nerf.weight_scheduler.constant_weight_scheduler")
    def __init__(self) -> None:
        pass

    def execute(self) -> _ConstantWeightScheduler:
        return _ConstantWeightScheduler()


class SDSWeightScheduler(BaseNode):

    @KatzukiNode(node_type="nerf.weight_scheduler.sds_weight_scheduler")
    def __init__(self) -> None:
        pass

    def execute(self, scheduler: DDIMScheduler) -> _SDSWeightScheduler:
        return _SDSWeightScheduler(alphas_cumprod=scheduler.alphas_cumprod)


class FantasiaWeightScheduler(BaseNode):

    @KatzukiNode(node_type="nerf.weight_scheduler.fantasia_weight_scheduler")
    def __init__(self) -> None:
        pass

    def execute(self, scheduler: DDIMScheduler) -> _FantasiaWeightScheduler:
        return _FantasiaWeightScheduler(alphas_cumprod=scheduler.alphas_cumprod)


class DiffusionSchedule(BaseNode):

    @KatzukiNode(node_type="nerf.schedule.diffusion_schedule")
    def __init__(self) -> None:
        pass

    def execute(
        self,
        time_scheduler: TimeScheduler = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
        weight_scheduler: WeightScheduler = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
        steps: int = 10000,
        start_step: int = 0,
    ) -> _DiffusionSchedule:

        if time_scheduler is None:
            time_scheduler = RandomDecayTimeScheduler().execute()
        if weight_scheduler is None:
            weight_scheduler = ConstantWeightScheduler().execute()

        return _DiffusionSchedule(
            time_scheduler=time_scheduler,
            weight_scheduler=weight_scheduler,
            steps=steps,
            start_step=start_step,
        )


class RunSDSNeRF(BaseNode):

    @KatzukiNode(node_type="nerf.basic.run_sds_nerf")
    def __init__(self) -> None:
        pass

    class ReturnDict(TypedDict):
        field: FieldBase
        renderers: List[Renderer]
        optimizer_nerf: torch.optim.Optimizer
        grad_scaler: GradScaler

    def execute(
            self,
            scheduler: DDIMScheduler, # WARNING: check range
            unet: UNet2DConditionModel,
            vae: AutoencoderKL,
            field: FieldBase,
            diffusion_schedule: _DiffusionSchedule,
            renderers: List[Renderer],
            optimizer_nerf: torch.optim.Optimizer,
            text_embeddings_conditional: Tensor, # [B, 77, 1024]
            text_embeddings_unconditional: Tensor, # [B, 77, 1024]
            dataset: _NeRFDataset = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
            h_img: int = 512,
            w_img: int = 512,
            cfg: float = 50.0,
            batch_size: int = 1,
            diffusion_steps: int = 1000,
            latent_nerf_scale: float = 1.0,
            reconstruction_loss: bool = True,
            cfg_rescale: float = 0.5,
            enable_autocast: bool = True,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cuda"),
    ) -> ReturnDict:
        if dataset is None:
            dataset = NeRFDataset().execute() # type: ignore (wait for https://peps.python.org/pep-0671/)

        grad_scaler = GradScaler(enabled=enable_autocast)
        scheduler.set_timesteps(diffusion_steps, device=device)

        pbar = tqdm(iterable=diffusion_schedule, total=len(diffusion_schedule))
        for step, (t_nerf, loss_weight) in enumerate(pbar):
            self.check_execution_state_change(clear_signal_event=lambda x: False)

            ray_bundle = dataset.get_train_ray_bundle(
                h_latent_dataset=h_img, # WARNING: h_latent_dataset naming is misleading due to historial reasons
                w_latent_dataset=w_img,
                cx_latent_dataset=w_img // 2,
                cy_latent_dataset=h_img // 2,
                batch_size=batch_size,
            )
            ray_bundle = ray_bundle.to_device(device=device, dtype=dtype)

            with autocast(enabled=enable_autocast, dtype=torch.float16):
                latents, reg_losses = field.get_latent(ray_bundle=ray_bundle, vae=vae, renderers=renderers, nerf_scale=latent_nerf_scale) # [R, B, C, H, W]
                latents_selected = latents[0, ...] # [B, C, H, W]
                noise_nerf = torch.randn_like(latents_selected) # [B, C, H, W]
                latents_selected_noised = scheduler.add_noise(latents_selected, noise_nerf, t_nerf) # [B, C, H, W] # type: ignore

                with torch.no_grad():
                    noise_pred_base, noise_pred_base_x0 = predict_noise_sd(
                        unet_sd=unet,
                        latents_noised=latents_selected_noised,
                        text_embeddings_conditional=text_embeddings_conditional,
                        text_embeddings_unconditional=text_embeddings_unconditional,
                        cfg=cfg,
                        lora_scale=0,
                        t=t_nerf,
                        scheduler=scheduler,
                        reconstruction_loss=reconstruction_loss,
                        cfg_rescale=cfg_rescale,
                    )
                if noise_pred_base_x0 is None:
                    loss = SpecifyGradient.apply(latents_selected_noised, noise_pred_base - noise_nerf)
                else:
                    loss = 0.5 * torch.nn.functional.mse_loss(latents_selected, noise_pred_base_x0.detach())
                assert isinstance(loss, Tensor)

                loss_nerf = (loss * loss_weight + torch.stack(reg_losses, dim=0).sum(dim=0)) / batch_size
                if not torch.isfinite(loss_nerf):
                    warnings.warn(f"loss_nerf is not finite: {loss_nerf}", RuntimeWarning)
                    loss_nerf = torch.nan_to_num(loss_nerf)

            if enable_autocast:
                grad_scaler.scale(loss_nerf).backward() # type: ignore
                grad_scaler.step(optimizer_nerf)
            else:
                loss_nerf.backward()
                optimizer_nerf.step()

            optimizer_nerf.zero_grad()
            reg_losses = [reg_loss.item() for reg_loss in reg_losses]

            if step % 100 == 0 or step == len(diffusion_schedule) - 1:
                with torch.no_grad():
                    if noise_pred_base is not None:
                        x0_base: Tensor = scheduler.step(noise_pred_base, t_nerf, latents_selected_noised).pred_original_sample # [B, 4, 64, 64] # type: ignore
                        x0_base = vae.decode(1 / vae.config['scaling_factor'] * x0_base).sample # type: ignore
                        self.set_output(f"PredBaseX0", (x0_base / 2 + 0.5).clamp(0, 1))
                    if latents_selected_noised is not None:
                        x0: Tensor = vae.decode(1 / vae.config['scaling_factor'] * latents_selected_noised).sample # type: ignore
                        self.set_output(f"NoisedX0", (x0 / 2 + 0.5).clamp(0, 1))
                    for i, latent in enumerate(latents):
                        x0: Tensor = vae.decode(1 / vae.config['scaling_factor'] * latent).sample # type: ignore
                        self.set_output(f"Truth_{i}", (x0 / 2 + 0.5).clamp(0, 1))
                self.set_output("log", f"{step}/{len(diffusion_schedule)} - T{t_nerf:03d}/{scheduler.timesteps[0].item():04d} [SDS] (NeRF: {'|'.join([f'{loss_value:.4f}' for loss_value in ([loss_nerf.item()] + reg_losses)])})")
                self.set_output("progress", int(100 * step / len(diffusion_schedule)))
                self.send_update()

            if enable_autocast:
                grad_scaler.update()

            # TODO: add NeRF panel update

        return self.ReturnDict(
            field=field,
            renderers=renderers,
            optimizer_nerf=optimizer_nerf,
            grad_scaler=grad_scaler,
        )


class RunZ123NeRF(BaseNode):

    @KatzukiNode(node_type="nerf.basic.run_z123_nerf")
    def __init__(self) -> None:
        pass

    class ReturnDict(TypedDict):
        field: FieldBase
        renderers: List[Renderer]
        optimizer_nerf: torch.optim.Optimizer
        grad_scaler: GradScaler

    def execute(
            self,
            image: Tensor,
            image_encoder_z123: CLIPVisionModelWithProjection,
            cc_projection_z123: CCProjection,
            scheduler: DDIMScheduler, # WARNING: check range
            unet: UNet2DConditionModel,
            vae: AutoencoderKL,
            field: FieldBase,
            diffusion_schedule: _DiffusionSchedule,
            renderers: List[Renderer],
            optimizer_nerf: torch.optim.Optimizer,
            dataset: _NeRFDataset = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
            h_img: int = 256,
            w_img: int = 256,
            cfg: float = 3.0,
            batch_size: int = 1,
            diffusion_steps: int = 1000,
            latent_nerf_scale: float = 1.0,
            reconstruction_loss: bool = True,
            cfg_rescale: float = 0.5,
            enable_autocast: bool = True,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cuda"),
    ) -> ReturnDict:
        if dataset is None:
            dataset = NeRFDataset().execute() # type: ignore (wait for https://peps.python.org/pep-0671/)

        grad_scaler = GradScaler(enabled=enable_autocast)
        scheduler.set_timesteps(diffusion_steps, device=device)

        with torch.no_grad():
            _ = Z123ImagePartialEmbedding().execute(
                image=image,
                image_encoder_z123=image_encoder_z123,
                vae=vae,
                dtype=dtype,
            )
            latents_image = _["latents_image"] # [B, C, H, W]
            image_embedding = _["image_embedding"] # [B, 768]

        pbar = tqdm(iterable=diffusion_schedule, total=len(diffusion_schedule))
        for step, (t_nerf, loss_weight) in enumerate(pbar):
            self.check_execution_state_change(clear_signal_event=lambda x: False)

            with torch.no_grad():
                ray_bundle = dataset.get_train_ray_bundle(
                    h_latent_dataset=h_img, # WARNING: h_latent_dataset naming is misleading due to historial reasons
                    w_latent_dataset=w_img,
                    cx_latent_dataset=w_img // 2,
                    cy_latent_dataset=h_img // 2,
                    batch_size=batch_size,
                )
                ray_bundle = ray_bundle.to_device(device=device, dtype=dtype)
                thetas, phis = ray_bundle.get_thetas_phis() # [B, 3], [B, 3]

                _ = Z123Embedding().execute(
                    image_embedding=image_embedding,
                    cc_projection_z123=cc_projection_z123,
                    thetas=thetas,
                    thetas_ref=90 / 180 * torch.pi,
                    phis=phis,
                    phis_ref=0,
                    zooms=torch.zeros(1, dtype=thetas.dtype, device=thetas.device),
                    zooms_ref=0,
                )
                angle_embeddings_conditional = _["angle_embeddings_conditional"] # [B, 768]
                angle_embeddings_unconditional = _["angle_embeddings_unconditional"] # [B, 768]

            with autocast(enabled=enable_autocast, dtype=torch.float16):
                latents, reg_losses = field.get_latent(ray_bundle=ray_bundle, vae=vae, renderers=renderers, nerf_scale=latent_nerf_scale) # [R, B, C, H, W]
                latents_selected = latents[0, ...] # [B, C, H, W]
                noise_nerf = torch.randn_like(latents_selected) # [B, C, H, W]
                latents_selected_noised = scheduler.add_noise(latents_selected, noise_nerf, t_nerf) # [B, C, H, W] # type: ignore

                with torch.no_grad():
                    noise_pred_z123, noise_pred_z123_x0 = predict_noise_z123(
                        unet_z123=unet,
                        latents_noised=latents_selected_noised,
                        latents_image=latents_image,
                        angle_embeddings_conditional=angle_embeddings_conditional.unsqueeze(-2), # [B, 768] -> [B, 1, 768],
                        angle_embeddings_unconditional=angle_embeddings_unconditional.unsqueeze(-2), # [B, 768] -> [B, 1, 768]
                        cfg=cfg,
                        lora_scale=0,
                        t=t_nerf,
                        scheduler=scheduler,
                        reconstruction_loss=reconstruction_loss,
                        cfg_rescale=cfg_rescale,
                    )
                if noise_pred_z123_x0 is None:
                    loss = SpecifyGradient.apply(latents_selected_noised, noise_pred_z123 - noise_nerf)
                else:
                    loss = 0.5 * torch.nn.functional.mse_loss(latents_selected, noise_pred_z123_x0.detach())
                assert isinstance(loss, Tensor)

                loss_nerf = (loss * loss_weight + torch.stack(reg_losses, dim=0).sum(dim=0)) / batch_size
                if not torch.isfinite(loss_nerf):
                    warnings.warn(f"loss_nerf is not finite: {loss_nerf}", RuntimeWarning)
                    loss_nerf = torch.nan_to_num(loss_nerf)

            if enable_autocast:
                grad_scaler.scale(loss_nerf).backward() # type: ignore
                grad_scaler.step(optimizer_nerf)
            else:
                loss_nerf.backward()
                optimizer_nerf.step()

            optimizer_nerf.zero_grad()
            reg_losses = [reg_loss.item() for reg_loss in reg_losses]

            if step % 100 == 0 or step == len(diffusion_schedule) - 1:
                with torch.no_grad():
                    if noise_pred_z123 is not None:
                        x0_base: Tensor = scheduler.step(noise_pred_z123, t_nerf, latents_selected_noised).pred_original_sample # [B, 4, 64, 64] # type: ignore
                        x0_base = vae.decode(1 / vae.config['scaling_factor'] * x0_base).sample # type: ignore
                        self.set_output(f"PredBaseX0", (x0_base / 2 + 0.5).clamp(0, 1))
                    if latents_selected_noised is not None:
                        x0: Tensor = vae.decode(1 / vae.config['scaling_factor'] * latents_selected_noised).sample # type: ignore
                        self.set_output(f"NoisedX0", (x0 / 2 + 0.5).clamp(0, 1))
                    for i, latent in enumerate(latents):
                        x0: Tensor = vae.decode(1 / vae.config['scaling_factor'] * latent).sample # type: ignore
                        self.set_output(f"Truth_{i}", (x0 / 2 + 0.5).clamp(0, 1))
                self.set_output("log", f"{step}/{len(diffusion_schedule)} - T{t_nerf:03d}/{scheduler.timesteps[0].item():04d} [SDS] (NeRF: {'|'.join([f'{loss_value:.4f}' for loss_value in ([loss_nerf.item()] + reg_losses)])})")
                self.set_output("progress", int(100 * step / len(diffusion_schedule)))
                self.send_update()

            # TODO: add NeRF panel update

            if enable_autocast:
                grad_scaler.update()

        return self.ReturnDict(
            field=field,
            renderers=renderers,
            optimizer_nerf=optimizer_nerf,
            grad_scaler=grad_scaler,
        )


class RunMVDreamNeRF(BaseNode):
    """Generating 3D NeRF using MVDream method.
    """

    @KatzukiNode(
        node_type="nerf.basic.run_mvdream_nerf",
        author="Koke_Cacao",
        author_link="https://kokecacao.me",
        input_description={
            "h_img": "The resolution of NeRF output (pixel space) image feeding to diffusion. [h_img * latent_nerf_scale] must be diffusion's image scale.",
            "w_img": "The resolution of NeRF output (pixel space) image feeding to diffusion. [w_img * latent_nerf_scale] must be diffusion's image scale.",
            "cfg": "Classifier Free Guidance (CFG) parameter. Default is 50.0.",
            "n_views": "Number of views MVDream generates. Default is 4 due to only 4-view model is released.",
            "latent_nerf_scale": "How much to resize on NeRF output (pixel space) image before feeding to diffusion. [h_img * latent_nerf_scale] must be diffusion's image scale.",
            "reconstruction_loss": "Whether to use reconstruction loss (x0 prediction loss) or standard SDS loss. Default is reconstruction loss.",
            "cfg_rescale": "Ratio to apply normalization for reconstruction loss. 0 to disable. Default is 0.5.",
        },
        output_description={
            "field": "Output NeRF field.",
            "renderers": "Output NeRF renderers, since renderers themselves store background parameters.",
            "optimizer_nerf": "Output NeRF optimizer, since optimizer have states.",
            "grad_scaler": "Output NeRF grad scaler, since grad scaler have states.",
        },
        signal_to_default_data={
            "break": "break", # TODO: add signals to other nodes (for and while loop)
        })
    def __init__(self) -> None:
        pass

    class ReturnDict(TypedDict):
        field: FieldBase
        renderers: List[Renderer]
        optimizer_nerf: torch.optim.Optimizer
        grad_scaler: GradScaler

    def __del__(self) -> None:
        # clean up socket.io handlers
        if f"{self._node_id}_camera" in sio.handlers['/']:
            sio.handlers['/'].pop(f"{self._node_id}_camera")
        return super().__del__()

    def execute(
            self,
            scheduler: DDIMScheduler, # WARNING: check range
            unet: MultiViewUNetWrapperModel,
            vae: AutoencoderKL,
            field: FieldBase,
            diffusion_schedule: _DiffusionSchedule,
            renderers: List[Renderer],
            optimizer_nerf: torch.optim.Optimizer,
            prompt_embeds_pos: Union[Tensor, List[Tensor]], # [V, 77, 1024]
            prompt_embeds_neg: Union[Tensor, List[Tensor]], # [V, 77, 1024]
            dataset: _NeRFDataset = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
            h_img: int = 256,
            w_img: int = 256,
            cfg: float = 50.0,
            batch_size: int = 1,
            diffusion_steps: int = 1000,
            n_views: int = 4,
            latent_nerf_scale: float = 1.0,
            reconstruction_loss: bool = True,
            cfg_rescale: float = 0.5,
            enable_autocast: bool = True,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cuda"),
    ) -> ReturnDict:
        if dataset is None:
            dataset = NeRFDataset().execute() # type: ignore (wait for https://peps.python.org/pep-0671/)

        # correctly format input tensors to [77, 768] x4
        if isinstance(prompt_embeds_pos, Tensor):
            if prompt_embeds_pos.shape[0] == n_views:
                prompt_embeds_pos = prompt_embeds_pos.unbind(0)
            else:
                prompt_embeds_pos = [prompt_embeds_pos] * n_views
        if isinstance(prompt_embeds_neg, Tensor):
            if prompt_embeds_neg.shape[0] == n_views:
                prompt_embeds_neg = prompt_embeds_neg.unbind(0)
            else:
                prompt_embeds_neg = [prompt_embeds_neg] * n_views
        if len(prompt_embeds_pos) == 1 and len(prompt_embeds_neg) == 1:
            prompt_embeds_pos = prompt_embeds_pos * n_views
            prompt_embeds_neg = prompt_embeds_neg * n_views
        if len(prompt_embeds_pos) != n_views or len(prompt_embeds_neg) != n_views:
            raise ValueError(f"prompt_embeds_pos and prompt_embeds_neg must be of length {n_views}")
        if prompt_embeds_pos[0].shape[0] != prompt_embeds_neg[0].shape[0]:
            raise ValueError(f"prompt_embeds_pos and prompt_embeds_neg must have the same batch size")
        if prompt_embeds_pos[0].shape[0] == 1:
            prompt_embeds_pos = [p.squeeze(0) for p in prompt_embeds_pos]
            prompt_embeds_neg = [p.squeeze(0) for p in prompt_embeds_neg]

        render_variable = {
            "data": {
                "camera": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 5, 1],
                "fov": 75.0,
                "focal": 15.204296016480733,
                "near": 0.001,
                "far": 1000,
            }
        }

        def camera(render_variable: Dict):

            async def f(sid: str, data: int):
                if sid in variable.sid2uuid and variable.sid2uuid[sid] == self._uuid:
                    render_variable["data"] = data

            return f

        sio.on(f"{self._node_id}_camera", camera(render_variable=render_variable))

        grad_scaler = GradScaler(enabled=enable_autocast)
        scheduler.set_timesteps(diffusion_steps, device=device)

        diffusion_schedule = diffusion_schedule.to_device(device=device)
        pbar = tqdm(iterable=diffusion_schedule, total=len(diffusion_schedule))

        render_state = {
            "start_timestamp": time.time(),
            "start_render_variable": None,
            "updated": False,
        }
        sid: str = self._sid

        def update(field: FieldBase, render_variables: Dict[str, Any], timestamp: float):
            if should_update(
                    target_fps=24,
                    start_timestamp=render_state["start_timestamp"],
                    start_render_variable=render_state["start_render_variable"],
                    sid=sid,
                    current_timestamp=timestamp,
                    current_render_variable=render_variables,
            ):
                with torch.no_grad():
                    try:
                        ray_bundle = dataset.get_eval_ray_bundle(
                            h_latent=h_img,
                            w_latent=w_img,
                            c2w=torch.tensor(render_variables["camera"], dtype=dtype, device=device).reshape(4, 4).t(),
                            fov_y=render_variables["fov"] / 180 * np.pi,
                        )
                        ray_bundle = ray_bundle.to_device(device=device, dtype=dtype)
                        # TODO: violate one render per update
                        renders: Tensor = field.get_image(ray_bundle=ray_bundle, vae=vae, renderers=[renderers[0]], nerf_scale=latent_nerf_scale) # [B, 3, H, W], in [0, 1]
                        renders = renders[0, ...] # [B, C, H, W]

                        image_buffer = io.BytesIO()
                        pil_image = Image.fromarray(np.uint8(renders.squeeze().permute(1, 2, 0).cpu().numpy() * 256))
                        pil_image.save(image_buffer, format='PNG')
                    except Exception as e:
                        print(e) # possibly CUDA OOM
                        return

                render_state["start_render_variable"] = render_variables
                render_state["start_timestamp"] = timestamp

                async def send():
                    await sio.emit(f"{self._node_id}_render", {
                        "image": f"data:image/png;base64,{base64.b64encode(image_buffer.getvalue()).decode('utf-8')}",
                    }, room=variable.uuid2sid[self._uuid])

                asyncio.run_coroutine_threadsafe(send(), self._loop)
            else:
                render_state["updated"] = False

        for step, (t_nerf, loss_weight) in enumerate(pbar):
            signal = self.check_execution_state_change(clear_signal_event=lambda signal: signal == "break")
            if signal == "break":
                break

            ray_bundle = dataset.get_train_ray_bundle(
                h_latent_dataset=h_img, # WARNING: h_latent_dataset naming is misleading due to historial reasons
                w_latent_dataset=w_img,
                cx_latent_dataset=w_img // 2,
                cy_latent_dataset=h_img // 2,
                batch_size=batch_size * n_views,
                mv_dream_views=n_views,
            )
            ray_bundle = ray_bundle.to_device(device=device, dtype=dtype)
            c2w = convert_opengl_to_blender(ray_bundle.c2w) # [B, 4, 4]

            with autocast(enabled=enable_autocast, dtype=torch.float16):
                latents, reg_losses = field.get_latent(ray_bundle=ray_bundle, vae=vae, renderers=renderers, nerf_scale=latent_nerf_scale) # [R, B, C, H, W]
                latents_selected = latents[0, ...] # [B, C, H, W]
                noise_nerf = torch.randn_like(latents_selected) # [B, C, H, W]
                latents_selected_noised = scheduler.add_noise(latents_selected, noise_nerf, t_nerf) # [B, C, H, W] # type: ignore

                with torch.no_grad():
                    noise_pred_base, noise_pred_base_x0 = predict_noise_mvdream(
                        unet_mvdream=unet,
                        latents_noised=latents_selected_noised, # [4, 4, 32, 32]
                        text_embeddings_conditional=prompt_embeds_pos * batch_size, # [77, 768] x4 -> [77, 768] x4xB
                        text_embeddings_unconditional=prompt_embeds_neg * batch_size, # [77, 768] x4 -> [77, 768] x4xB
                        camera_embeddings=c2w.reshape(batch_size * n_views, 16), # [4, 4, 4] -> [4, 16]
                        cfg=cfg,
                        t=t_nerf,
                        scheduler=scheduler,
                        n_views=n_views,
                        reconstruction_loss=reconstruction_loss,
                        cfg_rescale=cfg_rescale,
                    )
                if noise_pred_base_x0 is None:
                    loss = SpecifyGradient.apply(latents_selected_noised, noise_pred_base - noise_nerf)
                else:
                    loss = 0.5 * torch.nn.functional.mse_loss(latents_selected, noise_pred_base_x0.detach())
                assert isinstance(loss, Tensor)

                loss_nerf = (loss * loss_weight + torch.stack(reg_losses, dim=0).sum(dim=0)) / batch_size
                if not torch.isfinite(loss_nerf):
                    warnings.warn(f"loss_nerf is not finite: {loss_nerf}", RuntimeWarning)
                    loss_nerf = torch.nan_to_num(loss_nerf)

            if enable_autocast:
                grad_scaler.scale(loss_nerf).backward() # type: ignore
                grad_scaler.step(optimizer_nerf)
            else:
                loss_nerf.backward()
                optimizer_nerf.step()

            optimizer_nerf.zero_grad()
            reg_losses = [reg_loss.item() for reg_loss in reg_losses]

            if step % 100 == 0 or step == len(diffusion_schedule) - 1:
                with torch.no_grad():
                    if noise_pred_base is not None:
                        x0_base: Tensor = scheduler.step(noise_pred_base, t_nerf, latents_selected_noised).pred_original_sample # [B, 4, 64, 64] # type: ignore
                        x0_base = vae.decode(1 / vae.config['scaling_factor'] * x0_base).sample # type: ignore
                        self.set_output(f"PredBaseX0", (x0_base / 2 + 0.5).clamp(0, 1))
                    if latents_selected_noised is not None:
                        x0: Tensor = vae.decode(1 / vae.config['scaling_factor'] * latents_selected_noised).sample # type: ignore
                        self.set_output(f"NoisedX0", (x0 / 2 + 0.5).clamp(0, 1))
                    for i, latent in enumerate(latents):
                        x0: Tensor = vae.decode(1 / vae.config['scaling_factor'] * latent).sample # type: ignore
                        self.set_output(f"Truth_{i}", (x0 / 2 + 0.5).clamp(0, 1))
                self.set_output("log", f"{step}/{len(diffusion_schedule)} - T{t_nerf:03d}/{scheduler.timesteps[0].item():04d} [SDS] (NeRF: {'|'.join([f'{loss_value:.4f}' for loss_value in ([loss_nerf.item()] + reg_losses)])})")
                self.set_output("progress", int(100 * step / len(diffusion_schedule)))
                render_state["updated"] = True

            update(field=field, render_variables=render_variable["data"], timestamp=time.time())

            if render_state["updated"]:
                self.send_update()
                render_state["updated"] = False

            if enable_autocast:
                grad_scaler.update()

        return self.ReturnDict(
            field=field,
            renderers=renderers,
            optimizer_nerf=optimizer_nerf,
            grad_scaler=grad_scaler,
        )


class MVDreamLossGenerator(BaseNode):

    @KatzukiNode(
        node_type="nerf.basic.loss_generator.mvdream_loss_generator",
        author="Koke_Cacao",
        author_link="https://kokecacao.me",
    )
    def __init__(self) -> None:
        pass

    def execute(
            self,
            scheduler: DDIMScheduler, # WARNING: check range
            unet: MultiViewUNetWrapperModel,
            vae: AutoencoderKL,
            renderers: List[Renderer],
            prompt_embeds_pos: Union[Tensor, List[Tensor]], # [V, 77, 1024]
            prompt_embeds_neg: Union[Tensor, List[Tensor]], # [V, 77, 1024]
            dataset: _NeRFDataset = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
            h_img: int = 256,
            w_img: int = 256,
            cfg: float = 50.0,
            batch_size: int = 1,
            diffusion_steps: int = 1000,
            n_views: int = 4,
            latent_nerf_scale: float = 1.0,
            reconstruction_loss: bool = True,
            cfg_rescale: float = 0.5,
            enable_autocast: bool = True,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cuda"),
    ) -> Callable:
        if dataset is None:
            dataset = NeRFDataset().execute() # type: ignore (wait for https://peps.python.org/pep-0671/)

        # correctly format input tensors to [77, 768] x4
        if isinstance(prompt_embeds_pos, Tensor):
            if prompt_embeds_pos.shape[0] == n_views:
                prompt_embeds_pos = prompt_embeds_pos.unbind(0)
            else:
                prompt_embeds_pos = [prompt_embeds_pos] * n_views
        if isinstance(prompt_embeds_neg, Tensor):
            if prompt_embeds_neg.shape[0] == n_views:
                prompt_embeds_neg = prompt_embeds_neg.unbind(0)
            else:
                prompt_embeds_neg = [prompt_embeds_neg] * n_views
        if len(prompt_embeds_pos) == 1 and len(prompt_embeds_neg) == 1:
            prompt_embeds_pos = prompt_embeds_pos * n_views
            prompt_embeds_neg = prompt_embeds_neg * n_views
        if len(prompt_embeds_pos) != n_views or len(prompt_embeds_neg) != n_views:
            raise ValueError(f"prompt_embeds_pos and prompt_embeds_neg must be of length {n_views}")
        if prompt_embeds_pos[0].shape[0] != prompt_embeds_neg[0].shape[0]:
            raise ValueError(f"prompt_embeds_pos and prompt_embeds_neg must have the same batch size")
        if prompt_embeds_pos[0].shape[0] == 1:
            prompt_embeds_pos = [p.squeeze(0) for p in prompt_embeds_pos]
            prompt_embeds_neg = [p.squeeze(0) for p in prompt_embeds_neg]

        scheduler.set_timesteps(diffusion_steps, device=device)

        def fn(field: FieldBase, step: int, t_nerf: Tensor, loss_weight: Tensor, render_state: Dict[str, Any], sid: str):

            ray_bundle = dataset.get_train_ray_bundle(
                h_latent_dataset=h_img, # WARNING: h_latent_dataset naming is misleading due to historial reasons
                w_latent_dataset=w_img,
                cx_latent_dataset=w_img // 2,
                cy_latent_dataset=h_img // 2,
                batch_size=batch_size * n_views,
                mv_dream_views=n_views,
            )
            ray_bundle = ray_bundle.to_device(device=device, dtype=dtype)
            c2w = convert_opengl_to_blender(ray_bundle.c2w) # [B, 4, 4]

            with autocast(enabled=enable_autocast, dtype=torch.float16):
                latents, reg_losses = field.get_latent(ray_bundle=ray_bundle, vae=vae, renderers=renderers, nerf_scale=latent_nerf_scale) # [R, B, C, H, W]
                latents_selected = latents[0, ...] # [B, C, H, W]
                noise_nerf = torch.randn_like(latents_selected) # [B, C, H, W]
                latents_selected_noised = scheduler.add_noise(latents_selected, noise_nerf, t_nerf) # [B, C, H, W] # type: ignore

                with torch.no_grad():
                    noise_pred_base, noise_pred_base_x0 = predict_noise_mvdream(
                        unet_mvdream=unet,
                        latents_noised=latents_selected_noised, # [4, 4, 32, 32]
                        text_embeddings_conditional=prompt_embeds_pos * batch_size, # [77, 768] x4 -> [77, 768] x4xB
                        text_embeddings_unconditional=prompt_embeds_neg * batch_size, # [77, 768] x4 -> [77, 768] x4xB
                        camera_embeddings=c2w.reshape(batch_size * n_views, 16), # [4, 4, 4] -> [4, 16]
                        cfg=cfg,
                        t=t_nerf,
                        scheduler=scheduler,
                        n_views=n_views,
                        reconstruction_loss=reconstruction_loss,
                        cfg_rescale=cfg_rescale,
                    )
                if noise_pred_base_x0 is None:
                    loss = SpecifyGradient.apply(latents_selected_noised, noise_pred_base - noise_nerf)
                else:
                    loss = 0.5 * torch.nn.functional.mse_loss(latents_selected, noise_pred_base_x0.detach())
                assert isinstance(loss, Tensor)

                loss_nerf = (loss * loss_weight + torch.stack(reg_losses, dim=0).sum(dim=0)) / batch_size
                return loss_nerf

        return fn


class TrainNeRF(BaseNode):

    @KatzukiNode(
        node_type="nerf.basic.train_nerf",
        author="Koke_Cacao",
        author_link="https://kokecacao.me",
        signal_to_default_data={
            "break": "break",
        },
    )
    def __init__(self) -> None:
        pass

    class ReturnDict(TypedDict):
        field: FieldBase
        optimizer_nerf: torch.optim.Optimizer
        grad_scaler: GradScaler

    def __del__(self) -> None:
        # clean up socket.io handlers
        if f"{self._node_id}_camera" in sio.handlers['/']:
            sio.handlers['/'].pop(f"{self._node_id}_camera")
        return super().__del__()

    def execute(
            self,
            field: FieldBase,
            diffusion_schedule: _DiffusionSchedule,
            optimizer_nerf: torch.optim.Optimizer,
            loss_generators: List[Callable],
            loss_generators_chances: List[float],
            enable_autocast: bool = True,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cuda"),
    ) -> ReturnDict:

        render_variable = {
            "data": {
                "camera": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 5, 1],
                "fov": 75.0,
                "focal": 15.204296016480733,
                "near": 0.001,
                "far": 1000,
            }
        }

        def camera(render_variable: Dict):

            async def f(sid: str, data: int):
                if sid in variable.sid2uuid and variable.sid2uuid[sid] == self._uuid:
                    render_variable["data"] = data

            return f

        sio.on(f"{self._node_id}_camera", camera(render_variable=render_variable))

        grad_scaler = GradScaler(enabled=enable_autocast)

        diffusion_schedule = diffusion_schedule.to_device(device=device)
        pbar = tqdm(iterable=diffusion_schedule, total=len(diffusion_schedule))

        render_state = {
            "start_timestamp": time.time(),
            "start_render_variable": None,
            "updated": False,
        }
        sid: str = self._sid

        # renderers and dataset for viewing
        dataset: _NeRFDataset = NeRFDataset().execute() # type: ignore (wait for https://peps.python.org/pep-0671/)
        renderer: _RGBRenderer = RGBRenderer().execute(background_color='black') # type: ignore (wait for https://peps.python.org/pep-0671/)
        latent_nerf_scale: float = 1.0
        h_img: int = 256
        w_img: int = 256

        def update(field: FieldBase, render_variables: Dict[str, Any], timestamp: float):
            if should_update(
                    target_fps=24,
                    start_timestamp=render_state["start_timestamp"],
                    start_render_variable=render_state["start_render_variable"],
                    sid=sid,
                    current_timestamp=timestamp,
                    current_render_variable=render_variables,
            ):
                with torch.no_grad():
                    try:
                        ray_bundle = dataset.get_eval_ray_bundle(
                            h_latent=h_img,
                            w_latent=w_img,
                            c2w=torch.tensor(render_variables["camera"], dtype=dtype, device=device).reshape(4, 4).t(),
                            fov_y=render_variables["fov"] / 180 * np.pi,
                        )
                        ray_bundle = ray_bundle.to_device(device=device, dtype=dtype)
                        # TODO: violate one render per update
                        renders: Tensor = field.get_image(ray_bundle=ray_bundle, vae=None, renderers=[renderer], nerf_scale=latent_nerf_scale) # [B, 3, H, W], in [0, 1]
                        renders = renders[0, ...] # [B, C, H, W]

                        image_buffer = io.BytesIO()
                        pil_image = Image.fromarray(np.uint8(renders.squeeze().permute(1, 2, 0).cpu().numpy() * 256))
                        pil_image.save(image_buffer, format='PNG')
                    except Exception as e:
                        print(e) # possibly CUDA OOM
                        return

                render_state["start_render_variable"] = render_variables
                render_state["start_timestamp"] = timestamp

                async def send():
                    await sio.emit(f"{self._node_id}_render", {
                        "image": f"data:image/png;base64,{base64.b64encode(image_buffer.getvalue()).decode('utf-8')}",
                    }, room=variable.uuid2sid[self._uuid])

                asyncio.run_coroutine_threadsafe(send(), self._loop)
            else:
                render_state["updated"] = False

        # exclude zero or negative chance
        idx_zero = [i for i, chance in enumerate(loss_generators_chances) if chance <= 0]
        if len(idx_zero) > 0:
            warnings.warn(f"loss_generators_chances[{idx_zero}] <= 0, will be excluded", RuntimeWarning)
            loss_generators = [loss for i, loss in enumerate(loss_generators) if i not in idx_zero]
            loss_generators_chances = [chance for i, chance in enumerate(loss_generators_chances) if i not in idx_zero]

        # normalize chance
        loss_generators_chances_sum = sum(loss_generators_chances)
        loss_generators_chances = [chance / loss_generators_chances_sum for chance in loss_generators_chances]

        # for counting
        loss_generators_expected = [len(diffusion_schedule) * chance for chance in loss_generators_chances]
        loss_generators_current = [0 for _ in loss_generators_chances]

        for step, (t_nerf, loss_weight) in enumerate(pbar):
            signal = self.check_execution_state_change(clear_signal_event=lambda signal: signal == "break")
            if signal == "break":
                break

            loss_generators_current_div_expected = [current / expected for current, expected in zip(loss_generators_current, loss_generators_expected)]
            lowest_div_idx = loss_generators_current_div_expected.index(min(loss_generators_current_div_expected))
            choice = random.choices(range(len(loss_generators_chances)), weights=loss_generators_chances, k=1)[0]

            # select lowest_div_idx with 10% chance and choice with 90% chance
            if random.random() < 0.1:
                choice = lowest_div_idx

            loss_nerf = loss_generators[choice](
                field=field,
                step=step,
                t_nerf=t_nerf,
                loss_weight=loss_weight,
                render_state=render_state,
                sid=sid,
            )

            # run losses
            if not torch.isfinite(loss_nerf):
                warnings.warn(f"loss_nerf is not finite: {loss_nerf}", RuntimeWarning)
                loss_nerf = torch.nan_to_num(loss_nerf)

            if enable_autocast:
                grad_scaler.scale(loss_nerf).backward() # type: ignore
                grad_scaler.step(optimizer_nerf)
            else:
                loss_nerf.backward()
                optimizer_nerf.step()

            optimizer_nerf.zero_grad()

            update(field=field, render_variables=render_variable["data"], timestamp=time.time())
            if render_state["updated"]:
                self.send_update()
                render_state["updated"] = False

            if enable_autocast:
                grad_scaler.update()

        return self.ReturnDict(
            field=field,
            optimizer_nerf=optimizer_nerf,
            grad_scaler=grad_scaler,
        )


class RenderNeRF(BaseNode):
    """Render NeRF using specified field.
    """

    @KatzukiNode(
        node_type="nerf.basic.render_nerf",
        author="Koke_Cacao",
        author_link="https://kokecacao.me",
        input_description={
            "h_img": "The resolution of NeRF output (pixel space) image feeding to diffusion. [h_img * latent_nerf_scale] must be diffusion's image scale.",
            "w_img": "The resolution of NeRF output (pixel space) image feeding to diffusion. [w_img * latent_nerf_scale] must be diffusion's image scale.",
            "latent_nerf_scale": "How much to resize on NeRF output (pixel space) image before feeding to diffusion. [h_img * latent_nerf_scale] must be diffusion's image scale.",
        },
        signal_to_default_data={
            "break": "break",
        },
    )
    def __init__(self) -> None:
        pass

    def __del__(self) -> None:
        # clean up socket.io handlers
        if f"{self._node_id}_camera" in sio.handlers['/']:
            sio.handlers['/'].pop(f"{self._node_id}_camera")
        return super().__del__()

    def execute(
            self,
            field: FieldBase,
            renderers: List[Renderer],
            dataset: _NeRFDataset = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
            h_img: int = 256,
            w_img: int = 256,
            latent_nerf_scale: float = 1.0,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cuda"),
    ) -> None:

        render_variable = {
            "data": {
                "camera": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 5, 1],
                "fov": 75.0,
                "focal": 15.204296016480733,
                "near": 0.001,
                "far": 1000,
            }
        }

        def camera(render_variable: Dict):

            async def f(sid: str, data: int):
                if sid in variable.sid2uuid and variable.sid2uuid[sid] == self._uuid:
                    render_variable["data"] = data

            return f

        sio.on(f"{self._node_id}_camera", camera(render_variable=render_variable))

        with torch.no_grad():
            if dataset is None:
                dataset = NeRFDataset().execute() # type: ignore (wait for https://peps.python.org/pep-0671/)

            render_state = {
                "start_timestamp": time.time(),
                "start_render_variable": None,
                "updated": False,
            }
            sid: str = self._sid

            def update(field: FieldBase, render_variables: Dict[str, Any], timestamp: float):
                if should_update(
                        target_fps=24,
                        start_timestamp=render_state["start_timestamp"],
                        start_render_variable=render_state["start_render_variable"],
                        sid=sid,
                        current_timestamp=timestamp,
                        current_render_variable=render_variables,
                ):
                    ray_bundle = dataset.get_eval_ray_bundle(
                        h_latent=h_img,
                        w_latent=w_img,
                        c2w=torch.tensor(render_variables["camera"], dtype=dtype, device=device).reshape(4, 4).t(),
                        fov_y=render_variables["fov"] / 180 * np.pi,
                    )
                    ray_bundle = ray_bundle.to_device(device=device, dtype=dtype)
                    # TODO: violate one render per update
                    renders: Tensor = field.get_image(ray_bundle=ray_bundle, vae=None, renderers=[renderers[0]], nerf_scale=latent_nerf_scale) # [B, 3, H, W] # in [0, 1]
                    renders = renders[0, ...] # [B, C, H, W]

                    image_buffer = io.BytesIO()
                    pil_image = Image.fromarray(np.uint8(renders.squeeze().permute(1, 2, 0).cpu().numpy() * 256))
                    pil_image.save(image_buffer, format='PNG')

                    render_state["start_render_variable"] = render_variables
                    render_state["start_timestamp"] = timestamp

                    async def send():
                        await sio.emit(f"{self._node_id}_render", {
                            "image": f"data:image/png;base64,{base64.b64encode(image_buffer.getvalue()).decode('utf-8')}",
                        }, room=variable.uuid2sid[self._uuid])

                    asyncio.run_coroutine_threadsafe(send(), self._loop)
                else:
                    render_state["updated"] = False
                    time.sleep(1 / (24 * 2))

            step = 0
            while True:
                signal = self.check_execution_state_change(clear_signal_event=lambda signal: signal == "break")
                if signal == "break":
                    break

                self.set_output("log", f"{step}/inf")
                self.set_output("progress", int(100))
                step = step + 1
                update(field=field, render_variables=render_variable["data"], timestamp=time.time())

                if render_state["updated"]:
                    self.send_update()
                    render_state["updated"] = False
            return None
