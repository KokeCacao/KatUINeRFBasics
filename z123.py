import torch
import kornia
import time
import base64
import io
import asyncio
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch import Tensor
from typing import TypedDict, Optional, Union, Dict
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from transformers import CLIPVisionModelWithProjection
from kokikit.models import CCProjection
from kokikit.diffusion import predict_noise_z123
from kokikit.dataset import NeRFDataset
from backend.loader.decorator import KatzukiNode
from backend.nodes.builtin import BaseNode
from backend import variable
from backend.sio import sio

from KatUIDiffusionBasics.util import should_update


class Z123ModelLoader(BaseNode):

    @KatzukiNode(node_type="diffusion.z123.z123_model_loader")
    def __init__(self) -> None:
        pass

    class ReturnDict(TypedDict):
        unet_z123: UNet2DConditionModel
        image_encoder_z123: CLIPVisionModelWithProjection
        cc_projection_z123: CCProjection
        vae: AutoencoderKL

    def execute(self, z123_path: str = "kxic/zero123-xl", dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cuda")) -> ReturnDict:
        if "unet_z123" in self._connected_outputs:
            unet_z123: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(z123_path, subfolder="unet") # type: ignore
            unet_z123 = unet_z123.to(device=device)
        else:
            unet_z123 = None # type: ignore

        if "image_encoder_z123" in self._connected_outputs:
            image_encoder_z123: CLIPVisionModelWithProjection = CLIPVisionModelWithProjection.from_pretrained(
                z123_path,
                subfolder="image_encoder",
                device_map={'': 0} # BUG: https://github.com/tloen/alpaca-lora/issues/368 (NotImplementedError: Cannot copy out of meta tensor; no data!)
                # BUG: solution: https://huggingface.co/docs/transformers/main_classes/model
                ,
            ) # type: ignore
            image_encoder_z123 = image_encoder_z123.to(device=device) # type: ignore
        else:
            image_encoder_z123 = None # type: ignore

        if "cc_projection_z123" in self._connected_outputs:
            cc_projection_z123: CCProjection = CCProjection.from_pretrained(z123_path, subfolder="cc_projection") # type: ignore
            cc_projection_z123 = cc_projection_z123.to(device=device)
        else:
            cc_projection_z123 = None # type: ignore

        if "vae" in self._connected_outputs:
            vae: AutoencoderKL = AutoencoderKL.from_pretrained(z123_path, subfolder="vae") # type: ignore
            vae = vae.to(device=device, dtype=dtype)
        else:
            vae = None # type: ignore

        return self.ReturnDict(
            unet_z123=unet_z123,
            image_encoder_z123=image_encoder_z123,
            cc_projection_z123=cc_projection_z123,
            vae=vae,
        )


class Z123ImagePartialEmbedding(BaseNode):

    @KatzukiNode(node_type="diffusion.z123.z123_image_partial_embedding")
    def __init__(self) -> None:
        pass

    class ReturnDict(TypedDict):
        latents_image: Tensor
        image_embedding: Tensor

    def execute(self, image: Tensor, image_encoder_z123: CLIPVisionModelWithProjection, vae: AutoencoderKL, dtype: torch.dtype = torch.float32) -> ReturnDict:
        # encode image to latent
        image = torch.nn.functional.interpolate(image, size=(256, 256), mode='bicubic', align_corners=True) # [1, 3, 256, 256]
        latents_image = vae.encode(image).latent_dist.mode() # type: ignore
        # QUESTION: missing "vae.config['scaling_factor']"?
        # encode image to angle_embeddings_conditional
        image = kornia.geometry.resize(image, (224, 224), interpolation='bicubic', align_corners=True, antialias=False).to(dtype=dtype)
        image = image / 2 + 0.5 # in [0, 1]
        image = kornia.enhance.normalize(image, torch.Tensor([0.48145466, 0.4578275, 0.40821073]), torch.Tensor([0.26862954, 0.26130258, 0.27577711]))
        return self.ReturnDict(
            latents_image=latents_image, # [1, 4, 32, 32]
            image_embedding=image_encoder_z123(image).image_embeds, # [B, 768]
        )


class Z123Embedding(BaseNode):

    @KatzukiNode(node_type="diffusion.z123.z123_embedding")
    def __init__(self) -> None:
        pass

    class ReturnDict(TypedDict):
        angle_embeddings_conditional: Tensor
        angle_embeddings_unconditional: Tensor

    def execute(
        self,
        image_embedding: Tensor,
        cc_projection_z123: CCProjection,
        thetas: Tensor,
        phis: Tensor,
        zooms: Tensor,
        thetas_ref: Union[Tensor, float, int] = 0,
        phis_ref: Union[Tensor, float, int] = 0,
        zooms_ref: Union[Tensor, float, int] = 0,
    ) -> ReturnDict:
        # encode angle to angle_embeddings_conditional
        angle = torch.stack([
            thetas - thetas_ref,
            torch.sin(phis - phis_ref),
            torch.cos(phis - phis_ref),
            zooms - zooms_ref,
        ], dim=-1) # [B, 4]

        # combine image and angle to angle_embeddings_conditional
        angle_embeddings_conditional = torch.cat([image_embedding, angle], dim=-1) # [B, 772]
        angle_embeddings_conditional = cc_projection_z123(angle_embeddings_conditional) # [B, 768]

        # unconditional
        assert angle_embeddings_conditional is not None
        angle_embeddings_unconditional = torch.zeros_like(angle_embeddings_conditional) # [B, 768]

        return self.ReturnDict(
            angle_embeddings_conditional=angle_embeddings_conditional, # [B, 768]
            angle_embeddings_unconditional=angle_embeddings_unconditional, # [B, 768]
        )


class RunZ123(BaseNode):

    @KatzukiNode(node_type="diffusion.z123.run_z123")
    def __init__(self) -> None:
        pass

    def execute(
        self,
        latents_original: Tensor,
        latents_image: Tensor,
        angle_embeddings_conditional: Tensor,
        angle_embeddings_unconditional: Tensor,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        cfg: float = 3.0,
        vae: Optional[AutoencoderKL] = None,
        diffusion_steps: int = 50,
        reconstruction_loss: bool = True,
        cfg_rescale: float = 0.5,
    ) -> Tensor:
        device = latents_original.device
        scheduler.set_timesteps(diffusion_steps, device=device)

        with torch.no_grad():
            pbar = tqdm(scheduler.timesteps - 1)
            latents_noised = latents_original
            for step, time in enumerate(pbar):
                self.check_execution_state_change(clear_signal_event=lambda x: False)

                assert latents_original.shape[-2] == 32 and latents_original.shape[-1] == 32, "WARNING: z123 only support 32x32 latent for now"
                assert latents_image.shape[-2] == 32 and latents_image.shape[-1] == 32, "WARNING: z123 only support 32x32 latent for now"
                latents_noised = latents_original
                noise_pred, noise_pred_x0 = predict_noise_z123(
                    unet_z123=unet,
                    latents_noised=latents_noised, # [B, C, 256, 256]
                    latents_image=latents_image, # [1, 4, 32, 32]
                    angle_embeddings_conditional=angle_embeddings_conditional.unsqueeze(-2), # [B, 768] -> [B, 1, 768]
                    angle_embeddings_unconditional=angle_embeddings_unconditional.unsqueeze(-2), # [B, 768] -> [B, 1, 768]
                    cfg=cfg,
                    lora_scale=0,
                    t=time,
                    scheduler=scheduler,
                    reconstruction_loss=reconstruction_loss,
                    cfg_rescale=cfg_rescale,
                )
                latents_noised = scheduler.step(noise_pred, time, latents_noised).prev_sample # type: ignore
                assert noise_pred.shape[-2] == 32 and noise_pred.shape[-1] == 32, "WARNING: z123 only support 32x32 latent for now"
                latents_original = latents_noised

                if step % 10 == 0 or step == len(pbar) - 1:
                    if vae is not None:
                        _ = latents_noised if noise_pred_x0 is None else noise_pred_x0 # use x0 instead of actual image if available
                        image_batch = vae.decode(1 / vae.config['scaling_factor'] * _.clone().detach()).sample # [B, C, H, W] # type: ignore
                        image_batch = (image_batch / 2 + 0.5).clamp(0, 1) * 255
                        image_batch = image_batch.permute(0, 2, 3, 1).detach().cpu() # [B, C, H, W] -> [B, H, W, C]
                        self.set_output("latents_noised", image_batch)
                    self.set_output("progress", int(100 * step / len(pbar)))
                    self.send_update()
            return latents_noised


class RunZ123WhileLoop(BaseNode):

    @KatzukiNode(node_type="diffusion.z123.run_z123_while_loop")
    def __init__(self) -> None:
        pass

    def __del__(self) -> None:
        # clean up socket.io handlers
        if f"{self._node_id}_camera" in sio.handlers['/']:
            sio.handlers['/'].pop(f"{self._node_id}_camera")
        return super().__del__()

    def execute(
        self,
        latents_image: Tensor,
        image_embedding: Tensor,
        cc_projection_z123: CCProjection,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: DDIMScheduler,
        image_height: int = 256,
        image_width: int = 256,
        batch_size: int = 1,
        channel: int = 4,
        vae_scale: int = 8,
        cfg: float = 3.0,
        diffusion_steps: int = 50,
        reconstruction_loss: bool = True,
        cfg_rescale: float = 0.5,
    ) -> Tensor:
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
            device = latents_image.device
            dtype = latents_image.dtype
            scheduler.set_timesteps(diffusion_steps, device=device)
            render_state = {
                "start_timestamp": time.time(),
                "start_render_variable": None,
            }
            sid: str = self._sid

            while True:
                latents_original = torch.randn(batch_size, channel, image_height // vae_scale, image_width // vae_scale, dtype=dtype, device=device)
                current_render_variable = render_variable["data"]
                if should_update(
                        target_fps=24,
                        start_timestamp=render_state["start_timestamp"],
                        start_render_variable=render_state["start_render_variable"],
                        sid=sid,
                        current_timestamp=render_state["start_timestamp"], # disable time
                        current_render_variable=current_render_variable,
                ):
                    render_state["start_render_variable"] = current_render_variable
                    # encode angle to angle_embeddings_conditional
                    thetas, phis, r = NeRFDataset._get_thetas_phis_r(c2w=torch.tensor(current_render_variable["camera"], dtype=dtype, device=device).reshape(4, 4).t().unsqueeze(0))
                    angle = torch.stack([
                        thetas - 0,
                        torch.sin(phis - 0),
                        torch.cos(phis - 0),
                        r - 1.0,
                    ], dim=-1) # [B, 4]

                    # combine image and angle to angle_embeddings_conditional
                    angle_embeddings_conditional = torch.cat([image_embedding, angle], dim=-1) # [B, 772]
                    angle_embeddings_conditional = cc_projection_z123(angle_embeddings_conditional) # [B, 768]

                    # unconditional
                    assert angle_embeddings_conditional is not None
                    angle_embeddings_unconditional = torch.zeros_like(angle_embeddings_conditional) # [B, 768]

                    with torch.no_grad():
                        pbar = tqdm(scheduler.timesteps - 1)
                        latents_noised = latents_original
                        for step, t in enumerate(pbar):
                            self.check_execution_state_change(clear_signal_event=lambda x: False)

                            # start over if viewer changed camera view angle
                            current_render_variable = render_variable["data"]
                            if should_update(
                                    target_fps=24,
                                    start_timestamp=render_state["start_timestamp"],
                                    start_render_variable=render_state["start_render_variable"],
                                    sid=sid,
                                    current_timestamp=render_state["start_timestamp"], # disable time
                                    current_render_variable=current_render_variable,
                            ):
                                # don't update start_render_variable here since we want to enter the loop again
                                break
                            else:
                                render_state["start_render_variable"] = current_render_variable

                            assert latents_original.shape[-2] == 32 and latents_original.shape[-1] == 32, "WARNING: z123 only support 32x32 latent for now"
                            assert latents_image.shape[-2] == 32 and latents_image.shape[-1] == 32, "WARNING: z123 only support 32x32 latent for now"
                            latents_noised = latents_original
                            noise_pred, noise_pred_x0 = predict_noise_z123(
                                unet_z123=unet,
                                latents_noised=latents_noised, # [B, C, 256, 256]
                                latents_image=latents_image, # [1, 4, 32, 32]
                                angle_embeddings_conditional=angle_embeddings_conditional.unsqueeze(-2), # [B, 768] -> [B, 1, 768]
                                angle_embeddings_unconditional=angle_embeddings_unconditional.unsqueeze(-2), # [B, 768] -> [B, 1, 768]
                                cfg=cfg,
                                lora_scale=0,
                                t=t,
                                scheduler=scheduler,
                                reconstruction_loss=reconstruction_loss,
                                cfg_rescale=cfg_rescale,
                            )
                            latents_noised = scheduler.step(noise_pred, t, latents_noised).prev_sample # type: ignore
                            assert noise_pred.shape[-2] == 32 and noise_pred.shape[-1] == 32, "WARNING: z123 only support 32x32 latent for now"
                            latents_original = latents_noised

                            if step % 10 == 0 or step == len(pbar) - 1:
                                _ = latents_noised if noise_pred_x0 is None else noise_pred_x0 # use x0 instead of actual image if available
                                image_batch = vae.decode(1 / vae.config['scaling_factor'] * _.clone().detach()).sample # [B, C, H, W] # type: ignore
                                image_batch = (image_batch / 2 + 0.5).clamp(0, 1) * 255
                                image_batch = image_batch.permute(0, 2, 3, 1).detach().cpu() # [B, C, H, W] -> [B, H, W, C]

                                image_buffer = io.BytesIO()
                                pil_image = Image.fromarray(np.uint8(image_batch[0].numpy()))
                                pil_image.save(image_buffer, format='PNG')

                                self.set_output("latents_noised", image_batch)
                                self.set_output("progress", int(100 * step / len(pbar)))

                                async def send():
                                    await sio.emit(
                                        f"{self._node_id}_render",
                                        {"image": f"data:image/png;base64,{base64.b64encode(image_buffer.getvalue()).decode('utf-8')}"},
                                        room=variable.uuid2sid[self._uuid],
                                    )

                                asyncio.run_coroutine_threadsafe(send(), self._loop)

                                self.send_update()
                else:
                    render_state["start_render_variable"] = current_render_variable
                    self.check_execution_state_change(clear_signal_event=lambda x: False)
                    time.sleep(1 / (24 * 2))
