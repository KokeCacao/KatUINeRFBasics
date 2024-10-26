import torch
from tqdm import tqdm
from torch import Tensor
from typing import List, Optional, Union, TypedDict
from backend.loader.decorator import KatzukiNode
from backend.nodes.builtin import BaseNode
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from kokikit.models import MultiViewUNetWrapperModel, get_camera
from kokikit.diffusion import predict_noise_mvdream

from KatUIDiffusionBasics.basic import SchedulerLoader, LatentImage


class MVDreamModelLoader(BaseNode):

    @KatzukiNode(node_type="diffusion.mvdream.mvdream_model_loader")
    def __init__(self) -> None:
        pass

    class ReturnDict(TypedDict):
        text_encoder: CLIPTextModel
        tokenizer: CLIPTokenizer
        unet: MultiViewUNetWrapperModel
        vae: AutoencoderKL

    def execute(self, mvdream_path: str = "KokeCacao/mvdream-hf", dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cuda")) -> ReturnDict:
        if "text_encoder" in self._connected_outputs:
            text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
                mvdream_path,
                subfolder='text_encoder',
                device_map={'': 0},
            ) # type: ignore
        else:
            text_encoder = None # type: ignore
            text_encoder = text_encoder.to(device=device, dtype=dtype) # type: ignore

        if "tokenizer" in self._connected_outputs:
            tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(mvdream_path, subfolder='tokenizer', use_fast=False)
        else:
            tokenizer = None # type: ignore

        if "unet" in self._connected_outputs:
            unet_mvdream: MultiViewUNetWrapperModel = MultiViewUNetWrapperModel.from_pretrained(mvdream_path, subfolder="unet") # type: ignore
            unet_mvdream = unet_mvdream.to(device=device)
        else:
            unet_mvdream = None # type: ignore

        if "vae" in self._connected_outputs:
            vae: AutoencoderKL = AutoencoderKL.from_pretrained(mvdream_path, subfolder="vae") # type: ignore
            vae = vae.to(device=device, dtype=dtype)
        else:
            vae = None # type: ignore

        return self.ReturnDict(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet_mvdream,
            vae=vae,
        )


class MVCreamCameraEmbedding(BaseNode):

    @KatzukiNode(node_type="diffusion.mvdream.mvdream_camera_embedding")
    def __init__(self) -> None:
        pass

    def execute(self, batch_size: int = 4, device: torch.device = torch.device("cuda")) -> Tensor:
        return get_camera(batch_size).to(device=device)


class RunMVDream(BaseNode):

    @KatzukiNode(node_type="diffusion.mvdream.run_mvdream")
    def __init__(self) -> None:
        pass

    def execute(
        self,
        prompt_embeds_pos: Union[List[Tensor], Tensor],
        prompt_embeds_neg: Union[List[Tensor], Tensor],
        latents_original: Tensor = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
        camera_embeddings: Tensor = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
        unet: MultiViewUNetWrapperModel = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
        scheduler: DDIMScheduler = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
        cfg: float = 7.5,
        vae: Optional[AutoencoderKL] = None,
        diffusion_steps: int = 50,
        n_views: int = 4,
        reconstruction_loss: bool = True,
        cfg_rescale: float = 0.5,
    ) -> Tensor:
        if latents_original is None:
            latents_original = LatentImage().execute(image_height=256, image_width=256, batch_size=4)
        if camera_embeddings is None:
            camera_embeddings = MVCreamCameraEmbedding().execute()
        if unet is None:
            unet = MVDreamModelLoader().execute()
        if scheduler is None:
            scheduler = SchedulerLoader().execute()

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

        device = latents_original.device
        scheduler.set_timesteps(diffusion_steps, device=device)

        with torch.no_grad():
            pbar = tqdm(scheduler.timesteps - 1)
            latents_noised = latents_original
            for step, time in enumerate(pbar):
                self.check_execution_state_change(clear_signal_event=lambda x: False)

                latents_noised = latents_original
                noise_pred, noise_pred_x0 = predict_noise_mvdream(
                    unet_mvdream=unet,
                    latents_noised=latents_noised, # [4, 4, 32, 32]
                    text_embeddings_conditional=prompt_embeds_pos, # [77, 768] x4
                    text_embeddings_unconditional=prompt_embeds_neg, # [77, 768] x4
                    camera_embeddings=camera_embeddings, # [4, 16]
                    cfg=cfg,
                    t=time,
                    scheduler=scheduler,
                    n_views=n_views,
                    reconstruction_loss=reconstruction_loss,
                    cfg_rescale=cfg_rescale,
                )
                latents_noised = scheduler.step(noise_pred, time, latents_noised).prev_sample # type: ignore
                latents_original = latents_noised

                if vae is not None and (step % 10 == 0 or step == len(pbar) - 1):
                    _ = latents_noised if noise_pred_x0 is None else noise_pred_x0 # use x0 instead of actual image if available
                    image_batch = vae.decode(1 / vae.config['scaling_factor'] * _.clone().detach()).sample # [B, C, H, W] # type: ignore
                    image_batch = (image_batch / 2 + 0.5).clamp(0, 1) * 255
                    image_batch = image_batch.permute(0, 2, 3, 1).detach().cpu() # [B, C, H, W] -> [B, H, W, C]
                    self.set_output("latents_noised", image_batch)
                    self.set_output("progress", int(100 * step / len(pbar)))
                    self.send_update()
        return latents_noised
