import io
import torch
import torchvision
import pathlib

from torch import Tensor
from typing import TypedDict
from backend.loader.decorator import KatzukiNode
from backend.nodes.builtin import BaseNode
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from kokikit.lgm import LGM, save_ply
from torch.cuda.amp.autocast_mode import autocast


class LGMModelLoader(BaseNode):

    @KatzukiNode(node_type="diffusion.lgm.lgm_model_loader")
    def __init__(self) -> None:
        pass

    class ReturnDict(TypedDict):
        lgm_model: LGM

    def execute(
            self,
            lgm_path: str = "ashawkey/LGM",
            filename: str = "model_fp16.safetensors",
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cuda"),
    ) -> ReturnDict:
        # TODO: maybe load to GPU is faster?
        ckpt = load_file(hf_hub_download(repo_id=lgm_path, filename=filename), device='cuda') # TODO: change device
        lgm_model = LGM(
            down_channels=(64, 128, 256, 512, 1024, 1024),
            down_attention=(False, False, False, True, True, True),
            mid_attention=True,
            up_channels=(1024, 1024, 512, 256, 128), # using "big" config
            up_attention=(True, True, True, False, False), # using "big" config
            splat_H=128, # using "big" config
            splat_W=128, # using "big" config
        )
        lgm_model.load_state_dict(ckpt, strict=False, assign=True)
        lgm_model = lgm_model.half().to(device=device)
        lgm_model.eval()

        return self.ReturnDict(lgm_model=lgm_model)


class RunLGM(BaseNode):

    @KatzukiNode(node_type="diffusion.lgm.run_lgm", input_description={
        "images": "The input images tensor of shape [4, 3, 256, 256] with values in range [-1, 1] and white background",
    })
    def __init__(self) -> None:
        pass

    def execute(self, lgm_model: LGM, images: Tensor, device: torch.device = torch.device("cuda")) -> Tensor:
        images = (images + 1) / 2 # [-1, 1] -> [0, 1]

        # TODO: think more carefully about half and eval
        rays_embeddings = LGM.get_rays(device=device) # [B, 6, H, W]

        images = torch.nn.functional.interpolate(images, size=(256, 256), mode='bilinear', align_corners=False)
        images = torchvision.transforms.functional.normalize(images, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # type: ignore # [B, 3, H, W]
        embedding = torch.cat([images, rays_embeddings], dim=1).unsqueeze(0) # [B, 4, 9, H, W]

        # TODO: autocast vs nograd
        with autocast(dtype=torch.float16):
            return lgm_model.forward_gaussians(embedding) # [B, N, 14]


class RenderGaussian(BaseNode):

    @KatzukiNode(node_type="diffusion.lgm.render_gaussian")
    def __init__(self) -> None:
        pass

    class ReturnDict(TypedDict):
        gaussian: bytes

    def execute(self, gaussian: Tensor) -> ReturnDict:
        binary_stream = io.BytesIO()
        save_ply(gaussians=gaussian, stream=binary_stream, compatible=True)
        binary_data = binary_stream.getvalue()
        return self.ReturnDict(gaussian=binary_data)


class SaveGaussian(BaseNode):

    @KatzukiNode(node_type="diffusion.lgm.save_gaussian")
    def __init__(self) -> None:
        pass

    def execute(self, gaussian: Tensor, path: str = "gaussian.ply") -> pathlib.Path:
        assert path.endswith(".ply") or path.endswith(".splat")
        assert path.endswith(".ply"), "Splat is not supported yet" # TODO: support it
        pathlib_path = self.OUTPUT_PATH / path # katzuki/src/storage/{uuid}/output/{path}

        save_ply(gaussians=gaussian, stream=pathlib_path, compatible=True)

        return pathlib_path
