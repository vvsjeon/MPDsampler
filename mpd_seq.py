import math
import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional
import cv2

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))
import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config, append_dims
from torchvision.transforms import ToTensor
from tqdm import tqdm

def sample(
    input_start_path: str = "examples/1/00000.jpg",
    input_end_path: str = "examples/1/00024.jpg",
    video_path: str = "video",
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    version: str = "svd_xt",
    fps_id: int = 8,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 4,    # This eats most VRAM! Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = None,
    verbose: Optional[bool] = False,
    ratio: float = 0.01,
):
    if version == "svd_xt":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 25)
        output_folder = default(output_folder, "output")
        model_config = "scripts/sampling/configs/svd_xt.yaml"
    else:
        raise ValueError(f"Version {version} does not exist.")

    model, filter = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
        verbose,
    )
    torch.manual_seed(seed)

    def load_and_encode(
            image_path: str,
            model,
            device: str = "cuda",
            resize: tuple = (1024, 576),
            to_half: bool = True,
    ):
        image = Image.open(image_path).convert("RGB").resize(resize)
        image_tensor = ToTensor()(image) * 2.0 - 1.0  # Normalize to [-1, 1]
        image_tensor = image_tensor.unsqueeze(0).to(device)
        if to_half:
            image_tensor = image_tensor.to(torch.float16)

        latent = model.encode_first_stage(image_tensor)
        return image_tensor, latent
    
    image_start, latent_start = load_and_encode(input_start_path, model, device)
    image_end, latent_end = load_and_encode(input_end_path, model, device)

    H, W = image_start.shape[2:]
    assert image_start.shape[1] == 3
    F = 8
    C = 4
    shape = (num_frames, C, H // F, W // F)

    def make_value_dict(image_tensor, motion_bucket_id, fps_id, cond_aug, device):
        return {
            "cond_frames_without_noise": image_tensor,
            "cond_frames": image_tensor + cond_aug * torch.randn_like(image_tensor).to(device),
            "motion_bucket_id": motion_bucket_id,
            "fps_id": fps_id,
            "cond_aug": cond_aug,
        }
    
    value_dict_start = make_value_dict(image_start, motion_bucket_id, fps_id, cond_aug, device)
    value_dict_end = make_value_dict(image_end, motion_bucket_id, fps_id, cond_aug, device)

    with torch.no_grad():
        with torch.autocast(device):
            def prepare_conditioning(model, value_dict, num_frames, device):
                keys = get_unique_embedder_keys_from_conditioner(model.conditioner)

                batch, batch_uc = get_batch(keys, value_dict, [1, num_frames], T=num_frames, device=device)

                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=["cond_frames", "cond_frames_without_noise"],
                )

                for k in ["crossattn", "concat"]:
                    uc[k] = rearrange(repeat(uc[k], "b ... -> b t ...", t=num_frames), "b t ... -> (b t) ...", t=num_frames)
                    c[k] = rearrange(repeat(c[k], "b ... -> b t ...", t=num_frames), "b t ... -> (b t) ...", t=num_frames)

                return c, uc, batch

            c_start, uc_start, batch_start = prepare_conditioning(model, value_dict_start, num_frames, device)
            c_end, uc_end, batch_end = prepare_conditioning(model, value_dict_end, num_frames, device)

            randn = torch.randn(shape, device=device)

            additional_model_inputs = {
                "image_only_indicator": torch.zeros(2, num_frames).to(device),
                "num_video_frames": batch_start["num_video_frames"],
            }
            
            def denoiser(x, sigma, c, uc):
                c_out = dict()
                for k in c:
                    if k in ["vector", "crossattn", "concat"]:
                        c_out[k] = torch.cat((uc[k], c[k]), 0)
                    else:
                        assert c[k] == uc[k]
                        c_out[k] = c[k]
                denoiser_input, denoiser_sigma, denoiser_c = torch.cat([x] * 2), torch.cat([sigma] * 2), c_out
                sigma_shape = denoiser_sigma.shape
                denoiser_sigma = append_dims(denoiser_sigma, x.ndim)
                c_skip = 1.0 / (denoiser_sigma**2 + 1.0)
                c_out = -denoiser_sigma / (denoiser_sigma**2 + 1.0) ** 0.5
                c_in = 1.0 / (denoiser_sigma**2 + 1.0) ** 0.5
                c_noise = 0.25 * denoiser_sigma.log()
                c_noise = c_noise.reshape(sigma_shape)
                eps_pred = model.model(denoiser_input * c_in, c_noise, denoiser_c, **additional_model_inputs)
                denoised = eps_pred * c_out + denoiser_input * c_skip
                x_u, x_c = denoised.chunk(2)
                return x_u, x_c
            
            def CFG(x_u, x_c, scale):
                x_u = rearrange(x_u, "(b t) ... -> b t ...", t=num_frames)
                x_c = rearrange(x_c, "(b t) ... -> b t ...", t=num_frames)
                scale = torch.linspace(scale, scale, steps=num_frames).unsqueeze(0)
                scale = repeat(scale, "1 t -> b t", b=x_u.shape[0])
                scale = append_dims(scale, x_u.ndim).to(x_u.device)
                denoised =  rearrange(x_u + scale * (x_c - x_u), "b t ... -> (b t) ...")
                return denoised
        
            def masking(x, index):
                mask = torch.zeros_like(x)  
                mask[index, :, :, :] = 1
                return x * mask
            
            def CG(A, b, x, n_inner=5, eps=1e-5):
                r = b - A(x)
                p = r.clone()
                rsold = torch.sum(r * r, dim=[0, 1, 2, 3], keepdim=True)  
                for i in range(n_inner):
                    Ap = A(p)
                    a = rsold / torch.sum(p * Ap, dim=[0, 1, 2, 3], keepdim=True)  
                    x = x + a * p
                    r = r - a * Ap
                    rsnew = torch.sum(r * r, dim=[0, 1, 2, 3], keepdim=True)  
                    if torch.abs(torch.sqrt(rsnew)) < eps:
                        break
                    p = r + (rsnew / rsold) * p
                    rsold = rsnew
                return x
            
            def DDS(x, n_inner, latent):
                measurement = torch.zeros_like(x)
                measurement[-1, :, :, :] = latent
                A = lambda z: masking(z, -1)
                AT = lambda z: masking(z, -1)
                def Acg(x):
                    return AT(A(x))
                Acg_fn = Acg
                bcg = AT(measurement)
                return CG(Acg_fn, bcg, x, n_inner=n_inner)

            x_fwd, s_in, sigmas, num_sigmas, c_start, uc_start = model.sampler.prepare_sampling_loop(
                randn, c_start, uc_start, num_steps
            )

            ### Time Reversal Sampling ###
            for i in tqdm(model.sampler.get_sigma_gen(num_sigmas), total=num_sigmas - 1):
                ### parameters ###
                gamma = (
                    min(model.sampler.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                    if model.sampler.s_tmin <= sigmas[i] <= model.sampler.s_tmax
                    else 0.0
                )
                sigma = s_in * sigmas[i]
                next_sigma = s_in * sigmas[i + 1]
                sigma_hat = sigma * (gamma + 1.0)
                
                if gamma > 0:
                    eps = torch.randn_like(x_fwd) * model.sampler.s_noise
                    x_fwd = x_fwd + eps * append_dims(sigma_hat**2 - sigma**2, x_fwd.ndim) ** 0.5

                ##### Align two temporal paths with MPD #####
                if i < 0.2 * num_steps:
                    for _ in range(3):
                        x_bwd = torch.flip(x_fwd, dims=[0])
                        
                        ### Denoise forward path ###
                        x0_fwd_uc, x0_fwd_c = denoiser(x_fwd, sigma_hat, c_start, uc_start)
                        
                        ### Calculate residuals ###
                        eps_fwd = (x_fwd - x0_fwd_c) / append_dims(sigma_hat, x_fwd.ndim)    
                        eps_bwd_end = (x_bwd[:1] - latent_end) / append_dims(sigma_hat[0], x_bwd[:1].ndim)
                        delta_eps_fwd = eps_fwd[1:] - eps_fwd[:-1]
                        delta_eps_bwd_trg = -torch.flip(delta_eps_fwd, dims=[0])
                        
                        ### Init backward noise ###
                        eps_bwd_init = torch.empty_like(eps_fwd)
                        
                        ### Distill forward residuals to backward ###
                        eps_bwd_init[:1] = eps_bwd_end
                        eps_bwd_init[1:] = eps_bwd_init[0] + torch.cumsum(delta_eps_bwd_trg, dim=0)
                        x0_bwd_c = x_bwd - append_dims(sigma_hat, x_bwd.ndim) * eps_bwd_init
                        
                        ### Update ###
                        d = (x_bwd - torch.flip(x0_fwd_uc, dims=[0])) / append_dims(sigma_hat, x_bwd.ndim)
                        dt = append_dims(next_sigma, x_bwd.ndim)
                        x0_bwd_c = DDS(x0_bwd_c, n_inner=5, latent=latent_start)
                        x_bwd = x0_bwd_c + d * dt
                        x_fwd = torch.flip(x_bwd, dims=[0])
                        
                        ### Renoise ###
                        eps = torch.randn_like(x_fwd) * model.sampler.s_noise
                        x_fwd = x_fwd + eps * append_dims(sigma_hat**2 - next_sigma**2, x_fwd.ndim) ** 0.5
                    
                    x0_fwd_uc, x0_fwd_c = denoiser(x_fwd, sigma_hat, c_start, uc_start)
                    x0_fwd_c = DDS(x0_fwd_c, n_inner=5, latent=latent_end)
                    d = (x_fwd - x0_fwd_uc) / append_dims(sigma_hat, x_fwd.ndim)
                    dt = append_dims(next_sigma, x_fwd.ndim)
                    x_fwd = x0_fwd_c + d * dt
                    
                    eps = torch.randn_like(x_fwd) * model.sampler.s_noise
                    x_fwd = x_fwd + eps * append_dims(sigma_hat**2 - next_sigma**2, x_fwd.ndim) ** 0.5
                    
                    x_bwd = torch.flip(x_fwd, dims=[0])
                    x0_bwd_uc, x0_bwd_c = denoiser(x_bwd, sigma_hat, c_end, uc_end)
                    x0_bwd_c = DDS(x0_bwd_c, n_inner=5, latent=latent_start)  
                    d = (x_bwd - x0_bwd_uc) / append_dims(sigma_hat, x_bwd.ndim)
                    dt = append_dims(next_sigma, x_bwd.ndim)
                    
                    x_bwd = x0_bwd_c + d * dt
                    x_fwd = torch.flip(x_bwd, dims=[0])
                    

                else: 
                    x0_fwd_uc, x0_fwd_c = denoiser(x_fwd, sigma_hat, c_start, uc_start)
                    x0_fwd_c = DDS(x0_fwd_c, n_inner=5, latent=latent_end)                    
                    d = (x_fwd - x0_fwd_uc) / append_dims(sigma_hat, x_fwd.ndim)
                    dt = append_dims(next_sigma, x_fwd.ndim)
                    x_fwd = x0_fwd_c + d * dt
                
                    eps = torch.randn_like(x_fwd) * model.sampler.s_noise
                    x_fwd = x_fwd + eps * append_dims(sigma_hat**2 - next_sigma**2, x_fwd.ndim) ** 0.5
                    
                    x_bwd = torch.flip(x_fwd, dims=[0])
                    x0_bwd_uc, x0_bwd_c = denoiser(x_bwd, sigma_hat, c_end, uc_end)
                    x0_bwd_c = DDS(x0_bwd_c, n_inner=5, latent=latent_start)  
                    d = (x_bwd - x0_bwd_uc) / append_dims(sigma_hat, x_bwd.ndim)
                    dt = append_dims(next_sigma, x_bwd.ndim)
                    
                    x_bwd = x0_bwd_c + d * dt
                    x_fwd = torch.flip(x_bwd, dims=[0])
                    

            samples_z = x_fwd
            model.en_and_decode_n_samples_a_time = decoding_t
            model = model.to(torch.float32)
            samples_x = model.decode_first_stage(samples_z)
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

            samples = embed_watermark(samples)
            samples = filter(samples)
            vid = (
                (rearrange(samples, "t c h w -> t h w c") * 255)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            os.makedirs(output_folder, exist_ok=True)

            video_path = os.path.join(output_folder, f"{base_count:06d}.gif")

            images = [Image.fromarray(vid[i]) for i in range(vid.shape[0])]                
            duration = 125              
            images[0].save(video_path, save_all=True, append_images=images[1:], duration=duration, loop=0)


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames" or key == "cond_frames_without_noise":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0])
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
    verbose: bool = False,      
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.verbose = verbose
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()
    
    model = model.to(torch.float16)

    filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter


if __name__ == "__main__":
    Fire(sample)