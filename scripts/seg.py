# Smoothed Energy Guidance (SEG) for Stable Diffusion WebUI Forge
# 
# Based on:
# - Paper: "Smoothed Energy Guidance: Guiding Diffusion Models with Reduced Energy Curvature of Attention" 
#   by Susung Hong (NeurIPS 2024)
#   https://arxiv.org/abs/2408.00760
# - Official implementation: https://github.com/SusungHong/SEG-SDXL
# - ComfyUI implementation: https://github.com/pamparamm/sd-perturbed-attention
#
# SEG applies Gaussian blur to self-attention weights to smooth the energy landscape,
# using the blurred output as negative guidance.

import gradio as gr
import torch
import torch.nn.functional as F
from typing import Optional

from modules import scripts
from modules.script_callbacks import on_cfg_denoiser, remove_current_script_callbacks
from backend.patcher.base import set_model_options_patch_replace
from backend.sampling.sampling_function import calc_cond_uncond_batch
from modules.ui_components import InputAccordion


def gaussian_kernel_1d(sigma: float, kernel_size: Optional[int] = None) -> torch.Tensor | None:
    """Create 1D Gaussian kernel for attention blurring."""
    if sigma < 0:
        return None
        
    if kernel_size is None:
        kernel_size = int(2 * torch.ceil(torch.tensor(3 * sigma)).item() + 1)
        kernel_size = max(kernel_size, 3)
    
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    center = kernel_size // 2
    x = torch.arange(kernel_size, dtype=torch.float32) - center
    kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    
    return kernel


def create_seg_attention_processor(blur_sigma: float):
    """Create attention processor with Gaussian blur."""
    
    def seg_attention(q, k, v, extra_options):
        scores = torch.bmm(q, k.transpose(-2, -1))
        scores = scores / (q.shape[-1] ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        if blur_sigma < 0:
            seq_len = attn_weights.shape[-1]
            blurred_weights = torch.ones_like(attn_weights) / seq_len
        else:
            kernel = gaussian_kernel_1d(blur_sigma)
            if kernel is None:
                blurred_weights = attn_weights
            else:
                kernel = kernel.to(attn_weights.device, dtype=attn_weights.dtype)
                kernel_size = kernel.shape[0]
                padding = kernel_size // 2
                
                batch_heads, seq_len, _ = attn_weights.shape
                attn_flat = attn_weights.view(batch_heads * seq_len, 1, seq_len)
                kernel_conv = kernel.view(1, 1, -1)
                blurred = F.conv1d(attn_flat, kernel_conv, padding=padding)
                blurred_weights = blurred.view(batch_heads, seq_len, seq_len)
                blurred_weights = F.softmax(blurred_weights / blurred_weights.std(), dim=-1)
        
        output = torch.bmm(blurred_weights, v)
        return output
    
    return seg_attention


class SEGImproved(scripts.Script):
    sorting_priority = 14

    def title(self):
        return "SEG - Smoothed Energy Guidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label=self.title()) as enabled:
            gr.Markdown("💡 **Tip**: Adjust blur sigma for quality. Higher = clearer, sharper images.")
            
            with gr.Row():
                # Blur sigma is the PRIMARY control for SEG
                blur_sigma = gr.Slider(
                    label='Blur Sigma (Main Control)',
                    minimum=0.5,
                    maximum=5.0,
                    step=0.1,
                    value=2.0,
                    info="⭐ Primary quality control: 1.0-1.5=subtle, 2.0-3.0=standard, 4.0+=strong"
                )
            
            with gr.Row():
                # Scale is secondary for SEG (usually keep at 3.0)
                scale = gr.Slider(
                    label='Guidance Scale',
                    minimum=1.0,
                    maximum=7.0,
                    step=0.5,
                    value=3.0,
                    info="Usually keep at 3.0-4.0. Less critical than blur sigma for SEG."
                )

            with gr.Row():
                skip_hires = gr.Checkbox(
                    label='Skip HiRes Fix',
                    value=False,
                    visible=not is_img2img
                )
            
            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    # Block settings - middle is almost always best for SEG
                    unet_block = gr.Radio(
                        label='U-Net Block',
                        choices=['middle', 'output', 'input'],
                        value='middle',
                        info="Middle is recommended for SEG"
                    )
                    unet_block_id = gr.Slider(
                        label='Block ID',
                        minimum=0,
                        maximum=5,
                        step=1,
                        value=0,
                        info="Usually keep at 0"
                    )
                
                with gr.Row():
                    start_step = gr.Slider(
                        label='Start Step',
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.0
                    )
                    end_step = gr.Slider(
                        label='End Step',
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=1.0
                    )

            with gr.Accordion("Guide", open=False):
                gr.Markdown("""
                **Subtle Enhancement**: blur_sigma=1.2, scale=3.0
                **Standard (Recommended)**: blur_sigma=2.0, scale=3.0  
                **Strong Enhancement**: blur_sigma=3.5, scale=4.0
                **Maximum Clarity**: blur_sigma=5.0, scale=3.0
                **Uniform Attention**: blur_sigma=-1, scale=3.0
                """)

        self.infotext_fields = [
            (enabled, lambda d: d.get("seg_enabled", False)),
            (blur_sigma, "seg_blur_sigma"),
            (scale, "seg_scale"),
            (unet_block, "seg_block"),
            (unet_block_id, "seg_block_id"),
            (start_step, "seg_start_step"),
            (end_step, "seg_end_step"),
            (skip_hires, "seg_skip_hires"),
        ]

        return enabled, blur_sigma, scale, unet_block, unet_block_id, start_step, end_step, skip_hires

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        # Note: blur_sigma now comes BEFORE scale in args
        enabled, blur_sigma, scale, unet_block, unet_block_id, start_step, end_step, skip_hires = script_args

        if not enabled or (getattr(p, 'is_hr_pass', False) and skip_hires):
            return

        print(f"[SEG] blur_sigma={blur_sigma:.2f} (primary), scale={scale:.2f}, block={unet_block}[{int(unet_block_id)}]")

        SEGImproved.scale = scale
        SEGImproved.blur_sigma = blur_sigma
        SEGImproved.unet_block = unet_block
        SEGImproved.unet_block_id = int(unet_block_id)
        SEGImproved.start_step = start_step
        SEGImproved.end_step = end_step
        SEGImproved.do_seg = True

        def denoiser_callback(params):
            current_step = params.sampling_step / (params.total_sampling_steps - 1)
            SEGImproved.do_seg = (
                current_step >= SEGImproved.start_step and
                current_step <= SEGImproved.end_step
            )

        on_cfg_denoiser(denoiser_callback)

        unet = p.sd_model.forge_objects.unet.clone()
        seg_attn = create_seg_attention_processor(blur_sigma)

        def post_cfg_function(args):
            denoised = args["denoised"]

            if SEGImproved.scale <= 0.0 or not SEGImproved.do_seg:
                return denoised

            model = args["model"]
            cond_denoised = args["cond_denoised"]
            cond = args["cond"]
            sigma = args["sigma"]
            x = args["input"]
            options = args["model_options"].copy()

            new_options = set_model_options_patch_replace(
                options, seg_attn, "attn1",
                SEGImproved.unet_block, SEGImproved.unet_block_id
            )

            seg_cond_denoised, _ = calc_cond_uncond_batch(model, cond, None, x, sigma, new_options)
            result = denoised + (cond_denoised - seg_cond_denoised) * SEGImproved.scale

            return result

        unet.set_model_sampler_post_cfg_function(post_cfg_function)
        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params.update(dict(
            seg_enabled=enabled,
            seg_blur_sigma=blur_sigma,
            seg_scale=scale,
            seg_block=unet_block,
            seg_block_id=int(unet_block_id),
            seg_start_step=start_step,
            seg_end_step=end_step,
        ))

    def postprocess(self, p, processed, *args):
        remove_current_script_callbacks()