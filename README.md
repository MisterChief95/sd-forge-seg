# SEG (Smoothed Energy Guidance)

SEG blurs self-attention weights with a 1D Gaussian kernel and uses the smoothed attention as a negative guidance source, reducing curvature in the energy landscape per Susung Hong's SEG paper.

## How it works
- `create_seg_attention_processor` convolves the attention tensor with a Gaussian kernel (alpha-controlled by the `Blur Sigma` slider) and re-normalizes the blurred weights.
- The adjusted weights replace the `attn1` entry for the chosen U-Net block, and the script combines the conditioned and perturbed denoised predictions to create negative guidance.
- An extra callback limits the effect to any start/end step window, and it cleans up callback registrations after each run.

## UI controls
- **Blur Sigma** (0.5–5.0, default 2.0) is the primary control—higher values produce stronger smoothing, while `-1` forces a uniform attention for a severe flattening effect.
- **Guidance Scale** (1.0–7.0, default 3.0) mixes the negative guidance back into the denoised output.
- **U-Net block/ID** selections let you target middle/output/input blocks (middle is the recommended starting point) and fine-tune the offset.
- **Start/End step** define the sampling slice where SEG is active.
- **Skip HiRes Fix** avoids reapplying SEG during high-resolution passes when checked.

## Tips
- Keep scale near 3.0 for predictable results; tweak blur sigma for the level of energy smoothing you want.
- The guide accordion shares example presets from subtle to uniform attention.

## References
- Paper: https://arxiv.org/abs/2408.00760
- Official SEG: https://github.com/SusungHong/SEG-SDXL
- ComfyUI reference: https://github.com/pamparamm/sd-perturbed-attention
