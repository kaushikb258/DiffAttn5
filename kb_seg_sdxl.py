from pipeline_seg import StableDiffusionXLSEGPipeline
import torch
from diffusers.utils import make_image_grid
import PIL
from PIL import Image
import time


pipe = StableDiffusionXLSEGPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
)
device="cuda"
pipe = pipe.to(device)

prompts = [
"A raccoon playing guitar next to a blue ocean",
#"A painting of Mona Lisa in the style of Van Gogh",
#"A volcano erupting in New York City",
#"A blue VW van driving in the desert",
#"A cup of latte and a croissant.",
#"A man riding a bicycle near a busy vegetable market",
#"A painting of Mona Lisa in the style of starry nights",
]
seed = 5


output = []
for prompt in prompts:
    # Original
    generator = torch.Generator(device="cuda").manual_seed(seed)
    t1 = time.time()
    output += pipe(
            [prompt],
            num_inference_steps=25,
            height=768,
            width=768,
            guidance_scale=1.0,
            seg_scale=0.0,
            seg_blur_sigma=0.0,
            attn_guid_option=0,
            seg_applied_layers=['mid'],
            generator=generator,
        ).images
    print('time:', time.time()-t1)    
    
    # SEG gaussian blur sigma = 10
    generator = torch.Generator(device="cuda").manual_seed(seed)
    t1 = time.time()
    output += pipe(
            [prompt],
            num_inference_steps=25,
            height=768,          
            width=768,
            guidance_scale=1.0,
            seg_scale=3.0,
            seg_blur_sigma=10.0,
            attn_guid_option=0,
            seg_applied_layers=['mid'],
            generator=generator,
        ).images
    print('time:', time.time()-t1)
        
    # SEG gaussian blur sigma = inf    
    generator = torch.Generator(device="cuda").manual_seed(seed)
    t1 = time.time()
    output += pipe(
            [prompt],
            num_inference_steps=25,
            height=768,
            width=768,
            guidance_scale=1.0,
            seg_scale=3.0,
            seg_blur_sigma=10000000000.0,
            attn_guid_option=0,
            seg_applied_layers=['mid'],
            generator=generator,
        ).images
    print('time:', time.time()-t1)

grid = make_image_grid(output, rows=1, cols=3)
grid.save("monalisa_top.png")

output = []
for prompt in prompts:

    # ATTN GUIDANCE option = 1  (Wq + W_rnd)     
    generator = torch.Generator(device="cuda").manual_seed(seed)
    t1 = time.time()
    output += pipe(
            [prompt],
            num_inference_steps=25,
            height=768,
            width=768,
            guidance_scale=1.0,
            seg_scale=3.0,
            seg_blur_sigma=10000000000.0,
            attn_guid_option=1,
            seg_applied_layers=['mid'],
            generator=generator,
        ).images
    print('time:', time.time()-t1)
   
    # ATTN GUIDANCE option = 2 (QK dropout)     
    generator = torch.Generator(device="cuda").manual_seed(seed)
    t1 = time.time()
    output += pipe(
            [prompt],
            num_inference_steps=25,
            height=768,
            width=768,
            guidance_scale=1.0,
            seg_scale=3.0,
            seg_blur_sigma=10000000000.0,
            attn_guid_option=2,
            seg_applied_layers=['mid'],
            generator=generator,
        ).images
    print('time:', time.time()-t1)
    
    # ATTN GUIDANCE option = 3 (Bilateral filter)     
    generator = torch.Generator(device="cuda").manual_seed(seed)
    t1 = time.time()
    output += pipe(
            [prompt],
            num_inference_steps=25,
            height=768,
            width=768,
            guidance_scale=1.0,
            seg_scale=3.0,
            seg_blur_sigma=10000000000.0,
            attn_guid_option=3,
            seg_applied_layers=['mid'],
            generator=generator,
        ).images
    print('time:', time.time()-t1)
   
    # ATTN GUIDANCE option = 4 (Erosion)     
    generator = torch.Generator(device="cuda").manual_seed(seed)
    t1 = time.time()
    output += pipe(
            [prompt],
            num_inference_steps=25,
            height=768,
            width=768,
            guidance_scale=1.0,
            seg_scale=3.0,
            seg_blur_sigma=10000000000.0,
            attn_guid_option=4,
            seg_applied_layers=['mid'],
            generator=generator,
        ).images
    print('time:', time.time()-t1)
   
    # ATTN GUIDANCE option = 5 (Dilation)     
    generator = torch.Generator(device="cuda").manual_seed(seed)
    t1 = time.time()
    output += pipe(
            [prompt],
            num_inference_steps=25,
            height=768,
            width=768,
            guidance_scale=1.0,
            seg_scale=3.0,
            seg_blur_sigma=10000000000.0,
            attn_guid_option=5,
            seg_applied_layers=['mid'],
            generator=generator,
        ).images
    print('time:', time.time()-t1) 
   
grid = make_image_grid(output, rows=1, cols=5)
grid.save("monalisa_bottom.png")
