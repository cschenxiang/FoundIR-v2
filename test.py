import torch.cuda
import argparse
from foundirv2.create_foundirv2_model import foundirv2_pipeline
from PIL import Image
from CKPT_PTH import LLAVA_MODEL_PATH, SDXL_PATH, foundirv2_PATH, VAE_FP16_PATH, BSRNet_PATH
from utils.color_fix import wavelet_color_fix, adain_color_fix
from utils.image_process import check_image_size
from llava.llm_agent import LLavaAgent

import os
import numpy as np
import cv2

from foundirv2.create_foundirv2_model import create_bsrnet
from utils.image_process import image2tensor, tensor2image


if torch.cuda.device_count() >= 2:
    LLaVA_device = 'cuda:1'
    Diffusion_device = 'cuda:0'
elif torch.cuda.device_count() == 1:
    Diffusion_device = 'cuda:0'
    LLaVA_device = 'cuda:0'
else:
    raise ValueError('Currently support CUDA only.')


# hyparams here
parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", type=str)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--upscale", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--min_size", type=int, default=1024)
parser.add_argument("--latent_tiled_overlap", type=float, default=0.5)
parser.add_argument("--latent_tiled_size", type=int, default=1024)
parser.add_argument("--guidance_scale", type=float, default=5)
parser.add_argument("--num_inference_steps", type=int, default=20)
parser.add_argument("--no_llava", action='store_true', default=False)
parser.add_argument("--use_tile_vae", action='store_true', default=False)
parser.add_argument("--vae_tiled_overlap", type=float, default=0.25)
parser.add_argument("--vae_tiled_size", type=int, default=1024)
parser.add_argument("--use_bsrnet", action='store_true', default=False)
parser.add_argument("--load_8bit_llava", action='store_true', default=False)
parser.add_argument("--color_fix", type=str, choices=['wavelet', 'adain', 'nofix'], default='nofix')
parser.add_argument("--start_point", type=str, choices=['lr', 'noise'], default='lr')
parser.add_argument("--cpu_offload", action='store_true', default=False)
parser.add_argument("--use_fp8", action='store_true', default=False)
args = parser.parse_args()
print(args)
cpu_offload = args.cpu_offload
use_fp8 = args.use_fp8
use_llava = not args.no_llava
use_bsrnet = args.use_bsrnet

# load foundirv2 FP16
pipe = foundirv2_pipeline(sdxl_path=SDXL_PATH, VAE_FP16_path=VAE_FP16_PATH, foundirv2_path=foundirv2_PATH, use_fp8=use_fp8)
pipe = pipe.to(Diffusion_device)

if use_bsrnet:
    bsrnet = create_bsrnet(BSRNet_PATH)
    bsrnet.to(LLaVA_device)
    bsrnet.eval()
else:
    bsrnet = None

if args.use_tile_vae:
    ### enable_vae_tiling
    pipe.set_encoder_tile_settings()
    pipe.enable_vae_tiling()

if cpu_offload:
    pipe.enable_model_cpu_offload()

# load LLaVA
if use_llava:
    llava_agent = LLavaAgent(LLAVA_MODEL_PATH, device=LLaVA_device, load_8bit=args.load_8bit_llava, load_4bit=False)
else:
    llava_agent = None


os.makedirs(args.save_dir, exist_ok=True)

exist_file = os.listdir(args.save_dir)
with torch.no_grad():
    for file_name in sorted(os.listdir(args.img_dir)):
        img_name, ext = os.path.splitext(file_name)
        if ext == ".json":
            continue
        
        if f"{img_name}.png" in exist_file:
            print(f"{img_name}.png exist")
            continue
        else:
            print(img_name)
        image = Image.open(os.path.join(args.img_dir,file_name)).convert('RGB')

        if use_bsrnet:
            image_tensor = image2tensor(np.array(image))
            image_tensor = image_tensor.to(LLaVA_device)
            image_tensor = bsrnet.deg_remove(image_tensor)
            image_deg_remove = Image.fromarray(tensor2image(image_tensor))
        else:
            image_deg_remove = image

        # step 1: LLaVA
        if use_llava:
            captions = llava_agent.gen_image_caption([image_deg_remove])
        else:
            captions = ['']

        init_text = captions[0]
        words = init_text.split()
        words = words[3:]
        words[0] = words[0].capitalize()
        text = ' '.join(words)
        text = text.split('. ')
        text = '. '.join(text[:2]) + '.'
        print(text)

        # step 2: Restoration
        w, h = image.size

        if args.upscale == 0:
            if w * h < 1920 * 1080:
                current_upscale = 2
            else:
                current_upscale = 1
        else:
            current_upscale = args.upscale

        w *= current_upscale
        h *= current_upscale
        image = image.resize((w, h), Image.LANCZOS)
        input_image, width_init, height_init, width_now, height_now = check_image_size(image)
        prompt_init = text 
        negative_prompt_init = ""
        generator = torch.Generator(device='cuda').manual_seed(args.seed)
        gen_image = pipe(lr_img=input_image, prompt = prompt_init, negative_prompt = negative_prompt_init, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, generator=generator, start_point=args.start_point, height = height_now, width=width_now,  overlap=args.latent_tiled_overlap, target_size=(args.latent_tiled_size, args.latent_tiled_size)).images[0]
        path = os.path.join(args.save_dir, img_name+'.png')
        cropped_image = gen_image.crop((0, 0, width_init, height_init))
        if args.color_fix == 'nofix':
            out_image = cropped_image
        else:
            if args.color_fix == 'wavelet':
                out_image = wavelet_color_fix(cropped_image, image)
            elif args.color_fix == 'adain':
                out_image = adain_color_fix(cropped_image, image)
        out_image.save(path)
