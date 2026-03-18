import torch.cuda
import argparse
from foundirv2.create_foundirv2_model import foundirv2_pipeline
from PIL import Image
from CKPT_PTH import SDXL_PATH, foundirv2_PATH, VAE_FP16_PATH
from utils.color_fix import wavelet_color_fix, adain_color_fix
from utils.image_process import check_image_size
import os
import json


    
# hyparams here
parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", type=str)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--json_dir", type=str)
parser.add_argument("--upscale", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--min_size", type=int, default=1024)
parser.add_argument("--latent_tiled_overlap", type=float, default=0.5)
parser.add_argument("--latent_tiled_size", type=int, default=1024)
parser.add_argument("--guidance_scale", type=float, default=5)
parser.add_argument("--num_inference_steps", type=int, default=20)
parser.add_argument("--use_tile_vae", action='store_true', default=False)
parser.add_argument("--vae_tiled_overlap", type=float, default=0.25)
parser.add_argument("--vae_tiled_size", type=int, default=1024)
parser.add_argument("--color_fix", type=str, choices=['wavelet', 'adain', 'nofix'], default='nofix')
parser.add_argument("--start_point", type=str, choices=['lr', 'noise'], default='lr')
parser.add_argument("--cpu_offload", action='store_true', default=False)
parser.add_argument("--use_fp8", action='store_true', default=False)
args = parser.parse_args()
print(args)
cpu_offload = args.cpu_offload
use_fp8 = args.use_fp8


pipe = foundirv2_pipeline(sdxl_path=SDXL_PATH, VAE_FP16_path=VAE_FP16_PATH, foundirv2_path=foundirv2_PATH, use_fp8=use_fp8)
pipe = pipe.to('cuda:0')

if args.use_tile_vae:
    ### enable_vae_tiling
    pipe.set_encoder_tile_settings()
    pipe.enable_vae_tiling()

if cpu_offload:
    pipe.enable_model_cpu_offload()

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

        json_file = json.load(open(os.path.join(args.json_dir,img_name+'.json'))) 
        init_text = json_file["caption"]
        words = init_text.split()
        words = words[3:]
        words[0] = words[0].capitalize()
        text = ' '.join(words)
        text = text.split('. ')
        text = '. '.join(text[:2]) + '.'  


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
        generator = torch.Generator(device='cuda:0').manual_seed(args.seed)
        gen_image = pipe(lr_img=input_image, prompt = prompt_init, negative_prompt = negative_prompt_init, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, generator=generator, start_point=args.start_point, height = height_now, width=width_now, overlap=args.latent_tiled_overlap, target_size=(args.latent_tiled_size, args.latent_tiled_size)).images[0]
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

