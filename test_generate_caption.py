import torch.cuda
import argparse
from PIL import Image
from llava.llm_agent import LLavaAgent
from CKPT_PTH import LLAVA_MODEL_PATH, SDXL_PATH, foundirv2_PATH, VAE_FP16_PATH, BSRNet_PATH
import os
from torch.nn.functional import interpolate
import numpy as np
import cv2
import json
from foundirv2.create_foundirv2_model import create_bsrnet
from utils.image_process import image2tensor, tensor2image
# hyparams here
parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", type=str)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--upscale", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--no_llava", action='store_true', default=False)
parser.add_argument("--load_8bit_llava", action='store_true', default=False)
parser.add_argument("--use_bsrnet", action='store_true', default=False)
args = parser.parse_args()
print(args)
use_llava = not args.no_llava
use_bsrnet = args.use_bsrnet

# load LLaVA
if use_llava:
    llava_agent = LLavaAgent(LLAVA_MODEL_PATH, device='cuda:0', load_8bit=args.load_8bit_llava, load_4bit=False)
else:
    llava_agent = None

if use_bsrnet:
    bsrnet = create_bsrnet(BSRNet_PATH)
    bsrnet.to('cuda:0')
    bsrnet.eval()
else:
    bsrnet = None

os.makedirs(args.save_dir, exist_ok=True)
import json
with torch.no_grad():
    for file_name in sorted(os.listdir(args.img_dir)):
        img_name = os.path.splitext(file_name)[0]

        image = Image.open(os.path.join(args.img_dir,file_name)).convert('RGB')

        if use_bsrnet:
            image_tensor = image2tensor(np.array(image))
            image_tensor = image_tensor.to('cuda:0')
            image_tensor = bsrnet.deg_remove(image_tensor)
            image_deg_remove = Image.fromarray(tensor2image(image_tensor))
        else:
            image_deg_remove = image
        # step 1: LLaVA
        if use_llava:
            captions = llava_agent.gen_image_caption([image_deg_remove])
        else:
            captions = ['']

        data = {
            "caption": captions[0]
        }
        json_name = img_name+'.json'
        file_path = os.path.join(
            args.save_dir, json_name)
        with open(file_path, "w") as json_file:
            json.dump(data, json_file)
