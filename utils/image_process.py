from PIL import Image
import cv2
import numpy as np
import torch

from utils.system import torch_gc
def check_image_size(x, padder_size=8):
    # 获取图像的宽高
    width, height = x.size
    padder_size = padder_size
    # 计算需要填充的高度和宽度
    mod_pad_h = (padder_size - height % padder_size) % padder_size
    mod_pad_w = (padder_size - width % padder_size) % padder_size
    x_np = np.array(x)
    # 使用 ImageOps.expand 进行填充
    x_padded = cv2.copyMakeBorder(x_np, top=0, bottom=mod_pad_h, left=0, right=mod_pad_w, borderType=cv2.BORDER_REPLICATE)

    x = Image.fromarray(x_padded)
    # x = x.resize((width + mod_pad_w, height + mod_pad_h))
    
    return x, width, height, width + mod_pad_w, height + mod_pad_h


def image2tensor(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.).unsqueeze(0)


def tensor2image(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

# This function was copied and adapted from https://huggingface.co/spaces/gokaygokay/TileUpscalerV2, licensed under Apache 2.0.
def create_hdr_effect(original_image, hdr):
    """
    Applies an HDR (High Dynamic Range) effect to an image based on the specified intensity.

    Args:
        original_image (PIL.Image.Image): The original image to which the HDR effect will be applied.
        hdr (float): The intensity of the HDR effect, ranging from 0 (no effect) to 1 (maximum effect).

    Returns:
        PIL.Image.Image: The image with the HDR effect applied.
    """
    if hdr == 0:
        return original_image  # No effect applied if hdr is 0

    # Convert the PIL image to a NumPy array in BGR format (OpenCV format)
    cv_original = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

    # Define scaling factors for creating multiple exposures
    factors = [
        1.0 - 0.9 * hdr,
        1.0 - 0.7 * hdr,
        1.0 - 0.45 * hdr,
        1.0 - 0.25 * hdr,
        1.0,
        1.0 + 0.2 * hdr,
        1.0 + 0.4 * hdr,
        1.0 + 0.6 * hdr,
        1.0 + 0.8 * hdr,
    ]

    # Generate multiple exposure images by scaling the original image
    images = [cv2.convertScaleAbs(cv_original, alpha=factor) for factor in factors]

    # Merge the images using the Mertens algorithm to create an HDR effect
    merge_mertens = cv2.createMergeMertens()
    hdr_image = merge_mertens.process(images)

    # Convert the HDR image to 8-bit format (0-255 range)
    hdr_image_8bit = np.clip(hdr_image * 255, 0, 255).astype("uint8")

    torch_gc()
    
    # Convert the image back to RGB format and return as a PIL image
    return Image.fromarray(cv2.cvtColor(hdr_image_8bit, cv2.COLOR_BGR2RGB))