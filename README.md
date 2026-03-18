<p align="center">
  <img src="assets/logo.png" height=120>
</p>

### FoundIR-v2: Optimizing Pre-Training Data Mixtures for Image Restoration Foundation Model

> [[Paper](https://arxiv.org/abs/2512.09282)] &emsp; [Supplemental Material] &emsp; [[中文版解读](https://mp.weixin.qq.com/s/YEFRmdXPDr3tt8x8y8FTkQ)]

> [Xiang Chen](https://cschenxiang.github.io/), [Jinshan Pan](https://jspan.github.io/), [Jiangxin Dong](https://scholar.google.com/citations?user=ruebFVEAAAAJ&hl=zh-CN&oi=ao), [Jian Yang](https://scholar.google.com/citations?hl=en&user=6CIDtZQAAAAJ), [Jinhui Tang](https://scholar.google.com/citations?user=ByBLlEwAAAAJ&hl=zh-CN)  <br>
> Nanjing University of Science and Technology, Nanjing Forestry University

Welcome to visit our website (底层视觉社区平台&基础科研平台) for low-level vision: https://lowlevelcv.com/

---
<p align="center">
  <img width="800" src="./assets/network.png">
</p>

*Building upon FoundIR (ICCV 2025), which investigates the effects of training **data scaling laws** in image restoration foundation models, FoundIR-v2 further uncovers the **data mixing laws** that govern how mixture proportions influence all-in-one image restoration.*

---

### 🚩 **New Features/Updates**
- ✅ March 18, 2026. Our FoundIR-v2 code and pre-trained model are now available!
- ✅ February 21, 2026. Our FoundIR-v2 was accepted by **CVPR 2026**!
- ✅ December 11, 2025. Release FoundIR-v2 [paper](https://arxiv.org/abs/2512.09282).
- ✅ June 26, 2025. Our FoundIR was accepted by **ICCV 2025**! ([paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_FoundIR_Unleashing_Million-scale_Training_Data_to_Advance_Foundation_Models_for_ICCV_2025_paper.pdf) and [supplemental material](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Li_FoundIR_Unleashing_Million-scale_ICCV_2025_supplemental.pdf))

### ⚡ How to train

#### Environment

```
conda env create --name foundirv2 -f environment.yml
```

#### Dataset Structure

Please place the dataset in the `dataset` directory. It must contain three subfolders: `HQ_folder`, `LQ_folder`, and `json_folder`. Each subfolder must include directories from `Task1` to `Task5`.

```
dataset/
├── HQ_folder/   # High-quality target images 
│   ├── Task1/
│   ├── ...
│   └── Task5/
├── LQ_folder/   # Low-quality input images 
│   ├── ...
└── json_folder/ # JSON files   
    ├── ...
```

**⚠️ Note:** The `Task5` folder is specifically reserved for **Super-Resolution** task images.

#### Training Script

```Shell
# Stage 1
bash train_stage_1.sh

# After Stage 1 training, enter the checkpoints folder.
cd ./train_foundirv2_stage_1_offline/checkpoint-6000
python zero_to_fp32.py ./ ./pretrain.bin

# Stage 2
bash train_stage_2.sh

# After Stage 2 training, enter the checkpoints folder.
cd ./train_foundirv2_stage_2_offline/checkpoint
python zero_to_fp32.py ./ ./FaithDiff.bin
```

### 🚀 How to evaluate

#### Download Dependent Models

- [FoundIR-v2 Pre-trained model](https://huggingface.co/cschenxiang/FoundIR-v2)
- [SDXL RealVisXL_V4.0](https://huggingface.co/SG161222/RealVisXL_V4.0)
- [SDXL VAE FP16](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)
- [LLaVA CLIP](https://huggingface.co/openai/clip-vit-large-patch14-336)
- [LLaVA v1.5 13B](https://huggingface.co/liuhaotian/llava-v1.5-13b)
- [BSRNet](https://drive.usercontent.google.com/download?id=1JGJLiENPkOqi39bvQYa_jlIPlMk24iKH&export=download&authuser=0&confirm=t&uuid=ebaa5d11-ac76-4f54-aabf-90fa43997dec&at=AEz70l4zk_8LTafpGtR0ZSE50F1N:1742369984793)
- Put them in the `./checkpoints` folder and update the corresponding path in CKPT_path.py.

#### Inference Script

```Shell
# Script that support two GPUs. 
CUDA_VISIBLE_DEVICES=0,1 python test.py --img_dir='./dataset/input' --save_dir=./save/output --guidance_scale=5 --num_inference_steps=20 --load_8bit_llava 

# Scripts that support only one GPU.
CUDA_VISIBLE_DEVICES=0 python test_generate_caption.py --img_dir='./dataset/input' --save_dir=./save/input_caption --load_8bit_llava
CUDA_VISIBLE_DEVICES=0 python test_wo_llava.py --img_dir='./dataset/input' --json_dir=./save/input_caption --save_dir=./save/output --guidance_scale=5 --num_inference_steps=20

```

`--upscale` (int): Upscaling factor. Default is `0` (Auto mode: images under 1080P are automatically upscaled by 2x). Set to a specific integer (e.g., `2`, `4`) for custom upscaling multipliers.


---



### Citation

If this work is helpful for your research, please consider citing the following BibTeX entry.
```
@inproceedings{foundirv2,
      title={FoundIR-v2: Optimizing Pre-Training Data Mixtures for Image Restoration Foundation Model},
      author={Chen, Xiang and Pan, Jinshan and Dong, Jiangxin and Yang, Jian and Tang, Jinhui},
      booktitle={CVPR},
      year={2026}
}
```
```
@inproceedings{foundir,
      title={FoundIR: Unleashing Million-scale Training Data to Advance Foundation Models for Image Restoration},
      author={Li, Hao and Chen, Xiang and Dong, Jiangxin and Tang, Jinhui and Pan, Jinshan},
      booktitle={ICCV},
      year={2025}
}
```

### Contact
If you have any questions, please feel free to reach us out at <a href="mailto:chenxiang@njust.edu.cn">chenxiang@njust.edu.cn</a>.

