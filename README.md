# ComfyUI wrapper nodes for [InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk).

This project is based on [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper). The checkpoint for InfiniteTalk-ComfyUI can be found in [HuggingFace](https://huggingface.co/MeiGen-AI/InfiniteTalk/tree/main/comfyui).


# Installation
1. Clone this repo into `custom_nodes` folder.
2. Install dependencies: `pip install -r requirements.txt`
   or if you use the portable install, run this in ComfyUI_windows_portable -folder:

  `python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-WanVideoWrapper\requirements.txt`

## Models

https://huggingface.co/Kijai/WanVideo_comfy/tree/main

Text encoders to `ComfyUI/models/text_encoders`

Clip vision to `ComfyUI/models/clip_vision`

Transformer (main video model) to `ComfyUI/models/diffusion_models`

Vae to `ComfyUI/models/vae`

You can also use the native ComfyUI text encoding and clip vision loader with the wrapper instead of the original models:

![image](https://github.com/user-attachments/assets/6a2fd9a5-8163-4c93-b362-92ef34dbd3a4)

GGUF models can now be loaded in the main model loader as well.

---
Supported extra models:

SkyReels: https://huggingface.co/collections/Skywork/skyreels-v2-6801b1b93df627d441d0d0d9

WanVideoFun: https://huggingface.co/collections/alibaba-pai/wan21-fun-v11-680f514c89fe7b4df9d44f17

ReCamMaster: https://github.com/KwaiVGI/ReCamMaster

VACE: https://github.com/ali-vilab/VACE

Phantom: https://huggingface.co/bytedance-research/Phantom

ATI: https://huggingface.co/bytedance-research/ATI

Uni3C: https://github.com/alibaba-damo-academy/Uni3C

MiniMaxRemover: https://huggingface.co/zibojia/minimax-remover

MAGREF: https://huggingface.co/MAGREF-Video/MAGREF

FantasyTalking: https://github.com/Fantasy-AMAP/fantasy-talking

MultiTalk: https://github.com/MeiGen-AI/MultiTalk


Examples:
---

[ReCamMaster](https://github.com/KwaiVGI/ReCamMaster):

https://github.com/user-attachments/assets/c58a12c2-13ba-4af8-8041-e283dbef197e


TeaCache (with the old temporary WIP naive version, I2V):

**Note that with the new version the threshold values should be 10x higher**

Range of 0.25-0.30 seems good when using the coefficients, start step can be 0, with more aggressive threshold values it may make sense to start later to avoid any potential step skips early on, that generally ruin the motion.

https://github.com/user-attachments/assets/504a9a50-3337-43d2-97b8-8e1661f29f46


Context window test:

1025 frames using window size of 81 frames, with 16 overlap. With the 1.3B T2V model this used under 5GB VRAM and took 10 minutes to gen on a 5090:

https://github.com/user-attachments/assets/89b393af-cf1b-49ae-aa29-23e57f65911e

---


This very first test was 512x512x81

~16GB used with 20/40 blocks offloaded

https://github.com/user-attachments/assets/fa6d0a4f-4a4d-4de5-84a4-877cc37b715f

Vid2vid example:


with 14B T2V model:

https://github.com/user-attachments/assets/ef228b8a-a13a-4327-8a1b-1eb343cf00d8

with 1.3B T2V model

https://github.com/user-attachments/assets/4f35ba84-da7a-4d5b-97ee-9641296f391e



