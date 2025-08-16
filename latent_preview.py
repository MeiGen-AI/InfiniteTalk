import torch
from PIL import Image
from comfy.cli_args import args, LatentPreviewMethod
from comfy.taesd.taesd import TAESD
import comfy.model_management
import folder_paths
import comfy.utils
import logging
import os

from .taehv import TAEHV

MAX_PREVIEW_RESOLUTION = args.preview_size

def preview_to_image(latent_image):
        print("latent_image shape: ", latent_image.shape)#torch.Size([60, 104, 3])
        latents_ubyte = (((latent_image + 1.0) / 2.0).clamp(0, 1)  # change scale from -1..1 to 0..1
                            .mul(0xFF)  # to 0..255
                            )
        if comfy.model_management.directml_enabled:
                latents_ubyte = latents_ubyte.to(dtype=torch.uint8)
        latents_ubyte = latents_ubyte.to(device="cpu", dtype=torch.uint8, non_blocking=comfy.model_management.device_supports_non_blocking(latent_image.device))

        return Image.fromarray(latents_ubyte.numpy())

class LatentPreviewer:
    def decode_latent_to_preview(self, x0):
        pass

    def decode_latent_to_preview_image(self, preview_format, x0):
        preview_image = self.decode_latent_to_preview(x0)
        return ("JPEG", preview_image, MAX_PREVIEW_RESOLUTION)

class TAESDPreviewerImpl(LatentPreviewer):
    def __init__(self, taesd):
        self.taesd = taesd

    # def decode_latent_to_preview(self, x0):
    #     #x_sample = self.taesd.decode(x0[:1])[0].movedim(0, 2)
    #     print("x0 shape: ", x0.shape) #torch.Size([5, 16, 60, 104])
    #     x0 = x0.unsqueeze(0)
    #     print("x0 shape: ", x0.shape) #torch.Size([5, 16, 60, 104])
    #     x_sample = self.taesd.decode_video(x0, parallel=False)[0].permute(0, 2, 3, 1)[0]
    #     print("x_sample shape: ", x_sample.shape) 
    #     return preview_to_image(x_sample)


class Latent2RGBPreviewer(LatentPreviewer):
    def __init__(self, latent_rgb_factors, latent_rgb_factors_bias=None):
        self.latent_rgb_factors = torch.tensor(latent_rgb_factors, device="cpu").transpose(0, 1)
        self.latent_rgb_factors_bias = None
        if latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = torch.tensor(latent_rgb_factors_bias, device="cpu")

    def decode_latent_to_preview(self, x0):
        self.latent_rgb_factors = self.latent_rgb_factors.to(dtype=x0.dtype, device=x0.device)
        if self.latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = self.latent_rgb_factors_bias.to(dtype=x0.dtype, device=x0.device)

        if x0.ndim == 5:
            x0 = x0[0, :, 0]
        else:
            x0 = x0[0]

        latent_image = torch.nn.functional.linear(x0.movedim(0, -1), self.latent_rgb_factors, bias=self.latent_rgb_factors_bias)
        # latent_image = x0[0].permute(1, 2, 0) @ self.latent_rgb_factors

        return preview_to_image(latent_image)


def get_previewer(device, latent_format):
    previewer = None
    method = args.preview_method
    if method != LatentPreviewMethod.NoPreviews:
        # TODO previewer methods

        if method == LatentPreviewMethod.Auto:
            method = LatentPreviewMethod.Latent2RGB

        if method == LatentPreviewMethod.TAESD:
            taehv_path = os.path.join(folder_paths.models_dir, "vae_approx", "taew2_1.safetensors")
            if not os.path.exists(taehv_path):
                raise RuntimeError(f"Could not find {taehv_path}")
            taew_sd = comfy.utils.load_torch_file(taehv_path)
            taesd = TAEHV(taew_sd).to(device)
            previewer = TAESDPreviewerImpl(taesd)
            previewer = WrappedPreviewer(previewer, rate=16)

        if previewer is None:
            if latent_format.latent_rgb_factors is not None:
                previewer = Latent2RGBPreviewer(latent_format.latent_rgb_factors, latent_format.latent_rgb_factors_bias)
                previewer = WrappedPreviewer(previewer, rate=4)
    return previewer

def prepare_callback(model, steps, x0_output_dict=None):
    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    previewer = get_previewer(model.load_device, model.model.latent_format)

    if steps is not None:
        pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps):
        if x0_output_dict is not None:
            x0_output_dict["x0"] = x0

        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
        if step is not None:
            pbar.update_absolute(step + 1, total_steps, preview_bytes)
    return callback

#borrowed VideoHelperSuite https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/blob/main/videohelpersuite/latent_preview.py
import server
from threading import Thread
import torch.nn.functional as F
import io
import time
import struct
from importlib.util import find_spec
serv = server.PromptServer.instance

class WrappedPreviewer(LatentPreviewer):
    def __init__(self, previewer, rate=16):
        self.first_preview = True
        self.last_time = 0
        self.c_index = 0
        self.rate = rate
        self.swarmui_env = find_spec("SwarmComfyCommon") is not None
        if self.swarmui_env:
            print("previewer: SwarmUI output enabled")
        if hasattr(previewer, 'taesd'):
            self.taesd = previewer.taesd
        elif hasattr(previewer, 'latent_rgb_factors'):
            self.latent_rgb_factors = previewer.latent_rgb_factors
            self.latent_rgb_factors_bias = previewer.latent_rgb_factors_bias
        else:
            raise Exception('Unsupported preview type for VHS animated previews')

    def decode_latent_to_preview_image(self, preview_format, x0):
        if x0.ndim == 5:
            #Keep batch major
            x0 = x0.movedim(2,1)
            x0 = x0.reshape((-1,)+x0.shape[-3:])
        num_images = x0.size(0)
        new_time = time.time()
        num_previews = int((new_time - self.last_time) * self.rate)
        self.last_time = self.last_time + num_previews/self.rate
        if num_previews > num_images:
            num_previews = num_images
        elif num_previews <= 0:
            return None
        if self.first_preview:
            self.first_preview = False
            serv.send_sync('VHS_latentpreview', {'length':num_images, 'rate': self.rate})
            self.last_time = new_time + 1/self.rate
        if self.c_index + num_previews > num_images:
            x0 = x0.roll(-self.c_index, 0)[:num_previews]
        else:
            x0 = x0[self.c_index:self.c_index + num_previews]
        Thread(target=self.process_previews, args=(x0, self.c_index,
                                                   num_images)).run()
        self.c_index = (self.c_index + num_previews) % num_images
        return None
    def process_previews(self, image_tensor, ind, leng):
        max_size = 256
        image_tensor = self.decode_latent_to_preview(image_tensor)
        if image_tensor.size(1) > max_size or image_tensor.size(2) > max_size:
            image_tensor = image_tensor.movedim(-1,0)
            if image_tensor.size(2) < image_tensor.size(3):
                height = (max_size * image_tensor.size(2)) // image_tensor.size(3)
                image_tensor = F.interpolate(image_tensor, (height,max_size), mode='bilinear')
            else:
                width = (max_size * image_tensor.size(3)) // image_tensor.size(2)
                image_tensor = F.interpolate(image_tensor, (max_size, width), mode='bilinear')
            image_tensor = image_tensor.movedim(0,-1)
        previews_ubyte = (image_tensor.clamp(0, 1)
                         .mul(0xFF)  # to 0..255
                         ).to(device="cpu", dtype=torch.uint8)

        # Send VHS preview
        for preview in previews_ubyte:
            i = Image.fromarray(preview.numpy())
            message = io.BytesIO()
            message.write((1).to_bytes(length=4, byteorder='big')*2)
            message.write(ind.to_bytes(length=4, byteorder='big'))
            i.save(message, format="JPEG", quality=95, compress_level=1)
            #NOTE: send sync already uses call_soon_threadsafe
            serv.send_sync(server.BinaryEventTypes.PREVIEW_IMAGE,
                           message.getvalue(), serv.client_id)
            if self.rate == 16:
                ind = (ind + 1) % ((leng-1) * 4 - 1)
            else:
                ind = (ind + 1) % leng

        # Send SwarmUI preview if detected
        if self.swarmui_env:
            images = [Image.fromarray(preview.numpy()) for preview in previews_ubyte]
            message = io.BytesIO()
            header = struct.pack(">I", 3)
            message.write(header)
            images[0].save(
                message,
                save_all=True,
                duration=int(1000.0/self.rate),
                append_images=images[1 : len(images)],
                lossless=False,
                quality=80,
                method=0,
                format="WEBP",
            )
            message.seek(0)
            preview_bytes = message.getvalue()
            serv.send_sync(1, preview_bytes, sid=serv.client_id)
    def decode_latent_to_preview(self, x0):
        if hasattr(self, 'taesd'):
            x0 = x0.unsqueeze(0)
            x_sample = self.taesd.decode_video(x0, parallel=False, show_progress_bar=False)[0].permute(0, 2, 3, 1)
            return x_sample
        else:
            self.latent_rgb_factors = self.latent_rgb_factors.to(dtype=x0.dtype, device=x0.device)
            if self.latent_rgb_factors_bias is not None:
                self.latent_rgb_factors_bias = self.latent_rgb_factors_bias.to(dtype=x0.dtype, device=x0.device)
            latent_image = F.linear(x0.movedim(1, -1), self.latent_rgb_factors,
                                    bias=self.latent_rgb_factors_bias)
            latent_image = (latent_image + 1.0) / 2.0
            return latent_image 
        
