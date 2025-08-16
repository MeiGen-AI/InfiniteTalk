import os
import folder_paths
from .utils import extract_specific_frames, strip_path, calculate_file_hash

file_extensions = ['webm', 'mp4', 'mkv', 'gif', 'mov', 'jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'webp']

class VideoFrameSample:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("IMAGE",),
                "stride": ("INT", { "default": 72, "step": 1, }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("IMAGE", )
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, video, stride):
        res = video[::stride]
        return (res,)


class ImageOrVideoUpload:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1].lower() in file_extensions):
                    files.append(f)
        return {"required": {
                    "input file": (sorted(files), {"video_upload": True}),
                    "stride": ("INT", { "default": 72, "step": 1, }),
                    },
                }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("IMAGE", )
    FUNCTION = "load_video"
    CATEGORY = "WanVideoWrapper"

    def load_video(self, **kwargs):
        input_file = folder_paths.get_annotated_filepath(strip_path(kwargs['input file']))
        stride = kwargs['stride']
        res = extract_specific_frames(input_file, stride) # F, H, W, C, torch.Size([3001, 1080, 1920, 3])
        return (res,)
    

    

NODE_CLASS_MAPPINGS = {
    "VideoFrameSample": VideoFrameSample,
    "ImageOrVideoUpload": ImageOrVideoUpload
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFrameSample": "WanVideo Reference Video Sample",
    "ImageOrVideoUpload": "WanVideo Reference Image/Video Upload"
}