# from decord import VideoReader, cpu
import os
from PIL import Image
import gc
import numpy as np
import torch
import cv2
import hashlib

def is_video(path):
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg']
    return os.path.splitext(path)[1].lower() in video_exts

def extract_specific_frames(video_path, stride):
    frames = []
    if is_video(video_path):
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = (frame_rgb / 255.0).astype(np.float32)
                frames.append(frame_rgb)
            frame_idx += 1
        cap.release()
        try:
            frame = Image.fromarray(frame)
        except:
            print(frame_idx)

    else:
        img = Image.open(video_path).convert('RGB')
        frame = np.array(img)
        frames.append(frame)
    frames_np = np.stack(frames, axis=0)
    frames_tensor = torch.from_numpy(frames_np)
    return frames_tensor

def strip_path(path):
    #This leaves whitespace inside quotes and only a single "
    #thus ' ""test"' -> '"test'
    #consider path.strip(string.whitespace+"\"")
    #or weightier re.fullmatch("[\\s\"]*(.+?)[\\s\"]*", path).group(1)
    path = path.strip()
    if path.startswith("\""):
        path = path[1:]
    if path.endswith("\""):
        path = path[:-1]
    return path

# modified from https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
def calculate_file_hash(filename: str, hash_every_n: int = 1):
    #Larger video files were taking >.5 seconds to hash even when cached,
    #so instead the modified time from the filesystem is used as a hash
    h = hashlib.sha256()
    h.update(filename.encode())
    h.update(str(os.path.getmtime(filename)).encode())
    return h.hexdigest()