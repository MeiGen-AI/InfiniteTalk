
import torch.nn as nn
from ..utils import log
import comfy.model_management as mm
from comfy.utils import ProgressBar
from tqdm import tqdm

def update_transformer(transformer, state_dict):
    
    concat_dim = 4
    transformer.dwpose_embedding = nn.Sequential(
                nn.Conv3d(3, concat_dim * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                nn.SiLU(),
                nn.Conv3d(concat_dim * 4, concat_dim * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                nn.SiLU(),
                nn.Conv3d(concat_dim * 4, concat_dim * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                nn.SiLU(),
                nn.Conv3d(concat_dim * 4, concat_dim * 4, (3,3,3), stride=(1,2,2), padding=(1,1,1)),
                nn.SiLU(),
                nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2,2,2), padding=1),
                nn.SiLU(),
                nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2,2,2), padding=1),
                nn.SiLU(),
                nn.Conv3d(concat_dim * 4, 5120, (1,2,2), stride=(1,2,2), padding=0))

    randomref_dim = 20
    transformer.randomref_embedding_pose = nn.Sequential(
                nn.Conv2d(3, concat_dim * 4, 3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, randomref_dim, 3, stride=2, padding=1),
                )
    state_dict_new = {}
    for key in list(state_dict.keys()):
        if "dwpose_embedding" in key:
            state_dict_new[key.split("dwpose_embedding.")[1]] = state_dict.pop(key)
    transformer.dwpose_embedding.load_state_dict(state_dict_new, strict=True)
    state_dict_new = {}
    for key in list(state_dict.keys()):
        if "randomref_embedding_pose" in key:
            state_dict_new[key.split("randomref_embedding_pose.")[1]] = state_dict.pop(key)
    transformer.randomref_embedding_pose.load_state_dict(state_dict_new,strict=True)
    return transformer

# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
import torch
import numpy as np
import copy
import torch
import numpy as np
import math

from .dwpose.wholebody import Wholebody

def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = x0
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

class DWposeDetector:
    def __init__(self, model_det, model_pose):
        self.pose_estimation = Wholebody(model_det, model_pose)

    def __call__(self, oriImg, score_threshold=0.3):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(oriImg)
            candidate = candidate[0][np.newaxis, :, :]
            subset = subset[0][np.newaxis, :]
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:,:18].copy()
            body = body.reshape(nums*18, locs)
            score = subset[:,:18].copy()
            
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > score_threshold:
                        score[i][j] = int(18*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset<score_threshold
            candidate[un_visible] = -1

            bodyfoot_score = subset[:,:24].copy()
            for i in range(len(bodyfoot_score)):
                for j in range(len(bodyfoot_score[i])):
                    if bodyfoot_score[i][j] > score_threshold:
                        bodyfoot_score[i][j] = int(18*i+j)
                    else:
                        bodyfoot_score[i][j] = -1
            if -1 not in bodyfoot_score[:,18] and -1 not in bodyfoot_score[:,19]:
                bodyfoot_score[:,18] = np.array([18.]) 
            else:
                bodyfoot_score[:,18] = np.array([-1.])
            if -1 not in bodyfoot_score[:,21] and -1 not in bodyfoot_score[:,22]:
                bodyfoot_score[:,19] = np.array([19.]) 
            else:
                bodyfoot_score[:,19] = np.array([-1.])
            bodyfoot_score = bodyfoot_score[:, :20]

            bodyfoot = candidate[:,:24].copy()
            
            for i in range(nums):
                if -1 not in bodyfoot[i][18] and -1 not in bodyfoot[i][19]:
                    bodyfoot[i][18] = (bodyfoot[i][18]+bodyfoot[i][19])/2
                else:
                    bodyfoot[i][18] = np.array([-1., -1.])
                if -1 not in bodyfoot[i][21] and -1 not in bodyfoot[i][22]:
                    bodyfoot[i][19] = (bodyfoot[i][21]+bodyfoot[i][22])/2
                else:
                    bodyfoot[i][19] = np.array([-1., -1.])
            
            bodyfoot = bodyfoot[:,:20,:]
            bodyfoot = bodyfoot.reshape(nums*20, locs)

            foot = candidate[:,18:24]

            faces = candidate[:,24:92]

            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])
            
            # bodies = dict(candidate=body, subset=score)
            bodies = dict(candidate=bodyfoot, subset=bodyfoot_score, score=bodyfoot_score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            # return draw_pose(pose, H, W)
            return pose

def draw_pose(pose, H, W, stick_width=4,draw_body=True, draw_hands=True, draw_feet=True, 
              body_keypoint_size=4, hand_keypoint_size=4, draw_head=True):
    from .dwpose.util import draw_body_and_foot, draw_handpose, draw_facepose
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    score=bodies['score']

    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    canvas = draw_body_and_foot(canvas, candidate, subset, score, draw_body=draw_body, stick_width=stick_width, draw_feet=draw_feet, draw_head=draw_head, body_keypoint_size=body_keypoint_size)
    canvas = draw_handpose(canvas, hands, draw_hands=draw_hands, hand_keypoint_size=hand_keypoint_size)
    canvas_without_face = copy.deepcopy(canvas)
    canvas = draw_facepose(canvas, faces)

    return canvas_without_face, canvas


def pose_extract(pose_images, ref_image, dwpose_model, height, width, score_threshold, stick_width,
                 draw_body=True, draw_hands=True, hand_keypoint_size=4, draw_feet=True,
                 body_keypoint_size=4, handle_not_detected="repeat", draw_head=True):
    
    results_vis = []
    comfy_pbar = ProgressBar(len(pose_images))

    if ref_image is not None:
        try:
            pose_ref = dwpose_model(ref_image.squeeze(0), score_threshold=score_threshold)
        except:
            raise ValueError("No pose detected in reference image")
    prev_pose = None
    for img in tqdm(pose_images, desc="Pose Extraction", unit="image", total=len(pose_images)):
        try:
            pose = dwpose_model(img, score_threshold=score_threshold)
            if handle_not_detected == "repeat":
                prev_pose = pose
        except:
            if prev_pose is not None:
                pose = prev_pose
            else:
                pose = np.zeros_like(img)
        results_vis.append(pose)
        comfy_pbar.update(1)
    
    bodies = results_vis[0]['bodies']
    faces = results_vis[0]['faces']
    hands = results_vis[0]['hands']
    candidate = bodies['candidate']

    if ref_image is not None:
        ref_bodies = pose_ref['bodies']
        ref_faces = pose_ref['faces']
        ref_hands = pose_ref['hands']
        ref_candidate = ref_bodies['candidate']


        ref_2_x = ref_candidate[2][0]
        ref_2_y = ref_candidate[2][1]
        ref_5_x = ref_candidate[5][0]
        ref_5_y = ref_candidate[5][1]
        ref_8_x = ref_candidate[8][0]
        ref_8_y = ref_candidate[8][1]
        ref_11_x = ref_candidate[11][0]
        ref_11_y = ref_candidate[11][1]
        ref_center1 = 0.5*(ref_candidate[2]+ref_candidate[5])
        ref_center2 = 0.5*(ref_candidate[8]+ref_candidate[11])

        zero_2_x = candidate[2][0]
        zero_2_y = candidate[2][1]
        zero_5_x = candidate[5][0]
        zero_5_y = candidate[5][1]
        zero_8_x = candidate[8][0]
        zero_8_y = candidate[8][1]
        zero_11_x = candidate[11][0]
        zero_11_y = candidate[11][1]
        zero_center1 = 0.5*(candidate[2]+candidate[5])
        zero_center2 = 0.5*(candidate[8]+candidate[11])

        x_ratio = (ref_5_x-ref_2_x)/(zero_5_x-zero_2_x)
        y_ratio = (ref_center2[1]-ref_center1[1])/(zero_center2[1]-zero_center1[1])

        results_vis[0]['bodies']['candidate'][:,0] *= x_ratio
        results_vis[0]['bodies']['candidate'][:,1] *= y_ratio
        results_vis[0]['faces'][:,:,0] *= x_ratio
        results_vis[0]['faces'][:,:,1] *= y_ratio
        results_vis[0]['hands'][:,:,0] *= x_ratio
        results_vis[0]['hands'][:,:,1] *= y_ratio
        
        ########neck########
        l_neck_ref = ((ref_candidate[0][0] - ref_candidate[1][0]) ** 2 + (ref_candidate[0][1] - ref_candidate[1][1]) ** 2) ** 0.5
        l_neck_0 = ((candidate[0][0] - candidate[1][0]) ** 2 + (candidate[0][1] - candidate[1][1]) ** 2) ** 0.5
        neck_ratio = l_neck_ref / l_neck_0

        x_offset_neck = (candidate[1][0]-candidate[0][0])*(1.-neck_ratio)
        y_offset_neck = (candidate[1][1]-candidate[0][1])*(1.-neck_ratio)

        results_vis[0]['bodies']['candidate'][0,0] += x_offset_neck
        results_vis[0]['bodies']['candidate'][0,1] += y_offset_neck
        results_vis[0]['bodies']['candidate'][14,0] += x_offset_neck
        results_vis[0]['bodies']['candidate'][14,1] += y_offset_neck
        results_vis[0]['bodies']['candidate'][15,0] += x_offset_neck
        results_vis[0]['bodies']['candidate'][15,1] += y_offset_neck
        results_vis[0]['bodies']['candidate'][16,0] += x_offset_neck
        results_vis[0]['bodies']['candidate'][16,1] += y_offset_neck
        results_vis[0]['bodies']['candidate'][17,0] += x_offset_neck
        results_vis[0]['bodies']['candidate'][17,1] += y_offset_neck
        
        ########shoulder2########
        l_shoulder2_ref = ((ref_candidate[2][0] - ref_candidate[1][0]) ** 2 + (ref_candidate[2][1] - ref_candidate[1][1]) ** 2) ** 0.5
        l_shoulder2_0 = ((candidate[2][0] - candidate[1][0]) ** 2 + (candidate[2][1] - candidate[1][1]) ** 2) ** 0.5

        shoulder2_ratio = l_shoulder2_ref / l_shoulder2_0

        x_offset_shoulder2 = (candidate[1][0]-candidate[2][0])*(1.-shoulder2_ratio)
        y_offset_shoulder2 = (candidate[1][1]-candidate[2][1])*(1.-shoulder2_ratio)

        results_vis[0]['bodies']['candidate'][2,0] += x_offset_shoulder2
        results_vis[0]['bodies']['candidate'][2,1] += y_offset_shoulder2
        results_vis[0]['bodies']['candidate'][3,0] += x_offset_shoulder2
        results_vis[0]['bodies']['candidate'][3,1] += y_offset_shoulder2
        results_vis[0]['bodies']['candidate'][4,0] += x_offset_shoulder2
        results_vis[0]['bodies']['candidate'][4,1] += y_offset_shoulder2
        results_vis[0]['hands'][1,:,0] += x_offset_shoulder2
        results_vis[0]['hands'][1,:,1] += y_offset_shoulder2

        ########shoulder5########
        l_shoulder5_ref = ((ref_candidate[5][0] - ref_candidate[1][0]) ** 2 + (ref_candidate[5][1] - ref_candidate[1][1]) ** 2) ** 0.5
        l_shoulder5_0 = ((candidate[5][0] - candidate[1][0]) ** 2 + (candidate[5][1] - candidate[1][1]) ** 2) ** 0.5

        shoulder5_ratio = l_shoulder5_ref / l_shoulder5_0

        x_offset_shoulder5 = (candidate[1][0]-candidate[5][0])*(1.-shoulder5_ratio)
        y_offset_shoulder5 = (candidate[1][1]-candidate[5][1])*(1.-shoulder5_ratio)

        results_vis[0]['bodies']['candidate'][5,0] += x_offset_shoulder5
        results_vis[0]['bodies']['candidate'][5,1] += y_offset_shoulder5
        results_vis[0]['bodies']['candidate'][6,0] += x_offset_shoulder5
        results_vis[0]['bodies']['candidate'][6,1] += y_offset_shoulder5
        results_vis[0]['bodies']['candidate'][7,0] += x_offset_shoulder5
        results_vis[0]['bodies']['candidate'][7,1] += y_offset_shoulder5
        results_vis[0]['hands'][0,:,0] += x_offset_shoulder5
        results_vis[0]['hands'][0,:,1] += y_offset_shoulder5

        ########arm3########
        l_arm3_ref = ((ref_candidate[3][0] - ref_candidate[2][0]) ** 2 + (ref_candidate[3][1] - ref_candidate[2][1]) ** 2) ** 0.5
        l_arm3_0 = ((candidate[3][0] - candidate[2][0]) ** 2 + (candidate[3][1] - candidate[2][1]) ** 2) ** 0.5

        arm3_ratio = l_arm3_ref / l_arm3_0

        x_offset_arm3 = (candidate[2][0]-candidate[3][0])*(1.-arm3_ratio)
        y_offset_arm3 = (candidate[2][1]-candidate[3][1])*(1.-arm3_ratio)

        results_vis[0]['bodies']['candidate'][3,0] += x_offset_arm3
        results_vis[0]['bodies']['candidate'][3,1] += y_offset_arm3
        results_vis[0]['bodies']['candidate'][4,0] += x_offset_arm3
        results_vis[0]['bodies']['candidate'][4,1] += y_offset_arm3
        results_vis[0]['hands'][1,:,0] += x_offset_arm3
        results_vis[0]['hands'][1,:,1] += y_offset_arm3

        ########arm4########
        l_arm4_ref = ((ref_candidate[4][0] - ref_candidate[3][0]) ** 2 + (ref_candidate[4][1] - ref_candidate[3][1]) ** 2) ** 0.5
        l_arm4_0 = ((candidate[4][0] - candidate[3][0]) ** 2 + (candidate[4][1] - candidate[3][1]) ** 2) ** 0.5

        arm4_ratio = l_arm4_ref / l_arm4_0

        x_offset_arm4 = (candidate[3][0]-candidate[4][0])*(1.-arm4_ratio)
        y_offset_arm4 = (candidate[3][1]-candidate[4][1])*(1.-arm4_ratio)

        results_vis[0]['bodies']['candidate'][4,0] += x_offset_arm4
        results_vis[0]['bodies']['candidate'][4,1] += y_offset_arm4
        results_vis[0]['hands'][1,:,0] += x_offset_arm4
        results_vis[0]['hands'][1,:,1] += y_offset_arm4

        ########arm6########
        l_arm6_ref = ((ref_candidate[6][0] - ref_candidate[5][0]) ** 2 + (ref_candidate[6][1] - ref_candidate[5][1]) ** 2) ** 0.5
        l_arm6_0 = ((candidate[6][0] - candidate[5][0]) ** 2 + (candidate[6][1] - candidate[5][1]) ** 2) ** 0.5

        arm6_ratio = l_arm6_ref / l_arm6_0

        x_offset_arm6 = (candidate[5][0]-candidate[6][0])*(1.-arm6_ratio)
        y_offset_arm6 = (candidate[5][1]-candidate[6][1])*(1.-arm6_ratio)

        results_vis[0]['bodies']['candidate'][6,0] += x_offset_arm6
        results_vis[0]['bodies']['candidate'][6,1] += y_offset_arm6
        results_vis[0]['bodies']['candidate'][7,0] += x_offset_arm6
        results_vis[0]['bodies']['candidate'][7,1] += y_offset_arm6
        results_vis[0]['hands'][0,:,0] += x_offset_arm6
        results_vis[0]['hands'][0,:,1] += y_offset_arm6

        ########arm7########
        l_arm7_ref = ((ref_candidate[7][0] - ref_candidate[6][0]) ** 2 + (ref_candidate[7][1] - ref_candidate[6][1]) ** 2) ** 0.5
        l_arm7_0 = ((candidate[7][0] - candidate[6][0]) ** 2 + (candidate[7][1] - candidate[6][1]) ** 2) ** 0.5

        arm7_ratio = l_arm7_ref / l_arm7_0

        x_offset_arm7 = (candidate[6][0]-candidate[7][0])*(1.-arm7_ratio)
        y_offset_arm7 = (candidate[6][1]-candidate[7][1])*(1.-arm7_ratio)

        results_vis[0]['bodies']['candidate'][7,0] += x_offset_arm7
        results_vis[0]['bodies']['candidate'][7,1] += y_offset_arm7
        results_vis[0]['hands'][0,:,0] += x_offset_arm7
        results_vis[0]['hands'][0,:,1] += y_offset_arm7

        ########head14########
        l_head14_ref = ((ref_candidate[14][0] - ref_candidate[0][0]) ** 2 + (ref_candidate[14][1] - ref_candidate[0][1]) ** 2) ** 0.5
        l_head14_0 = ((candidate[14][0] - candidate[0][0]) ** 2 + (candidate[14][1] - candidate[0][1]) ** 2) ** 0.5

        head14_ratio = l_head14_ref / l_head14_0

        x_offset_head14 = (candidate[0][0]-candidate[14][0])*(1.-head14_ratio)
        y_offset_head14 = (candidate[0][1]-candidate[14][1])*(1.-head14_ratio)

        results_vis[0]['bodies']['candidate'][14,0] += x_offset_head14
        results_vis[0]['bodies']['candidate'][14,1] += y_offset_head14
        results_vis[0]['bodies']['candidate'][16,0] += x_offset_head14
        results_vis[0]['bodies']['candidate'][16,1] += y_offset_head14

        ########head15########
        l_head15_ref = ((ref_candidate[15][0] - ref_candidate[0][0]) ** 2 + (ref_candidate[15][1] - ref_candidate[0][1]) ** 2) ** 0.5
        l_head15_0 = ((candidate[15][0] - candidate[0][0]) ** 2 + (candidate[15][1] - candidate[0][1]) ** 2) ** 0.5

        head15_ratio = l_head15_ref / l_head15_0

        x_offset_head15 = (candidate[0][0]-candidate[15][0])*(1.-head15_ratio)
        y_offset_head15 = (candidate[0][1]-candidate[15][1])*(1.-head15_ratio)

        results_vis[0]['bodies']['candidate'][15,0] += x_offset_head15
        results_vis[0]['bodies']['candidate'][15,1] += y_offset_head15
        results_vis[0]['bodies']['candidate'][17,0] += x_offset_head15
        results_vis[0]['bodies']['candidate'][17,1] += y_offset_head15

        ########head16########
        l_head16_ref = ((ref_candidate[16][0] - ref_candidate[14][0]) ** 2 + (ref_candidate[16][1] - ref_candidate[14][1]) ** 2) ** 0.5
        l_head16_0 = ((candidate[16][0] - candidate[14][0]) ** 2 + (candidate[16][1] - candidate[14][1]) ** 2) ** 0.5

        head16_ratio = l_head16_ref / l_head16_0

        x_offset_head16 = (candidate[14][0]-candidate[16][0])*(1.-head16_ratio)
        y_offset_head16 = (candidate[14][1]-candidate[16][1])*(1.-head16_ratio)

        results_vis[0]['bodies']['candidate'][16,0] += x_offset_head16
        results_vis[0]['bodies']['candidate'][16,1] += y_offset_head16

        ########head17########
        l_head17_ref = ((ref_candidate[17][0] - ref_candidate[15][0]) ** 2 + (ref_candidate[17][1] - ref_candidate[15][1]) ** 2) ** 0.5
        l_head17_0 = ((candidate[17][0] - candidate[15][0]) ** 2 + (candidate[17][1] - candidate[15][1]) ** 2) ** 0.5

        head17_ratio = l_head17_ref / l_head17_0

        x_offset_head17 = (candidate[15][0]-candidate[17][0])*(1.-head17_ratio)
        y_offset_head17 = (candidate[15][1]-candidate[17][1])*(1.-head17_ratio)

        results_vis[0]['bodies']['candidate'][17,0] += x_offset_head17
        results_vis[0]['bodies']['candidate'][17,1] += y_offset_head17
        
        ########MovingAverage########
        
        ########left leg########
        l_ll1_ref = ((ref_candidate[8][0] - ref_candidate[9][0]) ** 2 + (ref_candidate[8][1] - ref_candidate[9][1]) ** 2) ** 0.5
        l_ll1_0 = ((candidate[8][0] - candidate[9][0]) ** 2 + (candidate[8][1] - candidate[9][1]) ** 2) ** 0.5
        ll1_ratio = l_ll1_ref / l_ll1_0

        x_offset_ll1 = (candidate[9][0]-candidate[8][0])*(ll1_ratio-1.)
        y_offset_ll1 = (candidate[9][1]-candidate[8][1])*(ll1_ratio-1.)

        results_vis[0]['bodies']['candidate'][9,0] += x_offset_ll1
        results_vis[0]['bodies']['candidate'][9,1] += y_offset_ll1
        results_vis[0]['bodies']['candidate'][10,0] += x_offset_ll1
        results_vis[0]['bodies']['candidate'][10,1] += y_offset_ll1
        results_vis[0]['bodies']['candidate'][19,0] += x_offset_ll1
        results_vis[0]['bodies']['candidate'][19,1] += y_offset_ll1

        l_ll2_ref = ((ref_candidate[9][0] - ref_candidate[10][0]) ** 2 + (ref_candidate[9][1] - ref_candidate[10][1]) ** 2) ** 0.5
        l_ll2_0 = ((candidate[9][0] - candidate[10][0]) ** 2 + (candidate[9][1] - candidate[10][1]) ** 2) ** 0.5
        ll2_ratio = l_ll2_ref / l_ll2_0

        x_offset_ll2 = (candidate[10][0]-candidate[9][0])*(ll2_ratio-1.)
        y_offset_ll2 = (candidate[10][1]-candidate[9][1])*(ll2_ratio-1.)

        results_vis[0]['bodies']['candidate'][10,0] += x_offset_ll2
        results_vis[0]['bodies']['candidate'][10,1] += y_offset_ll2
        results_vis[0]['bodies']['candidate'][19,0] += x_offset_ll2
        results_vis[0]['bodies']['candidate'][19,1] += y_offset_ll2

        ########right leg########
        l_rl1_ref = ((ref_candidate[11][0] - ref_candidate[12][0]) ** 2 + (ref_candidate[11][1] - ref_candidate[12][1]) ** 2) ** 0.5
        l_rl1_0 = ((candidate[11][0] - candidate[12][0]) ** 2 + (candidate[11][1] - candidate[12][1]) ** 2) ** 0.5
        rl1_ratio = l_rl1_ref / l_rl1_0

        x_offset_rl1 = (candidate[12][0]-candidate[11][0])*(rl1_ratio-1.)
        y_offset_rl1 = (candidate[12][1]-candidate[11][1])*(rl1_ratio-1.)

        results_vis[0]['bodies']['candidate'][12,0] += x_offset_rl1
        results_vis[0]['bodies']['candidate'][12,1] += y_offset_rl1
        results_vis[0]['bodies']['candidate'][13,0] += x_offset_rl1
        results_vis[0]['bodies']['candidate'][13,1] += y_offset_rl1
        results_vis[0]['bodies']['candidate'][18,0] += x_offset_rl1
        results_vis[0]['bodies']['candidate'][18,1] += y_offset_rl1

        l_rl2_ref = ((ref_candidate[12][0] - ref_candidate[13][0]) ** 2 + (ref_candidate[12][1] - ref_candidate[13][1]) ** 2) ** 0.5
        l_rl2_0 = ((candidate[12][0] - candidate[13][0]) ** 2 + (candidate[12][1] - candidate[13][1]) ** 2) ** 0.5
        rl2_ratio = l_rl2_ref / l_rl2_0

        x_offset_rl2 = (candidate[13][0]-candidate[12][0])*(rl2_ratio-1.)
        y_offset_rl2 = (candidate[13][1]-candidate[12][1])*(rl2_ratio-1.)

        results_vis[0]['bodies']['candidate'][13,0] += x_offset_rl2
        results_vis[0]['bodies']['candidate'][13,1] += y_offset_rl2
        results_vis[0]['bodies']['candidate'][18,0] += x_offset_rl2
        results_vis[0]['bodies']['candidate'][18,1] += y_offset_rl2

        offset = ref_candidate[1] - results_vis[0]['bodies']['candidate'][1]

        results_vis[0]['bodies']['candidate'] += offset[np.newaxis, :]
        results_vis[0]['faces'] += offset[np.newaxis, np.newaxis, :]
        results_vis[0]['hands'] += offset[np.newaxis, np.newaxis, :]

        for i in range(1, len(results_vis)):
            results_vis[i]['bodies']['candidate'][:,0] *= x_ratio
            results_vis[i]['bodies']['candidate'][:,1] *= y_ratio
            results_vis[i]['faces'][:,:,0] *= x_ratio
            results_vis[i]['faces'][:,:,1] *= y_ratio
            results_vis[i]['hands'][:,:,0] *= x_ratio
            results_vis[i]['hands'][:,:,1] *= y_ratio

            ########neck########
            x_offset_neck = (results_vis[i]['bodies']['candidate'][1][0]-results_vis[i]['bodies']['candidate'][0][0])*(1.-neck_ratio)
            y_offset_neck = (results_vis[i]['bodies']['candidate'][1][1]-results_vis[i]['bodies']['candidate'][0][1])*(1.-neck_ratio)

            results_vis[i]['bodies']['candidate'][0,0] += x_offset_neck
            results_vis[i]['bodies']['candidate'][0,1] += y_offset_neck
            results_vis[i]['bodies']['candidate'][14,0] += x_offset_neck
            results_vis[i]['bodies']['candidate'][14,1] += y_offset_neck
            results_vis[i]['bodies']['candidate'][15,0] += x_offset_neck
            results_vis[i]['bodies']['candidate'][15,1] += y_offset_neck
            results_vis[i]['bodies']['candidate'][16,0] += x_offset_neck
            results_vis[i]['bodies']['candidate'][16,1] += y_offset_neck
            results_vis[i]['bodies']['candidate'][17,0] += x_offset_neck
            results_vis[i]['bodies']['candidate'][17,1] += y_offset_neck

            ########shoulder2########
            

            x_offset_shoulder2 = (results_vis[i]['bodies']['candidate'][1][0]-results_vis[i]['bodies']['candidate'][2][0])*(1.-shoulder2_ratio)
            y_offset_shoulder2 = (results_vis[i]['bodies']['candidate'][1][1]-results_vis[i]['bodies']['candidate'][2][1])*(1.-shoulder2_ratio)

            results_vis[i]['bodies']['candidate'][2,0] += x_offset_shoulder2
            results_vis[i]['bodies']['candidate'][2,1] += y_offset_shoulder2
            results_vis[i]['bodies']['candidate'][3,0] += x_offset_shoulder2
            results_vis[i]['bodies']['candidate'][3,1] += y_offset_shoulder2
            results_vis[i]['bodies']['candidate'][4,0] += x_offset_shoulder2
            results_vis[i]['bodies']['candidate'][4,1] += y_offset_shoulder2
            results_vis[i]['hands'][1,:,0] += x_offset_shoulder2
            results_vis[i]['hands'][1,:,1] += y_offset_shoulder2

            ########shoulder5########

            x_offset_shoulder5 = (results_vis[i]['bodies']['candidate'][1][0]-results_vis[i]['bodies']['candidate'][5][0])*(1.-shoulder5_ratio)
            y_offset_shoulder5 = (results_vis[i]['bodies']['candidate'][1][1]-results_vis[i]['bodies']['candidate'][5][1])*(1.-shoulder5_ratio)

            results_vis[i]['bodies']['candidate'][5,0] += x_offset_shoulder5
            results_vis[i]['bodies']['candidate'][5,1] += y_offset_shoulder5
            results_vis[i]['bodies']['candidate'][6,0] += x_offset_shoulder5
            results_vis[i]['bodies']['candidate'][6,1] += y_offset_shoulder5
            results_vis[i]['bodies']['candidate'][7,0] += x_offset_shoulder5
            results_vis[i]['bodies']['candidate'][7,1] += y_offset_shoulder5
            results_vis[i]['hands'][0,:,0] += x_offset_shoulder5
            results_vis[i]['hands'][0,:,1] += y_offset_shoulder5

            ########arm3########

            x_offset_arm3 = (results_vis[i]['bodies']['candidate'][2][0]-results_vis[i]['bodies']['candidate'][3][0])*(1.-arm3_ratio)
            y_offset_arm3 = (results_vis[i]['bodies']['candidate'][2][1]-results_vis[i]['bodies']['candidate'][3][1])*(1.-arm3_ratio)

            results_vis[i]['bodies']['candidate'][3,0] += x_offset_arm3
            results_vis[i]['bodies']['candidate'][3,1] += y_offset_arm3
            results_vis[i]['bodies']['candidate'][4,0] += x_offset_arm3
            results_vis[i]['bodies']['candidate'][4,1] += y_offset_arm3
            results_vis[i]['hands'][1,:,0] += x_offset_arm3
            results_vis[i]['hands'][1,:,1] += y_offset_arm3

            ########arm4########

            x_offset_arm4 = (results_vis[i]['bodies']['candidate'][3][0]-results_vis[i]['bodies']['candidate'][4][0])*(1.-arm4_ratio)
            y_offset_arm4 = (results_vis[i]['bodies']['candidate'][3][1]-results_vis[i]['bodies']['candidate'][4][1])*(1.-arm4_ratio)

            results_vis[i]['bodies']['candidate'][4,0] += x_offset_arm4
            results_vis[i]['bodies']['candidate'][4,1] += y_offset_arm4
            results_vis[i]['hands'][1,:,0] += x_offset_arm4
            results_vis[i]['hands'][1,:,1] += y_offset_arm4

            ########arm6########

            x_offset_arm6 = (results_vis[i]['bodies']['candidate'][5][0]-results_vis[i]['bodies']['candidate'][6][0])*(1.-arm6_ratio)
            y_offset_arm6 = (results_vis[i]['bodies']['candidate'][5][1]-results_vis[i]['bodies']['candidate'][6][1])*(1.-arm6_ratio)

            results_vis[i]['bodies']['candidate'][6,0] += x_offset_arm6
            results_vis[i]['bodies']['candidate'][6,1] += y_offset_arm6
            results_vis[i]['bodies']['candidate'][7,0] += x_offset_arm6
            results_vis[i]['bodies']['candidate'][7,1] += y_offset_arm6
            results_vis[i]['hands'][0,:,0] += x_offset_arm6
            results_vis[i]['hands'][0,:,1] += y_offset_arm6

            ########arm7########

            x_offset_arm7 = (results_vis[i]['bodies']['candidate'][6][0]-results_vis[i]['bodies']['candidate'][7][0])*(1.-arm7_ratio)
            y_offset_arm7 = (results_vis[i]['bodies']['candidate'][6][1]-results_vis[i]['bodies']['candidate'][7][1])*(1.-arm7_ratio)

            results_vis[i]['bodies']['candidate'][7,0] += x_offset_arm7
            results_vis[i]['bodies']['candidate'][7,1] += y_offset_arm7
            results_vis[i]['hands'][0,:,0] += x_offset_arm7
            results_vis[i]['hands'][0,:,1] += y_offset_arm7

            ########head14########

            x_offset_head14 = (results_vis[i]['bodies']['candidate'][0][0]-results_vis[i]['bodies']['candidate'][14][0])*(1.-head14_ratio)
            y_offset_head14 = (results_vis[i]['bodies']['candidate'][0][1]-results_vis[i]['bodies']['candidate'][14][1])*(1.-head14_ratio)

            results_vis[i]['bodies']['candidate'][14,0] += x_offset_head14
            results_vis[i]['bodies']['candidate'][14,1] += y_offset_head14
            results_vis[i]['bodies']['candidate'][16,0] += x_offset_head14
            results_vis[i]['bodies']['candidate'][16,1] += y_offset_head14

            ########head15########

            x_offset_head15 = (results_vis[i]['bodies']['candidate'][0][0]-results_vis[i]['bodies']['candidate'][15][0])*(1.-head15_ratio)
            y_offset_head15 = (results_vis[i]['bodies']['candidate'][0][1]-results_vis[i]['bodies']['candidate'][15][1])*(1.-head15_ratio)

            results_vis[i]['bodies']['candidate'][15,0] += x_offset_head15
            results_vis[i]['bodies']['candidate'][15,1] += y_offset_head15
            results_vis[i]['bodies']['candidate'][17,0] += x_offset_head15
            results_vis[i]['bodies']['candidate'][17,1] += y_offset_head15

            ########head16########

            x_offset_head16 = (results_vis[i]['bodies']['candidate'][14][0]-results_vis[i]['bodies']['candidate'][16][0])*(1.-head16_ratio)
            y_offset_head16 = (results_vis[i]['bodies']['candidate'][14][1]-results_vis[i]['bodies']['candidate'][16][1])*(1.-head16_ratio)

            results_vis[i]['bodies']['candidate'][16,0] += x_offset_head16
            results_vis[i]['bodies']['candidate'][16,1] += y_offset_head16

            ########head17########
            x_offset_head17 = (results_vis[i]['bodies']['candidate'][15][0]-results_vis[i]['bodies']['candidate'][17][0])*(1.-head17_ratio)
            y_offset_head17 = (results_vis[i]['bodies']['candidate'][15][1]-results_vis[i]['bodies']['candidate'][17][1])*(1.-head17_ratio)

            results_vis[i]['bodies']['candidate'][17,0] += x_offset_head17
            results_vis[i]['bodies']['candidate'][17,1] += y_offset_head17

            # ########MovingAverage########

            ########left leg########
            x_offset_ll1 = (results_vis[i]['bodies']['candidate'][9][0]-results_vis[i]['bodies']['candidate'][8][0])*(ll1_ratio-1.)
            y_offset_ll1 = (results_vis[i]['bodies']['candidate'][9][1]-results_vis[i]['bodies']['candidate'][8][1])*(ll1_ratio-1.)

            results_vis[i]['bodies']['candidate'][9,0] += x_offset_ll1
            results_vis[i]['bodies']['candidate'][9,1] += y_offset_ll1
            results_vis[i]['bodies']['candidate'][10,0] += x_offset_ll1
            results_vis[i]['bodies']['candidate'][10,1] += y_offset_ll1
            results_vis[i]['bodies']['candidate'][19,0] += x_offset_ll1
            results_vis[i]['bodies']['candidate'][19,1] += y_offset_ll1



            x_offset_ll2 = (results_vis[i]['bodies']['candidate'][10][0]-results_vis[i]['bodies']['candidate'][9][0])*(ll2_ratio-1.)
            y_offset_ll2 = (results_vis[i]['bodies']['candidate'][10][1]-results_vis[i]['bodies']['candidate'][9][1])*(ll2_ratio-1.)

            results_vis[i]['bodies']['candidate'][10,0] += x_offset_ll2
            results_vis[i]['bodies']['candidate'][10,1] += y_offset_ll2
            results_vis[i]['bodies']['candidate'][19,0] += x_offset_ll2
            results_vis[i]['bodies']['candidate'][19,1] += y_offset_ll2

            ########right leg########

            x_offset_rl1 = (results_vis[i]['bodies']['candidate'][12][0]-results_vis[i]['bodies']['candidate'][11][0])*(rl1_ratio-1.)
            y_offset_rl1 = (results_vis[i]['bodies']['candidate'][12][1]-results_vis[i]['bodies']['candidate'][11][1])*(rl1_ratio-1.)

            results_vis[i]['bodies']['candidate'][12,0] += x_offset_rl1
            results_vis[i]['bodies']['candidate'][12,1] += y_offset_rl1
            results_vis[i]['bodies']['candidate'][13,0] += x_offset_rl1
            results_vis[i]['bodies']['candidate'][13,1] += y_offset_rl1
            results_vis[i]['bodies']['candidate'][18,0] += x_offset_rl1
            results_vis[i]['bodies']['candidate'][18,1] += y_offset_rl1


            x_offset_rl2 = (results_vis[i]['bodies']['candidate'][13][0]-results_vis[i]['bodies']['candidate'][12][0])*(rl2_ratio-1.)
            y_offset_rl2 = (results_vis[i]['bodies']['candidate'][13][1]-results_vis[i]['bodies']['candidate'][12][1])*(rl2_ratio-1.)

            results_vis[i]['bodies']['candidate'][13,0] += x_offset_rl2
            results_vis[i]['bodies']['candidate'][13,1] += y_offset_rl2
            results_vis[i]['bodies']['candidate'][18,0] += x_offset_rl2
            results_vis[i]['bodies']['candidate'][18,1] += y_offset_rl2

            results_vis[i]['bodies']['candidate'] += offset[np.newaxis, :]
            results_vis[i]['faces'] += offset[np.newaxis, np.newaxis, :]
            results_vis[i]['hands'] += offset[np.newaxis, np.newaxis, :]
    
    dwpose_woface_list = []
    for i in range(len(results_vis)):
        #try:
        dwpose_woface, dwpose_wface = draw_pose(results_vis[i], H=height, W=width, stick_width=stick_width,
                                                    draw_body=draw_body, draw_hands=draw_hands, hand_keypoint_size=hand_keypoint_size,
                                                    draw_feet=draw_feet, body_keypoint_size=body_keypoint_size, draw_head=draw_head)
        result = torch.from_numpy(dwpose_woface)
        #except:
        #    result = torch.zeros((height, width, 3), dtype=torch.uint8)
        dwpose_woface_list.append(result)
    dwpose_woface_tensor = torch.stack(dwpose_woface_list, dim=0)

    dwpose_woface_ref_tensor = None
    if ref_image is not None:
        dwpose_woface_ref, dwpose_wface_ref = draw_pose(pose_ref, H=height, W=width, stick_width=stick_width,
                                                        draw_body=draw_body, draw_hands=draw_hands, hand_keypoint_size=hand_keypoint_size,
                                                        draw_feet=draw_feet, body_keypoint_size=body_keypoint_size, draw_head=draw_head)
        dwpose_woface_ref_tensor = torch.from_numpy(dwpose_woface_ref)

    return dwpose_woface_tensor, dwpose_woface_ref_tensor

class WanVideoUniAnimateDWPoseDetector:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "pose_images": ("IMAGE", {"tooltip": "Pose images"}),
                "score_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Score threshold for pose detection"}),
                "stick_width": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1, "tooltip": "Stick width for drawing keypoints"}),
                "draw_body": ("BOOLEAN", {"default": True, "tooltip": "Draw body keypoints"}),
                "body_keypoint_size": ("INT", {"default": 4, "min": 0, "max": 100, "step": 1, "tooltip": "Body keypoint size"}),
                "draw_feet": ("BOOLEAN", {"default": True, "tooltip": "Draw feet keypoints"}),
                "draw_hands": ("BOOLEAN", {"default": True, "tooltip": "Draw hand keypoints"}),
                "hand_keypoint_size": ("INT", {"default": 4, "min": 0, "max": 100, "step": 1, "tooltip": "Hand keypoint size"}),
                "colorspace": (["RGB", "BGR"], {"tooltip": "Color space for the output image"}),
                "handle_not_detected": (["empty", "repeat"], {"default": "empty", "tooltip": "How to handle undetected poses, empty inserts black and repeat inserts previous detection"}),
                "draw_head": ("BOOLEAN", {"default": True, "tooltip": "Draw head keypoints"}),
            },
            "optional": {
                "reference_pose_image": ("IMAGE", {"tooltip": "Reference pose image"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", )
    RETURN_NAMES = ("poses", "reference_pose",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, pose_images, score_threshold, stick_width, reference_pose_image=None, draw_body=True, body_keypoint_size=4, 
                draw_feet=True, draw_hands=True, hand_keypoint_size=4, colorspace="RGB", handle_not_detected="empty", draw_head=True):

        device = mm.get_torch_device()
        
        #model loading
        dw_pose_model = "dw-ll_ucoco_384_bs5.torchscript.pt"
        yolo_model = "yolox_l.torchscript.pt"

        script_directory = os.path.dirname(os.path.abspath(__file__))
        model_base_path = os.path.join(script_directory, "models", "DWPose")

        model_det=os.path.join(model_base_path, yolo_model)
        model_pose=os.path.join(model_base_path, dw_pose_model)

        if not os.path.exists(model_det):
            log.info(f"Downloading yolo model to: {model_base_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="hr16/yolox-onnx", 
                                allow_patterns=[f"*{yolo_model}*"],
                                local_dir=model_base_path, 
                                local_dir_use_symlinks=False)
            
        if not os.path.exists(model_pose):
            log.info(f"Downloading dwpose model to: {model_base_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="hr16/DWPose-TorchScript-BatchSize5", 
                                allow_patterns=[f"*{dw_pose_model}*"],
                                local_dir=model_base_path, 
                                local_dir_use_symlinks=False)

        if not hasattr(self, "det") or not hasattr(self, "pose"):
            self.det = torch.jit.load(model_det, map_location=device)
            self.pose = torch.jit.load(model_pose, map_location=device)
            self.dwpose_detector = DWposeDetector(self.det, self.pose) 

        #model inference
        height, width = pose_images.shape[1:3]
        
        pose_np = pose_images.cpu().numpy() * 255
        ref_np = None
        if reference_pose_image is not None:
            ref = reference_pose_image
            ref_np = ref.cpu().numpy() * 255

        poses, reference_pose = pose_extract(pose_np, ref_np, self.dwpose_detector, height, width, score_threshold, stick_width=stick_width,
                                             draw_body=draw_body, body_keypoint_size=body_keypoint_size, draw_feet=draw_feet, 
                                             draw_hands=draw_hands, hand_keypoint_size=hand_keypoint_size, handle_not_detected=handle_not_detected, draw_head=draw_head)
        poses = poses / 255.0
        if reference_pose_image is not None:
            reference_pose = reference_pose.unsqueeze(0) / 255.0
        else:
            reference_pose = torch.zeros(1, 64, 64, 3, device=torch.device("cpu"))

        if colorspace == "BGR":
            poses=torch.flip(poses, dims=[-1])

        return (poses, reference_pose, )

class WanVideoUniAnimatePoseInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pose_images": ("IMAGE", {"tooltip": "Pose images"}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Strength of the pose control"}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percentage for the pose control"}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percentage for the pose control"}),
            },
            "optional": {
                "reference_pose_image": ("IMAGE", {"tooltip": "Reference pose image"}),
            },
        }

    RETURN_TYPES = ("UNIANIMATE_POSE", )
    RETURN_NAMES = ("unianimate_poses",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, pose_images, strength, start_percent, end_percent, reference_pose_image=None):

        pose = pose_images.permute(3, 0, 1, 2).unsqueeze(0).contiguous()

        ref = None
        if reference_pose_image is not None:
            ref = reference_pose_image.permute(0, 3, 1, 2).contiguous()

        unianim_poses = {
            "pose": pose,
            "ref": ref,
            "strength": strength,
            "start_percent": start_percent,
            "end_percent": end_percent
        }

        return (unianim_poses,)


NODE_CLASS_MAPPINGS = {
    "WanVideoUniAnimatePoseInput": WanVideoUniAnimatePoseInput,
    "WanVideoUniAnimateDWPoseDetector": WanVideoUniAnimateDWPoseDetector,
 
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoUniAnimatePoseInput": "WanVideo UniAnimate Pose Input",
    "WanVideoUniAnimateDWPoseDetector": "WanVideo UniAnimate DWPose Detector",
    }

    