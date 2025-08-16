import math
import numpy as np
import colorsys
import cv2


eps = 0.01


def smart_resize(x, s):
    Ht, Wt = s
    if x.ndim == 2:
        Ho, Wo = x.shape
        Co = 1
    else:
        Ho, Wo, Co = x.shape
    if Co == 3 or Co == 1:
        k = float(Ht + Wt) / float(Ho + Wo)
        return cv2.resize(x, (int(Wt), int(Ht)), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4)
    else:
        return np.stack([smart_resize(x[:, :, i], s) for i in range(Co)], axis=2)


def smart_resize_k(x, fx, fy):
    if x.ndim == 2:
        Ho, Wo = x.shape
        Co = 1
    else:
        Ho, Wo, Co = x.shape
    Ht, Wt = Ho * fy, Wo * fx
    if Co == 3 or Co == 1:
        k = float(Ht + Wt) / float(Ho + Wo)
        return cv2.resize(x, (int(Wt), int(Ht)), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4)
    else:
        return np.stack([smart_resize_k(x[:, :, i], fx, fy) for i in range(Co)], axis=2)


def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    return transfered_model_weights


def draw_bodypose(canvas, candidate, subset):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    return canvas

def alpha_blend_color(color, alpha):
    """blend color according to point conf
    """
    return [int(c * alpha) for c in color]

def draw_body_and_foot(canvas, candidate, subset, score, stick_width=4, draw_body=True, draw_feet=True, body_keypoint_size=4, draw_head=True):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)
    limbSeq_and_colors = []

    if draw_body:
        limbSeq_and_colors = [
            ([2, 3], [255, 0, 0]),   # Neck to Right Shoulder
            ([2, 6], [255, 85, 0]),  # Neck to Left Shoulder
            ([3, 4], [255, 170, 0]), # Right Shoulder to Right Elbow
            ([4, 5], [255, 255, 0]), # Right Elbow to Right Wrist
            ([6, 7], [170, 255, 0]), # Left Shoulder to Left Elbow
            ([7, 8], [85, 255, 0]),  # Left Elbow to Left Wrist
            ([2, 9], [0, 255, 0]),   # Neck to Right Hip
            ([9, 10], [0, 255, 85]), # Right Hip to Right Knee
            ([10, 11], [0, 255, 170]), # Right Knee to Right Ankle
            ([2, 12], [0, 255, 255]), # Neck to Left Hip
            ([12, 13], [0, 170, 255]), # Left Hip to Left Knee
            ([13, 14], [0, 85, 255]), # Left Knee to Left Ankle
        ]
    else:
        limbSeq_and_colors = [
            ([2, 3], [0, 0, 0]),   # Neck to Right Shoulder
            ([2, 6], [0, 0, 0]),  # Neck to Left Shoulder
            ([3, 4], [0, 0, 0]), # Right Shoulder to Right Elbow
            ([4, 5], [0, 0, 0]), # Right Elbow to Right Wrist
            ([6, 7], [0, 0, 0]), # Left Shoulder to Left Elbow
            ([7, 8], [0, 0, 0]),  # Left Elbow to Left Wrist
            ([2, 9], [0, 0, 0]),   # Neck to Right Hip
            ([9, 10], [0, 0, 0]), # Right Hip to Right Knee
            ([10, 11], [0, 0, 0]), # Right Knee to Right Ankle
            ([2, 12], [0, 0, 0]), # Neck to Left Hip
            ([12, 13], [0, 0, 0]), # Left Hip to Left Knee
            ([13, 14], [0, 0, 0]), # Left Knee to Left Ankle
        ]

    # Conditionally add head-related elements
    if draw_head:
        head_elements = [
            ([2, 1], [0, 0, 255]), # Neck to Nose
            ([1, 15], [0, 0, 255]), # Nose to Right Eye
            ([15, 17], [85, 0, 255]), # Right Eye to Right Ear
            ([1, 16], [170, 0, 255]), # Nose to Left Eye
            ([16, 18], [255, 0, 255]), # Left Eye to Left Ear
            ([3, 17], [255, 0, 170]), # Right Shoulder to Right Ear
            ([6, 18], [255, 0, 85])   # Left Shoulder to Left Ear
        ]
    else:
        head_elements = [
            ([2, 1], [0, 0, 0]),   # Neck to Nose
            ([1, 15], [0, 0, 0]), # Nose to Right Eye
            ([15, 17], [0, 0, 0]), # Right Eye to Right Ear
            ([1, 16], [0, 0, 0]), # Nose to Left Eye
            ([16, 18], [0, 0, 0]), # Left Eye to Left Ear
            ([3, 17], [0, 0, 0]), # Right Shoulder to Right Ear
            ([6, 18], [0, 0, 0])   # Left Shoulder to Left Ear
        ]
    if draw_feet:
        limbSeq_and_colors += [
            ([14, 19], [170, 255, 255]), # Left Ankle to Right Foot
            ([11, 20], [255, 255, 0]), # Right Ankle to Left Foot
        ]

    # Append head elements based on the condition
    limbSeq_and_colors += head_elements

    for limb_info in limbSeq_and_colors[:17]:
        limbSeq, color = limb_info
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq) - 1]
            conf = score[n][np.array(limbSeq) - 1]
            if conf[0] < 0.3 or conf[1] < 0.3:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = np.sqrt((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2)
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stick_width), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, alpha_blend_color(color, conf[0] * conf[1]))

    canvas = (canvas * 0.6).astype(np.uint8)

    for limb_info in limbSeq_and_colors[:18]:
        limbSeq, color = limb_info
        for i in limbSeq:
            for n in range(len(subset)):
                index = int(subset[n][i - 1])
                if index == -1:
                    continue
                x, y = candidate[index][0:2]
                conf = score[n][i - 1]
                if not np.isfinite(x) or not np.isfinite(y):
                    continue
                x = int(np.clip(x * W, 0, W - 1
                ))
                y = int(np.clip(y * H, 0, H - 1))
                # x = int(x * W)
                # y = int(y * H)
                cv2.circle(canvas, (x, y), 4, alpha_blend_color(color, conf), thickness=-1)

    return canvas


def draw_handpose(canvas, all_hand_peaks, draw_hands=True, hand_keypoint_size=4):
    H, W, C = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for peaks in all_hand_peaks:
        peaks = np.array(peaks)

        if draw_hands:
            for ie, e in enumerate(edges):
                x1, y1 = peaks[e[0]]
                x2, y2 = peaks[e[1]]
                x1 = int(x1 * W)
                y1 = int(y1 * H)
                x2 = int(x2 * W)
                y2 = int(y2 * H)
                if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                    h = (ie / float(len(edges))) % 1.0
                    s, v = 1.0, 1.0
                    r, g, b = colorsys.hsv_to_rgb(h, s, v)
                    color = (int(255 * r), int(255 * g), int(255 * b))
                    cv2.line(canvas, (x1, y1), (x2, y2), color, thickness=2)
        if hand_keypoint_size > 0:
            for i, keypoint in enumerate(peaks):
                x, y = keypoint
                x = int(x * W)
                y = int(y * H)
                if x > eps and y > eps:
                    cv2.circle(canvas, (x, y), hand_keypoint_size, (0, 0, 255), thickness=-1)
    return canvas


def draw_facepose(canvas, all_lmks):
    H, W, C = canvas.shape
    for lmks in all_lmks:
        lmks = np.array(lmks)
        for lmk in lmks:
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas


# detect hand according to body pose keypoints
# please refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/hand/handDetector.cpp
def handDetect(candidate, subset, oriImg):
    # right hand: wrist 4, elbow 3, shoulder 2
    # left hand: wrist 7, elbow 6, shoulder 5
    ratioWristElbow = 0.33
    detect_result = []
    image_height, image_width = oriImg.shape[0:2]
    for person in subset.astype(int):
        # if any of three not detected
        has_left = np.sum(person[[5, 6, 7]] == -1) == 0
        has_right = np.sum(person[[2, 3, 4]] == -1) == 0
        if not (has_left or has_right):
            continue
        hands = []
        #left hand
        if has_left:
            left_shoulder_index, left_elbow_index, left_wrist_index = person[[5, 6, 7]]
            x1, y1 = candidate[left_shoulder_index][:2]
            x2, y2 = candidate[left_elbow_index][:2]
            x3, y3 = candidate[left_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, True])
        # right hand
        if has_right:
            right_shoulder_index, right_elbow_index, right_wrist_index = person[[2, 3, 4]]
            x1, y1 = candidate[right_shoulder_index][:2]
            x2, y2 = candidate[right_elbow_index][:2]
            x3, y3 = candidate[right_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, False])

        for x1, y1, x2, y2, x3, y3, is_left in hands:
            
            x = x3 + ratioWristElbow * (x3 - x2)
            y = y3 + ratioWristElbow * (y3 - y2)
            distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            width = 1.5 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
            # x-y refers to the center --> offset to topLeft point
            # handRectangle.x -= handRectangle.width / 2.f;
            # handRectangle.y -= handRectangle.height / 2.f;
            x -= width / 2
            y -= width / 2  # width = height
            # overflow the image
            if x < 0: x = 0
            if y < 0: y = 0
            width1 = width
            width2 = width
            if x + width > image_width: width1 = image_width - x
            if y + width > image_height: width2 = image_height - y
            width = min(width1, width2)
            # the max hand box value is 20 pixels
            if width >= 20:
                detect_result.append([int(x), int(y), int(width), is_left])

    '''
    return value: [[x, y, w, True if left hand else False]].
    width=height since the network require squared input.
    x, y is the coordinate of top left 
    '''
    return detect_result


# Written by Lvmin
def faceDetect(candidate, subset, oriImg):
    # left right eye ear 14 15 16 17
    detect_result = []
    image_height, image_width = oriImg.shape[0:2]
    for person in subset.astype(int):
        has_head = person[0] > -1
        if not has_head:
            continue

        has_left_eye = person[14] > -1
        has_right_eye = person[15] > -1
        has_left_ear = person[16] > -1
        has_right_ear = person[17] > -1

        if not (has_left_eye or has_right_eye or has_left_ear or has_right_ear):
            continue

        head, left_eye, right_eye, left_ear, right_ear = person[[0, 14, 15, 16, 17]]

        width = 0.0
        x0, y0 = candidate[head][:2]

        if has_left_eye:
            x1, y1 = candidate[left_eye][:2]
            d = max(abs(x0 - x1), abs(y0 - y1))
            width = max(width, d * 3.0)

        if has_right_eye:
            x1, y1 = candidate[right_eye][:2]
            d = max(abs(x0 - x1), abs(y0 - y1))
            width = max(width, d * 3.0)

        if has_left_ear:
            x1, y1 = candidate[left_ear][:2]
            d = max(abs(x0 - x1), abs(y0 - y1))
            width = max(width, d * 1.5)

        if has_right_ear:
            x1, y1 = candidate[right_ear][:2]
            d = max(abs(x0 - x1), abs(y0 - y1))
            width = max(width, d * 1.5)

        x, y = x0, y0

        x -= width
        y -= width

        if x < 0:
            x = 0

        if y < 0:
            y = 0

        width1 = width * 2
        width2 = width * 2

        if x + width > image_width:
            width1 = image_width - x

        if y + width > image_height:
            width2 = image_height - y

        width = min(width1, width2)

        if width >= 20:
            detect_result.append([int(x), int(y), int(width)])

    return detect_result


# get max index of 2d array
def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j
