from typing import List
import numpy as np
import cv2
from PIL import Image

# local package
from kkannotation.util.com import check_type_list


__all__ = [
    "COLORS",
    "draw_annotation",
    "pil2cv",
    "cv2pil",
]


COLORS=[
    (255,   0,   0),
    (  0, 255,   0),
    (  0,   0, 255),
    (255, 255,   0),
    (  0, 255, 255),
    (255,   0, 255),
    (128,   0,   0),
    (  0, 128,   0),
    (  0,   0, 128),
    (128, 128,   0),
    (  0, 128, 128),
    (128,   0, 128),
    (  0,   0,   0),
    (128, 128, 128),
    (192, 192, 192),
    (255, 255, 255),
]

def draw_annotation(
    img: np.ndarray, bbox, catecory_name: str=None,
    segmentations: List[List[int]]=None,
    keypoints: List[int]=None, keypoints_name: List[str]=None, 
    keypoints_skeleton: List[List[str]]=None,
    color_id: int=None,
    color_bbox=(0,255,0),
    color_seg =(255,0,0),
    color_kpts=(0,0,255),
) -> np.ndarray:
    """
    Params::
        img: numpy ndarray.
        bbox: x_min, y_min, width, height
        segmentations:
            [[x11, y11, x12, y12, ...], [x21, y21, x22, y22, ...], ...]
        keypoints:
            [x1, y1, v1, x2, y2, v2, ...]
    """
    assert isinstance(img, np.ndarray)
    assert check_type_list(bbox, [int, float]) and sum([(x >= 0) for x in bbox]) == 4
    assert catecory_name is None or isinstance(catecory_name, str)
    if segmentations is not None:
        assert check_type_list(segmentations, list, [int, float])
        for seg in segmentations:
            assert len(seg) % 2 == 0
    else:
        segmentations = []
    assert keypoints is None or (check_type_list(keypoints, [int, float]) and len(keypoints) % 3 == 0)
    if keypoints_name is not None:
        assert check_type_list(keypoints_name, str)
        assert len(keypoints_name) * 3 == len(keypoints)
    if keypoints_skeleton is not None:
        assert keypoints_name is not None
        assert check_type_list(keypoints_skeleton, list, str)
        assert sum([len(x) == 2 for x in keypoints_skeleton]) == len(keypoints_skeleton)
    if isinstance(color_id, int):
        color_bbox = COLORS[ (color_id + 0) % len(COLORS) ]
        color_seg  = COLORS[ (color_id + 1) % len(COLORS) ]
        color_kpts = COLORS[ (color_id + 2) % len(COLORS) ]
    img = img.copy()
    # draw bbox
    x, y, w, h = bbox
    img = cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), color_bbox, 2)
    if catecory_name is not None: cv2.putText(img, catecory_name, (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color_bbox, thickness=2)
    # draw segmentation
    imgwk = np.zeros_like(img)
    for seg in segmentations:
        seg   = np.array(seg)
        img   = cv2.polylines(img, [seg.reshape(-1,1,2).astype(np.int32)], True, (0,0,0))
        imgwk = cv2.fillConvexPoly(imgwk, points=seg.reshape(-1, 2).astype(np.int32), color=color_seg)
    img = cv2.addWeighted(img, 1, imgwk, 0.8, 0)
    # draw keypoint
    if keypoints is not None:
        keypoints = np.array(keypoints).reshape(-1, 3).astype(int)
        if keypoints_name is not None:
            keypoints_name = np.array(keypoints_name)
        for j, (x, y, v, ) in enumerate(keypoints):
            color = (0, 0, 0,)
            if   v == 1: color = color_kpts
            elif v == 2: color = tuple((255 - np.array(color_kpts).astype(np.uint8)).astype(np.uint8).tolist())
            if v > 0:
                img = cv2.circle(img, (int(x), int(y)), 5, color, thickness=-1)
                if keypoints_name is not None:
                    cv2.putText(img, keypoints_name[j], (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1)
        if keypoints_skeleton is not None:
            for name_p1, name_p2 in keypoints_skeleton:
                index_p1 = np.where(keypoints_name == name_p1)[0][0]
                index_p2 = np.where(keypoints_name == name_p2)[0][0]
                img = cv2.line(img, tuple(keypoints[index_p1][:2]), tuple(keypoints[index_p2][:2]), color_kpts)
    return img

def pil2cv(img: Image) -> np.ndarray:
    new_image = np.array(img, dtype=np.uint8)
    if new_image.ndim == 2:  # gray
        pass
    elif new_image.shape[2] == 3:  # RGB
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # RGBA
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def cv2pil(img: np.ndarray):
    new_image = img.copy()
    if new_image.ndim == 2:  # gray
        pass
    elif new_image.shape[2] == 3:  # RGB
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # RGBA
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image
