import math
from typing import List, Union
import numpy as np
import cv2
from PIL import Image

# local package
from kkannotation.util.com import check_type_list, convert_1d_array


__all__ = [
    "COLORS",
    "draw_annotation",
    "pil2cv",
    "cv2pil",
    "mask_from_bool_to_polygon",
    "fit_resize",
    "line_cross_frame",
    "estimate_matrix_affine",
    "affine_matrixes",
    "calc_iou",
    "nms_numpy",
    "template_matching",
    "create_figure_points",
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
    img: np.ndarray, bbox: List[Union[int, float]]=None, catecory_name: str=None,
    segmentations: List[List[int]]=None,
    keypoints: List[Union[int, float]]=None, keypoints_name: List[str]=None, 
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
    assert bbox is None or (check_type_list(bbox, [int, float]) and sum([(x >= 0) for x in bbox]) == 4)
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
    if bbox is not None:
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
                img = cv2.circle(img, (int(x), int(y)), 3, color, thickness=-1)
                if keypoints_name is not None:
                    cv2.putText(img, keypoints_name[j], (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1)
        if keypoints_skeleton is not None:
            for name_p1, name_p2 in keypoints_skeleton:
                index_p1 = np.where(keypoints_name == name_p1)[0][0]
                index_p2 = np.where(keypoints_name == name_p2)[0][0]
                img = cv2.line(img, tuple(keypoints[index_p1][:2]), tuple(keypoints[index_p2][:2]), color_kpts)
    return img

def line_cross_frame(slope: float, px: float, py: float, width: int, height: int):
    assert isinstance(slope, float)
    assert isinstance(px, float)
    assert isinstance(py, float)
    assert isinstance(width,  int)
    assert isinstance(height, int)
    b = py - px * slope
    list_p = []
    py_w0  = slope * 0     + b
    py_w1  = slope * width + b
    px_h0  = (0      - b) / slope
    px_h1  = (height - b) / slope
    if py_w0 >= 0 and py_w0 <= height: list_p.append([0,          int(py_w0)])
    if py_w1 >= 0 and py_w1 <= height: list_p.append([width,      int(py_w1)])
    if px_h0 >= 0 and px_h0 <= width:  list_p.append([int(px_h0), 0         ])
    if px_h1 >= 0 and px_h1 <= width:  list_p.append([int(px_h1), height    ])
    assert len(list_p) >= 2
    return list_p[:2]

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

def mask_from_bool_to_polygon(img: np.ndarray, ignore_n_point: int=6):
    list_polygons = []
    contours, _   = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for ndfwk in contours:
        listwk = ndfwk.reshape(-1).tolist()
        if len(listwk) < 2 * ignore_n_point: continue
        list_polygons.append(listwk)
    return list_polygons

def fit_resize(img: np.ndarray, dim: str, scale: int):
    """
    Params::
        img: image
        dim: x or y
        scale: width or height
    """
    if dim not in ["x","y"]: raise Exception(f"dim: {dim} is 'x' or 'y'.")
    height = img.shape[0]
    width  = img.shape[1]
    height_after, width_after = None, None
    if   type(scale) == int and scale > 10:
        if   dim == "x":
            width_after  = int(scale)
            height_after = int(height * (scale / width))
        elif dim == "y":
            height_after = int(scale)
            width_after  = int(width * (scale / height))
    else:
        raise Exception(f"scale > 10.")
    img = cv2.resize(img , (width_after, height_after)) # w, h
    return img

def estimate_matrix_affine(ndf_p1: np.ndarray, ndf_p2: np.ndarray):
    """
    https://natsutan.hatenablog.com/entry/20120928/1348831765
    """
    from scipy import linalg
    def matrix_Ai(p1: np.ndarray, p2: np.ndarray):
        assert isinstance(p1, np.ndarray)
        assert isinstance(p2, np.ndarray)
        assert len(p1.shape) == 1 and p1.shape[0] == 2 and p1.shape == p2.shape
        x1, y1 = p1
        x2, y2 = p2
        w      = 1
        return np.array([
            [ 0,  0, 0, x1, y1, w,  y2 * x1,  y2 * y1,  y2 * w],
            [x1, y1, w,  0,  0, 0, -x2 * x1, -x2 * y1, -x2 * w]
        ])
    assert isinstance(ndf_p1, np.ndarray)
    assert isinstance(ndf_p2, np.ndarray)
    assert len(ndf_p1.shape) == 2 and ndf_p1.shape[-1] == 2 and ndf_p1.shape == ndf_p2.shape
    matrix_A = np.concatenate([matrix_Ai(p1, p2) for p1, p2 in zip(ndf_p1, ndf_p2)], axis=0)
    U, s, Vh = linalg.svd(matrix_A)
    mat   = (Vh[-1] / Vh[-1][-1]).reshape(-1, 3)
    rad   = -math.atan2(mat[1,0],mat[0,0])  # radian
    scale = mat[0,0] / math.cos(rad) # scale
    m       = np.zeros([2, 2])
    m[0, 0] = 1 - mat[0, 0]
    m[0, 1] = -mat[0, 1]
    m[1, 0] = mat[0, 1]
    m[1, 1] = m[0, 0]
    mm = np.zeros([2, 1])
    mm[0, 0] = mat[0, 2]
    mm[1, 0] = mat[1, 2]
    center = np.dot(np.linalg.inv(m), mm)
    return mat, rad, scale, center

def affine_matrixes(
    range_scale: List[float]=None, range_shift: List[int]=None, range_degree: List[float]=None,
    step_scale: Union[float, List[float]]=1.0, step_shift: Union[int, List[int]]=1, step_degree: float=0.1
):
    """
    create affine matrixes. order is scale -> rotation -> shift.
    Usage::
        >>> mat, pat = affine_matrixes([0.9, 1.1, 0.9, 1.1], [0, 10, 0, 10], [-1.0, 1.0], step_scale=0.02, step_shift=2, step_degree=1.0)
    """
    if range_scale is None:
        range_scale = [1.0, 1.0, 1.0, 1.0]
        step_scale  = 1.0
    if range_shift is None:
        range_shift = [0, 0, 0, 0]
        step_shift  = 1
    if range_degree is None:
        range_degree = [0.0, 0.0]
        step_degree  = 1.0
    assert check_type_list(range_scale, float)  and (len(range_scale) == 2 or len(range_scale) == 4)
    assert check_type_list(range_shift, int)    and (len(range_shift) == 2 or len(range_shift) == 4)
    assert check_type_list(range_degree, float) and len(range_degree) == 2
    if isinstance(step_scale, float): step_scale = [step_scale, step_scale]
    assert check_type_list(step_scale, float)
    if isinstance(step_shift, int): step_shift = [step_shift, step_shift]
    assert check_type_list(step_shift, int)
    assert isinstance(step_degree, float)
    def matrix_scale(x, y):
        return np.array([
            [x, 0, 0],
            [0, y, 0],
            [0, 0, 1],
        ], dtype=np.float32)
    def matrix_shift(x, y):
        return np.array([
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1],
        ], dtype=np.float32)
    def matrix_rot(degree):
        return np.array([
            [np.cos(np.pi * degree / 180.), -np.sin(np.pi * degree / 180.), 0],
            [np.sin(np.pi * degree / 180.),  np.cos(np.pi * degree / 180.), 0],
            [                            0,                              0, 1],
        ], dtype=np.float32)
    def work(ndf1, ndf2):
        assert ndf1.shape[1:] == ndf2.shape[1:] and len(ndf1.shape) in [1, 3]
        if len(ndf1.shape) == 3:
            shape1 = [1] * int(len(ndf1.shape)-2)
            ndf1a  = np.tile(ndf1, (ndf2.shape[0],             1, *shape1))
            ndf2a  = np.tile(ndf2, (            1, ndf1.shape[0], *shape1)).reshape(-1, *ndf2.shape[1:])
            return np.einsum("abc,acd->abd", ndf1a, ndf2a)
        else:
            ndf1a = np.tile(  ndf1, ndf2.shape[0])
            ndf2a = np.repeat(ndf2, ndf1.shape[0])
            return [(x, y) for x, y in zip(ndf1a, ndf2a)]
    # scale
    if len(range_scale) == 2:
        pat_scale = [(x, x) for x in np.arange(range_scale[0], range_scale[1] + step_scale[0]/2, step_scale[0])]
        ndf_scale = np.stack([matrix_scale(x, y) for x, y in pat_scale])
    else:
        pat_scale_x = np.arange(range_scale[0], range_scale[1] + step_scale[0]/2, step_scale[0])
        pat_scale_y = np.arange(range_scale[2], range_scale[3] + step_scale[1]/2, step_scale[1])
        ndf_scale_x = np.stack([matrix_scale(  x, 1.0) for x in pat_scale_x])
        ndf_scale_y = np.stack([matrix_scale(1.0,   y) for y in pat_scale_y])
        pat_scale   = work(pat_scale_x, pat_scale_y)
        ndf_scale   = work(ndf_scale_x, ndf_scale_y)
    # shift
    if len(range_shift) == 2:
        range_shift = [range_shift[0], range_shift[1], range_shift[0], range_shift[1]]
    pat_shift_x = np.arange(range_shift[0], range_shift[1] + 0.5, step_shift[0], dtype=int)
    pat_shift_y = np.arange(range_shift[2], range_shift[3] + 0.5, step_shift[1], dtype=int)
    ndf_shift_x = np.stack([matrix_shift(x, 0) for x in pat_shift_x])
    ndf_shift_y = np.stack([matrix_shift(0, y) for y in pat_shift_y])
    pat_shift   = work(pat_shift_x, pat_shift_y)
    ndf_shift   = work(ndf_shift_x, ndf_shift_y)
    # rotation
    pat_rot = np.arange(range_degree[0], range_degree[1] + step_degree/2, step_degree)
    ndf_rot = np.stack([matrix_rot(x) for x in pat_rot])
    # merge
    matrix  = work(work(ndf_shift, ndf_rot), ndf_scale)
    pattern = work(
        np.array(work(
            np.array(pat_shift + ["a"], dtype=object)[:-1], 
            pat_rot
        ) + ["a"], dtype=object)[:-1],
        np.array(pat_scale + ["a"], dtype=object)[:-1]
    )
    pattern = np.array(convert_1d_array(pattern)).reshape(-1, 5)
    return matrix, pattern

def calc_iou(bbox_target: np.ndarray, bboxes_other: np.ndarray, bbox_target_area: np.ndarray=None, bboxes_other_area: np.ndarray=None):
    if bbox_target_area is None:
        bbox_target_area = (bbox_target[2] - bbox_target[0] + 1) * (bbox_target[3] - bbox_target[1] + 1)
    if bboxes_other_area is None:
        bboxes_other_area = (bboxes_other[:,2] - bboxes_other[:,0] + 1) * (bboxes_other[:,3] - bboxes_other[:,1] + 1)
    abx_min = np.maximum(bbox_target[0], bboxes_other[:,0]) # xmin
    aby_min = np.maximum(bbox_target[1], bboxes_other[:,1]) # ymin
    abx_max = np.minimum(bbox_target[2], bboxes_other[:,2]) # xmax
    aby_max = np.minimum(bbox_target[3], bboxes_other[:,3]) # ymax
    w       = np.maximum(0, abx_max - abx_min + 1)
    h       = np.maximum(0, aby_max - aby_min + 1)
    intersect = w*h
    iou       = intersect / (bbox_target_area + bboxes_other_area - intersect)
    return iou

def nms_numpy(bboxes: np.ndarray, scores: np.ndarray, iou_threshold: float=0.5):
    """
    Usage::
        >>> bboxes = np.concatenate([np.random.randint(0, 100, (1000, 2)), np.random.randint(100, 200, (1000, 2))], axis=-1)
        >>> scores = np.random.rand(1000)
        >>> nms_numpy(bboxes, scores, iou_threshold=0.5)
        array([918, 577, 792, 895, 677,   5,  26, 431, 658, 352, 597, 765, 573,
                345, 913, 668,  58, 449,  52, 610, 153, 747, 785, 644, 188, 567,
                673, 885, 215, 925, 966, 528, 308, 967, 687, 761,   0])
    """
    assert isinstance(bboxes, np.ndarray) and len(bboxes.shape) == 2 and bboxes.shape[-1] == 4
    assert isinstance(scores, np.ndarray) and len(scores.shape) == 1
    assert bboxes.shape[0] == scores.shape[0]
    assert isinstance(iou_threshold, float) and iou_threshold > 0 and iou_threshold < 1
    areas = (bboxes[:,2] - bboxes[:,0] + 1) * (bboxes[:,3] - bboxes[:,1] + 1)
    sort_index = np.argsort(scores)[::-1]
    i = 0
    while(len(sort_index) > (i + 1)):
        max_scr_ind = sort_index[i     ]
        ind_list    = sort_index[i + 1:]
        iou         = calc_iou(bboxes[max_scr_ind], bboxes[ind_list], bbox_target_area=areas[max_scr_ind], bboxes_other_area=areas[ind_list])
        del_index   = ind_list[(iou >= iou_threshold)]
        sort_index  = sort_index[~np.isin(sort_index, del_index)]
        i += 1
    return sort_index

def template_matching(
    img_binary: np.ndarray, temp_binary: np.ndarray, matrix: np.ndarray,
    score_threshold: float=0.95, iou_threshold: float=0.1, show: bool=False
):
    assert isinstance(img_binary,  np.ndarray) and img_binary.dtype  in [bool, np.bool_] and len(img_binary. shape) == 2
    assert isinstance(temp_binary, np.ndarray) and temp_binary.dtype in [bool, np.bool_] and len(temp_binary.shape) == 2
    assert isinstance(matrix, np.ndarray) and len(matrix.shape) == 3 and matrix.shape[1:] == (3, 3)
    height, width = img_binary.shape
    if show:
        cv2.imshow(__name__, (img_binary  * 255).astype(np.uint8))
        cv2.waitKey(0)
        cv2.imshow(__name__, (temp_binary * 255).astype(np.uint8))
        cv2.waitKey(0)
    imgwk  = img_binary.copy().reshape(1, *img_binary.shape)
    points = np.concatenate([
        np.stack(np.where(temp_binary)[::-1]),
        np.ones(temp_binary.sum()).reshape(1, -1)
    ], axis=0).astype(np.float32)
    ndf_conv = np.einsum("abc,cd->abd", matrix[:, :2, :], points)
    ndf_conv = ndf_conv.astype(np.int32)
    ndf_conv[:, 0, :][ndf_conv[:, 0, :] < 0] = 0
    ndf_conv[:, 1, :][ndf_conv[:, 1, :] < 0] = 0
    ndf_conv[:, 0, :][ndf_conv[:, 0, :] >= width ] = width - 1
    ndf_conv[:, 1, :][ndf_conv[:, 1, :] >= height] = height - 1
    ndf_mask = np.zeros((ndf_conv.shape[0], height, width), dtype=bool)
    ndf_mask[
        np.repeat(np.arange(ndf_conv.shape[0], dtype=np.int64), ndf_conv.shape[-1]),
        ndf_conv[:, 1, :].reshape(-1), ndf_conv[:, 0, :].reshape(-1)
    ] = True
    ndf_mask[:,          0,         0] = False # (0, 0) is special pixel.
    ndf_mask[:, height - 1, width - 1] = False # (w, h) is special pixel.
    imgwk = imgwk * ndf_mask
    score       = imgwk.   sum(axis=-1).sum(axis=-1)
    score_base  = ndf_mask.sum(axis=-1).sum(axis=-1)
    score_ratio = score / score_base
    bboxes      = np.concatenate([ndf_conv.min(axis=-1), ndf_conv.max(axis=-1)], axis=-1)
    boolwk      = (score_ratio > score_threshold)
    indexes_nms = nms_numpy(bboxes[boolwk], score_ratio[boolwk], iou_threshold=iou_threshold)
    indexes_nms = np.arange(0, score_ratio.shape[0], dtype=int)[boolwk][indexes_nms]
    if show:
        ret_mask = ndf_mask[indexes_nms]
        img      = (img_binary.copy() * 255).astype(np.uint8)
        img      = np.concatenate([img.reshape(*img.shape, 1) for _ in range(3)], axis=-1)
        for mask in ret_mask:
            draw = img.copy()
            draw[mask] = [255, 0, 0]
            cv2.imshow(__name__, draw)
            cv2.waitKey(0)
    return indexes_nms, ndf_mask, score, score_base, bboxes

def create_figure_points(height: int, width: int, points: List[Union[List[int], List[float]]], thickness=1, show: bool=False):
    assert isinstance(height, int) and height > 0
    assert isinstance(width,  int) and width  > 0
    assert check_type_list(points, list, int) or check_type_list(points, list, float)
    img = np.zeros((height, width, 3), np.uint8)
    ndf = np.array(points).reshape(-1, 4)
    if isinstance(points[0][0], float):
        ndf[:, 0] = ndf[:, 0] * (width  - 1)
        ndf[:, 1] = ndf[:, 1] * (height - 1)
        ndf[:, 2] = ndf[:, 2] * (width  - 1)
        ndf[:, 3] = ndf[:, 3] * (height - 1)
    for x1, y1, x2, y2 in ndf:
        img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), thickness=thickness, lineType=cv2.LINE_4)
    if show:
        cv2.imshow(__name__, img)
        cv2.waitKey(0)
    img = np.min(img, axis=-1)
    return img
