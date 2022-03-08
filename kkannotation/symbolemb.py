import copy
from typing import List, Union
from functools import partial
import numpy as np
import cv2
from PIL import ImageDraw, ImageFont

# local package
from kkannotation.util.image import cv2pil, pil2cv
from kkannotation.util.com import check_type_list


__all__ = [
    "SymbolEmbedding"
]


class SymbolEmbedding:
    def __init__(self, height: int=None, width: int=None, canvas: dict=None, procs: List[dict]=None):
        assert isinstance(height, int)
        assert isinstance(width,  int)
        assert isinstance(canvas, dict) and "type" in canvas and canvas["type"] in ["noise", "background"]
        if canvas["type"] == "noise":
            assert "range_noise" in canvas
        else:
            pass
        assert check_type_list(procs, dict)
        for x in procs:
            assert "type" in x and x["type"] in ["text"]
        self.height = height
        self.width  = width
        canvas = copy.deepcopy(canvas)
        procs  = copy.deepcopy(procs)
        del canvas["type"]
        self.canvas = partial(self.create_canvas, height=self.height, width=self.width, **canvas)
        _procs = []
        for x in procs:
            if x["type"] == "text":
                del x["type"]
                _procs.append(partial(self.draw_text, height=self.height, width=self.width, **x))
        self.procs = _procs

    def create_image(self):
        img = self.canvas()
        adds = []
        for proc in self.procs:
            adds += proc()
        adds = np.stack(adds)
        adds = adds[np.random.permutation(np.arange(adds.shape[0]))]
        mask = (adds.sum(axis=-1) > 0)
        for i, add in enumerate(adds):
            img[mask[i]] = add[mask[i]]
        return img

    @classmethod
    def create_canvas(cls, height: int=None, width: int=None, range_noise: Union[List[int], List[List[int]]]=[255, 256]):
        assert isinstance(height, int)
        assert isinstance(width,  int)
        __class__.check_color_range(range_noise)
        img = np.zeros((height, width, 3)).astype(np.uint8)
        if len(range_noise) == 2:
            noise = np.random.randint(range_noise[0], range_noise[1]+1, (height, width)).astype(np.uint8)
            noise = noise.reshape(*noise.shape, 1).repeat(3, axis=-1)
            img   = noise
        else:
            list_noise = []
            for x in range_noise:
                list_noise.append(np.random.randint(x[0], x[1]+1, (height, width)).astype(np.uint8).reshape(height, width, 1))
            img = np.concatenate(list_noise, axis=-1)
        return img
    
    @classmethod
    def rotate_affine(cls, img: np.ndarray, angle: int, is_adjust_border: bool=False, center=None, color=[0, 0, 0]):
        assert isinstance(img,   np.ndarray)
        assert isinstance(angle, int)
        assert isinstance(is_adjust_border, bool)
        assert center is None or (check_type_list(center, int) and len(center) == 2)
        assert check_type_list(color, int) and len(color) == 3
        h, w, _   = img.shape
        angle_rad = angle / 180.0 * np.pi
        if is_adjust_border:
            w_rot    = int(np.round(h*np.absolute(np.sin(angle_rad))+w*np.absolute(np.cos(angle_rad))))
            h_rot    = int(np.round(h*np.absolute(np.cos(angle_rad))+w*np.absolute(np.sin(angle_rad))))
            size_rot = (w_rot, h_rot)
        else:
            size_rot = (w, h)
        if center is None:
            center = (int(w/2), int(h/2))
        scale  = 1
        affine_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        if is_adjust_border:
            affine_matrix[0][2] = affine_matrix[0][2] -w/2 + w_rot/2
            affine_matrix[1][2] = affine_matrix[1][2] -h/2 + h_rot/2
        img = cv2.warpAffine(img, affine_matrix, size_rot, flags=cv2.INTER_LINEAR, borderValue=tuple(color))
        return img

    @classmethod
    def draw_text(
        cls, height: int=None, width: int=None, chars: Union[str, List[str]]=None, iters: int=None, range_scale: List[int]=[0.5, 2.0], 
        range_thickness: List[int]=[1, 3], range_color: Union[List[int], List[List[int]]]=[255, 255], 
        range_rotation: List[int]=[-20, 20], is_PIL: bool=False, font_pil: str=None
    ):
        assert isinstance(height, int)
        assert isinstance(width,  int)
        if isinstance(chars, str): chars = list(chars)
        assert check_type_list(chars, str)
        assert check_type_list(range_scale, [int, float]) and len(range_scale) == 2 and range_scale[0] < range_scale[1]
        if is_PIL: assert isinstance(font_pil, str)
        else: assert check_type_list(range_thickness, int) and len(range_thickness) == 2 and range_thickness[0] < range_thickness[1]
        __class__.check_color_range(range_color)
        img              = img = np.zeros((height, width, 3)).astype(np.uint8)
        chars            = np.array(chars).copy()
        chars            = chars[np.random.randint(0, chars.shape[0], iters)]
        range_scale      = np.arange(range_scale[0], range_scale[1], (range_scale[1] - range_scale[0])/100.0)
        range_scale      = range_scale[np.random.randint(0, range_scale.shape[0], iters)]
        range_thickness  = np.arange(range_thickness[0], range_thickness[1], 1)
        range_thickness  = range_thickness[np.random.randint(0, range_thickness.shape[0], iters)]
        range_rotation   = np.arange(range_rotation[0], range_rotation[1]+1, 1)
        range_rotation   = range_rotation[np.random.randint(0, range_rotation.shape[0], iters)]
        loc_x            = np.random.randint(0, width,  iters).tolist()
        loc_y            = np.random.randint(0, height, iters).tolist()
        if len(range_color) == 2:
            range_color  = np.arange(range_color[0], range_color[1]+1, 1)
            range_color  = range_color[np.random.randint(0, range_color.shape[0], iters)]
            range_color  = range_color.repeat(3).reshape(-1, 3)
        else:
            list_range = []
            for x in range_color:
                rangewk  = np.arange(x[0], x[1]+1, 1)
                list_range.append(rangewk[np.random.randint(0, rangewk.shape[0], iters)])
            range_color = np.stack(list_range).T
        range_color = range_color.tolist()
        list_imgs   = []
        for i in np.arange(iters):
            loc = [loc_x[i], loc_y[i]]
            if is_PIL:
                tmp     = cv2pil(img.copy())
                draw    = ImageDraw.Draw(tmp)
                fontPIL = ImageFont.truetype(font=font_pil, size=int(range_scale[i]))
                draw.text(
                    xy=loc,
                    text=chars[i],
                    fill=tuple(range_color[i]),
                    font=fontPIL
                )
                tmp = pil2cv(tmp).copy()
            else:
                tmp = cv2.putText(
                    img.copy(),
                    text=chars[i],
                    org=loc,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=range_scale[i], 
                    color=tuple(range_color[i]), 
                    thickness=range_thickness[i],
                    lineType=cv2.LINE_AA,
                )
            tmp = __class__.rotate_affine(tmp, int(range_rotation[i]), is_adjust_border=False, center=loc)
            list_imgs.append(tmp)
        return list_imgs
    
    @classmethod
    def check_color_range(cls, range_color: Union[List[int], List[List[int]]]):
        if check_type_list(range_color, int):
            assert len(range_color) == 2
        elif check_type_list(range_color, list, int):
            assert len(range_color) == 3
            for x in range_color:
                assert len(x) == 2
        else:
            raise AttributeError("not match color attribute.")
        assert (np.array(range_color) >= 256).sum() == 0
        assert (np.array(range_color) <    0).sum() == 0

    def draw_shape(
        img: np.ndarray, n: int, thickness: int,
        shape_list: List[str]=["line", "tline", "circle", "rectangle"], color_range: (int, int)=(0, 255)
    ):
        for _ in range(n):
            shape = np.random.permutation(shape_list)[0]
            if shape in ["line", "tline"]:
                p1_x, p2_x = np.random.randint(img.shape[1]), np.random.randint(img.shape[1])
                p1_y, p2_y = np.random.randint(img.shape[0]), np.random.randint(img.shape[0])
                color  = np.random.randint(color_range[0], color_range[1]+1)
                tscale = np.random.randint(1, thickness)
                if shape in ["tline"]:
                    space = np.random.randint(img.shape[0]//100, img.shape[0]//20)
                    for i in range(3):
                        img = cv2.line(img, (p1_x, p1_y+(i*space)), (p2_x, p2_y+(i*space)), color, thickness=tscale, lineType=cv2.LINE_8, shift=0)
                else:
                    img = cv2.line(img, (p1_x, p1_y), (p2_x, p2_y), color, thickness=tscale, lineType=cv2.LINE_8, shift=0)
            elif shape in ["circle"]:
                p1_x, p1_y = np.random.randint(img.shape[1]), np.random.randint(img.shape[0])
                color  = np.random.randint(color_range[0], color_range[1]+1)
                tscale = np.random.randint(1, thickness)
                radius = np.random.randint(5, min(img.shape[0], img.shape[1]))
                img = cv2.circle(img, (p1_x, p1_y), radius, color, thickness=tscale)
            elif shape in ["rectangle"]:
                p1_x, p2_x = np.random.randint(img.shape[1]), np.random.randint(img.shape[1])
                p1_y, p2_y = np.random.randint(img.shape[0]), np.random.randint(img.shape[0])
                color  = np.random.randint(color_range[0], color_range[1]+1)
                tscale = np.random.randint(1, thickness)
                img = cv2.rectangle(img, (p1_x, p1_y), (p2_x, p2_y), color, thickness=tscale, lineType=cv2.LINE_8, shift=0)
        return img
    @classmethod
    def draw_shape(
        cls, height: int=None, width: int=None, iters: int=None, range_thickness: List[int]=[1, 3], 
        range_color: Union[List[int], List[List[int]]]=[255, 255], range_rotation: List[int]=[-20, 20],
    ):
        assert isinstance(height, int)
        assert isinstance(width,  int)
        assert check_type_list(range_thickness, int) and len(range_thickness) == 2 and range_thickness[0] < range_thickness[1]
        __class__.check_color_range(range_color)
        img              = img = np.zeros((height, width, 3)).astype(np.uint8)
        chars            = np.array(chars).copy()
        chars            = chars[np.random.randint(0, chars.shape[0], iters)]
        range_scale      = np.arange(range_scale[0], range_scale[1], (range_scale[1] - range_scale[0])/100.0)
        range_scale      = range_scale[np.random.randint(0, range_scale.shape[0], iters)]
        range_thickness  = np.arange(range_thickness[0], range_thickness[1], 1)
        range_thickness  = range_thickness[np.random.randint(0, range_thickness.shape[0], iters)]
        range_rotation   = np.arange(range_rotation[0], range_rotation[1]+1, 1)
        range_rotation   = range_rotation[np.random.randint(0, range_rotation.shape[0], iters)]
        loc_x            = np.random.randint(0, width,  iters).tolist()
        loc_y            = np.random.randint(0, height, iters).tolist()
        if len(range_color) == 2:
            range_color  = np.arange(range_color[0], range_color[1]+1, 1)
            range_color  = range_color[np.random.randint(0, range_color.shape[0], iters)]
            range_color  = range_color.repeat(3).reshape(-1, 3)
        else:
            list_range = []
            for x in range_color:
                rangewk  = np.arange(x[0], x[1]+1, 1)
                list_range.append(rangewk[np.random.randint(0, rangewk.shape[0], iters)])
            range_color = np.stack(list_range).T
        range_color = range_color.tolist()
        list_imgs   = []
        for i in np.arange(iters):
            loc = [loc_x[i], loc_y[i]]
            if is_PIL:
                tmp     = cv2pil(img.copy())
                draw    = ImageDraw.Draw(tmp)
                fontPIL = ImageFont.truetype(font=font_pil, size=int(range_scale[i]))
                draw.text(
                    xy=loc,
                    text=chars[i],
                    fill=tuple(range_color[i]),
                    font=fontPIL
                )
                tmp = pil2cv(tmp).copy()
            else:
                tmp = cv2.putText(
                    img.copy(),
                    text=chars[i],
                    org=loc,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=range_scale[i], 
                    color=tuple(range_color[i]), 
                    thickness=range_thickness[i],
                    lineType=cv2.LINE_AA,
                )
            tmp = __class__.rotate_affine(tmp, int(range_rotation[i]), is_adjust_border=False, center=loc)
            list_imgs.append(tmp)
        return list_imgs
