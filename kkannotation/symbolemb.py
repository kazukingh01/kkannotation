import copy, glob
from typing import List, Union
from functools import partial
import numpy as np
import cv2
from PIL import ImageDraw, ImageFont

# local package
import kkannotation
from kkannotation.coco import CocoManager
from kkannotation.util.image import cv2pil, pil2cv, mask_from_bool_to_polygon
from kkannotation.util.com import check_type_list


__all__ = [
    "SymbolEmbedding"
]


class SymbolEmbedding:
    def __init__(self, height: int=None, width: int=None, color_init: List[int]=[255, 255, 255], canvas: dict=None, labels: dict=None, procs: List[dict]=None):
        assert check_type_list(color_init, int) and len(color_init) == 3
        assert isinstance(canvas, dict) and "type" in canvas and canvas["type"] in ["noise", "background"]
        if canvas["type"] == "noise":
            for x in ["height", "width", "range_noise"]: assert x in canvas
        else:
            assert "path" in canvas
        assert labels is None or check_type_list(labels, dict)
        assert check_type_list(procs, dict)
        for x in procs:
            assert "type" in x and x["type"] in ["text", "line", "circle", "ellipse", "rectangle", "label"]
        self.height      = height
        self.width       = width
        self.color_init  = color_init
        self.procs       = copy.deepcopy(procs)
        self.procs_draw  = None
        self.procs_label = None
        canvas = copy.deepcopy(canvas)
        if canvas["type"] == "noise":
            self.canvas = partial(self.create_canvas, height=canvas["height"], width=canvas["width"], range_noise=canvas["range_noise"])
        else:
            self.backgrounds = glob.glob(canvas["path"])
            self.canvas      = lambda: cv2.imread(self.backgrounds[np.random.randint(0, len(self.backgrounds))])
        self.labels = copy.deepcopy(labels)
        for dictwk in self.labels:
            dictwk["img"]  = cv2.imread(dictwk["path"])
            if "mask" in dictwk: dictwk["mask"] = cv2.imread(dictwk["mask"])
            else:                dictwk["mask"] = __class__.get_mask(dictwk["img"], thre=dictwk.get("thre"))
    
    def set_process(self, height, width):
        _procs, _procs_label = [], []
        for x in self.procs:
            _type = x["type"]
            del x["type"]
            if   _type == "text":      _procs.append(partial(self.draw_text,            height=height, width=width, color_init=self.color_init, **x))
            elif _type == "line":      _procs.append(partial(self.draw_shape_line,      height=height, width=width, color_init=self.color_init, **x))
            elif _type == "circle":    _procs.append(partial(self.draw_shape_circle,    height=height, width=width, color_init=self.color_init, **x))
            elif _type == "ellipse":   _procs.append(partial(self.draw_shape_ellipse,   height=height, width=width, color_init=self.color_init, **x))
            elif _type == "rectangle": _procs.append(partial(self.draw_shape_rectangle, height=height, width=width, color_init=self.color_init, **x))
            elif _type == "label":
                _procs_label.append(partial(
                    self.draw_label, height=height, width=width, color_init=self.color_init, 
                    imgs_label=[x["img"] for x in self.labels], imgs_mask=[x["mask"] for x in self.labels], **x
                ))
        self.procs_draw  = _procs
        self.procs_label = _procs_label

    def create_image(self, filepath: str, is_save: bool=False):
        img  = self.canvas()
        self.set_process(img.shape[0], img.shape[1])
        coco = CocoManager()
        adds = []
        for proc in self.procs_draw:
            adds += proc()
        adds = np.stack(adds)
        mask = []
        for i, x in enumerate(self.color_init):
            mask.append(adds[:, :, :, i] == x)
        mask = ~(np.stack(mask).astype(int).sum(axis=0) == 3)
        mask_duplication = np.zeros((img.shape[0], img.shape[1])).astype(bool)
        for proc in self.procs_label:
            label, label_mask, indexes = proc()
            for a, b, c in zip(label, label_mask, indexes):
                if mask_duplication[b].sum() > 0: continue
                mask_duplication[b] = True
                ndf_y, ndf_x = np.where(b)
                coco.add(
                    filepath, img.shape[0], img.shape[1], 
                    [int(ndf_x.min()), int(ndf_y.min()), int(ndf_x.max()-ndf_x.min()), int(ndf_y.max()-ndf_y.min())],
                    self.labels[c]["name"], segmentations=mask_from_bool_to_polygon(b.astype(np.uint8))
                )
                adds = np.concatenate([adds, a.reshape(1, *a.shape)], axis=0)
                mask = np.concatenate([mask, b.reshape(1, *b.shape)], axis=0)
        indexes = np.random.permutation(np.arange(adds.shape[0]))
        adds    = adds[indexes]
        mask    = mask[indexes]
        for i, add in enumerate(adds):
            img[mask[i]] = add[mask[i]]
        coco.concat_added()
        if is_save:
            cv2.imwrite(filepath, img)
            coco.save(filepath + ".json")
        return img, coco

    @classmethod
    def create_canvas(cls, height: int=None, width: int=None, range_noise: Union[List[int], List[List[int]]]=[255, 256]):
        assert isinstance(height, int)
        assert isinstance(width,  int)
        range_noise = __class__.convert_color_range(range_noise, iters=(height * width)).reshape(height, width, 3)
        return range_noise.astype(np.uint8)
    
    @classmethod
    def rotate_affine(cls, img: np.ndarray, angle: int, is_adjust_border: bool=False, center=None, color=[0, 0, 0]):
        assert isinstance(img,   np.ndarray)
        assert isinstance(angle, int)
        assert isinstance(is_adjust_border, bool)
        assert center is None or (check_type_list(center, int) and len(center) == 2)
        if isinstance(color, int): color = [color, color, color]
        assert check_type_list(color, int) and len(color) == 3
        if len(img.shape) == 2: h, w    = img.shape
        else:                   h, w, _ = img.shape
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

    @classmethod
    def convert_color_range(cls, range_color: Union[List[int], List[List[int]]], iters: int=None):
        assert isinstance(iters, int)
        __class__.check_color_range(range_color)
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
        return range_color
   
    @classmethod
    def get_mask(cls, img: np.ndarray, thre: List[int]=None):
        assert isinstance(img,  np.ndarray) and len(img.shape) == 3 and img.shape[-1] == 3
        if thre is not None:
            __class__.check_color_range(thre)
            mask = []
            if len(thre) == 2:
                thre = [thre, thre, thre]
            for i, (x, y) in enumerate(thre):
                mask.append(((img[:, :, i] >= x) & (img[:, :, i] <= y)).astype(int))
            mask = np.stack(mask)
            mask = (mask.sum(axis=0) == 3).astype(np.uint8) * 255
        else:
            mask = (np.ones(img.shape[:2]) * 255).astype(np.uint8)
        return mask
    
    @classmethod
    def create_zero_imgs(cls, height: int=None, width: int=None, iters: int=None, color_init: List[int]=[255, 255, 255]):
        img = np.zeros((iters, height, width, 3)).astype(np.uint8)
        for i, x in enumerate(color_init): img[:, :, :, i] = x
        return img.astype(np.uint8)

    @classmethod
    def draw_text(
        cls, height: int=None, width: int=None, iters: int=None, color_init=None, chars: Union[str, List[str]]=None, range_scale: List[int]=[0.5, 2.0], 
        range_thickness: List[int]=[1, 3], range_color: Union[List[int], List[List[int]]]=[255, 255], 
        range_rotation: List[int]=[-20, 20], is_PIL: bool=False, font_pil: str=None
    ):
        assert isinstance(iters, int) and iters >= 0
        if iters == 0: return []
        assert isinstance(height, int)
        assert isinstance(width,  int)
        if isinstance(chars, str): chars = list(chars)
        assert check_type_list(chars, str)
        assert check_type_list(range_scale, [int, float]) and len(range_scale)    == 2 and range_scale[0]    <= range_scale[1]
        assert check_type_list(range_rotation, int)       and len(range_rotation) == 2 and range_rotation[0] <= range_rotation[1]
        if is_PIL:
            if font_pil is None: font_pil=f"/{kkannotation.__path__[0]}/font/ipaexg.ttf"
            assert isinstance(font_pil, str)
        else: assert check_type_list(range_thickness, int) and len(range_thickness) == 2 and range_thickness[0] <= range_thickness[1]
        range_color      = __class__.convert_color_range(range_color, iters=iters).tolist()
        img              = __class__.create_zero_imgs(height=height, width=width, iters=iters, color_init=color_init)
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
        list_imgs        = []
        for i in np.arange(iters):
            loc = [loc_x[i], loc_y[i]]
            if is_PIL:
                tmp     = cv2pil(img[i])
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
                    img[i],
                    text=chars[i],
                    org=loc,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=range_scale[i], 
                    color=tuple(range_color[i]), 
                    thickness=range_thickness[i],
                    lineType=cv2.LINE_AA,
                )
            tmp = __class__.rotate_affine(tmp, int(range_rotation[i]), is_adjust_border=False, center=loc, color=color_init)
            list_imgs.append(tmp)
        return list_imgs
    
    @classmethod
    def draw_shape_line(
        cls, height: int=None, width: int=None, iters: int=None, color_init=None, range_thickness: List[int]=[1, 3], 
        range_color: Union[List[int], List[List[int]]]=[255, 255], n_lines: int=1
    ):
        assert isinstance(iters, int) and iters >= 0
        if iters == 0: return []
        assert isinstance(height, int)
        assert isinstance(width,  int)
        assert check_type_list(range_thickness, int) and len(range_thickness) == 2 and range_thickness[0] <= range_thickness[1]
        range_color      = __class__.convert_color_range(range_color, iters=iters).tolist()
        img              = __class__.create_zero_imgs(height=height, width=width, iters=iters, color_init=color_init)
        range_thickness  = np.arange(range_thickness[0], range_thickness[1], 1)
        range_thickness  = range_thickness[np.random.randint(0, range_thickness.shape[0], iters)]
        loc_x            = np.random.randint(0, width,  iters*2).reshape(-1, 2).tolist()
        loc_y            = np.random.randint(0, height, iters*2).reshape(-1, 2).tolist()
        mean             = (height + width) / 2
        range_space      = np.random.randint(mean//100, mean//20, iters)
        list_imgs        = []
        for i in np.arange(iters):
            p1_x, p1_y = loc_x[i][0], loc_y[i][0]
            p2_x, p2_y = loc_x[i][1], loc_y[i][1]
            tmp = img[i]
            for j in range(n_lines):
                tmp = cv2.line(tmp, (p1_x, p1_y+(j*range_space[i])), (p2_x, p2_y+(j*range_space[i])), tuple(range_color[i]), thickness=range_thickness[i], lineType=cv2.LINE_8, shift=0)
            list_imgs.append(tmp)
        return list_imgs

    @classmethod
    def draw_shape_circle(
        cls, height: int=None, width: int=None, iters: int=None, color_init=None, range_thickness: List[int]=[1, 3], 
        range_color: Union[List[int], List[List[int]]]=[255, 255], range_radius: List[int]=[1, 3], 
    ):
        assert isinstance(iters, int) and iters >= 0
        if iters == 0: return []
        assert isinstance(height, int)
        assert isinstance(width,  int)
        assert check_type_list(range_thickness, int) and len(range_thickness) == 2 and range_thickness[0] <= range_thickness[1]
        assert check_type_list(range_radius,    int) and len(range_radius)    == 2 and range_radius[0]    <= range_radius[1]
        range_color      = __class__.convert_color_range(range_color, iters=iters).tolist()
        img              = __class__.create_zero_imgs(height=height, width=width, iters=iters, color_init=color_init)
        range_thickness  = np.arange(range_thickness[0], range_thickness[1], 1)
        range_thickness  = range_thickness[np.random.randint(0, range_thickness.shape[0], iters)]
        range_radius     = np.arange(range_radius[0], range_radius[1], 1)
        range_radius     = range_radius[np.random.randint(0, range_radius.shape[0], iters)]
        loc_x            = np.random.randint(0, width,  iters).tolist()
        loc_y            = np.random.randint(0, height, iters).tolist()
        list_imgs        = []
        for i in np.arange(iters):
            p1_x, p1_y = loc_x[i], loc_y[i]
            tmp = cv2.circle(img[i], (p1_x, p1_y), range_radius[i], range_color[i], thickness=range_thickness[i])
            list_imgs.append(tmp)
        return list_imgs

    @classmethod
    def draw_shape_ellipse(
        cls, height: int=None, width: int=None, iters: int=None, color_init=None, range_thickness: List[int]=[1, 3], 
        range_color: Union[List[int], List[List[int]]]=[255, 255], range_rotation: List[int]=[-20, 20],
        range_scale: List[int]=[100, 200], 
    ):
        assert isinstance(iters, int) and iters >= 0
        if iters == 0: return []
        assert isinstance(height, int)
        assert isinstance(width,  int)
        assert check_type_list(range_thickness, int) and len(range_thickness) == 2 and range_thickness[0] <= range_thickness[1]
        assert check_type_list(range_rotation, int)  and len(range_rotation)  == 2 and range_rotation[0]  <= range_rotation[1]
        assert check_type_list(range_scale, int)     and len(range_scale)     == 2 and range_scale[0]     <= range_scale[1]
        range_color      = __class__.convert_color_range(range_color, iters=iters).tolist()
        img              = __class__.create_zero_imgs(height=height, width=width, iters=iters, color_init=color_init)
        range_thickness  = np.arange(range_thickness[0], range_thickness[1], 1)
        range_thickness  = range_thickness[np.random.randint(0, range_thickness.shape[0], iters)]
        loc_x            = np.random.randint(0, width,  iters).tolist()
        loc_y            = np.random.randint(0, height, iters).tolist()
        range_rotation   = np.arange(range_rotation[0], range_rotation[1]+1, 1)
        range_rotation   = range_rotation[np.random.randint(0, range_rotation.shape[0], iters)].tolist()
        range_scale      = np.arange(range_scale[0], range_scale[1]+1, 1)
        range_scale      = range_scale[np.random.randint(0, range_scale.shape[0], iters*2)].reshape(-1, 2).tolist()
        list_imgs        = []
        for i in np.arange(iters):
            p1_x, p1_y = loc_x[i], loc_y[i]
            tmp = cv2.ellipse(img[i], ((p1_x, p1_y), (range_scale[i][0], range_scale[i][1]), range_rotation[i]), range_color[i], thickness=range_thickness[i])
            list_imgs.append(tmp)
        return list_imgs

    @classmethod
    def draw_shape_rectangle(
        cls, height: int=None, width: int=None, iters: int=None, color_init=None, range_thickness: List[int]=[1, 3], 
        range_color: Union[List[int], List[List[int]]]=[255, 255], range_rotation: List[int]=[-20, 20],
        range_scale: List[int]=[100, 200], 
    ):
        assert isinstance(iters, int) and iters >= 0
        if iters == 0: return []
        assert isinstance(height, int)
        assert isinstance(width,  int)
        assert check_type_list(range_thickness, int) and len(range_thickness) == 2 and range_thickness[0] <= range_thickness[1]
        assert check_type_list(range_rotation, int)  and len(range_rotation)  == 2 and range_rotation[0]  <= range_rotation[1]
        assert check_type_list(range_scale, int)     and len(range_scale)     == 2 and range_scale[0]     <= range_scale[1]
        range_color      = __class__.convert_color_range(range_color, iters=iters).tolist()
        img              = __class__.create_zero_imgs(height=height, width=width, iters=iters, color_init=color_init)
        range_thickness  = np.arange(range_thickness[0], range_thickness[1], 1)
        range_thickness  = range_thickness[np.random.randint(0, range_thickness.shape[0], iters)]
        loc_x            = np.random.randint(0, width,  iters).tolist()
        loc_y            = np.random.randint(0, height, iters).tolist()
        range_rotation   = np.arange(range_rotation[0], range_rotation[1]+1, 1)
        range_rotation   = range_rotation[np.random.randint(0, range_rotation.shape[0], iters)]
        range_scale      = np.arange(range_scale[0], range_scale[1]+1, 1)
        range_scale      = range_scale[np.random.randint(0, range_scale.shape[0], iters*2)].reshape(-1, 2).tolist()
        list_imgs        = []
        ndf              = np.ones(iters).astype(int)
        ndf[np.random.randint(0, 2, iters).astype(bool)] = -1
        for i in np.arange(iters):
            p1_x, p1_y = loc_x[i], loc_y[i]
            p2_x, p2_y = p1_x + ndf[i] * range_scale[i][0], p1_y + ndf[i] * range_scale[i][1]
            tmp = cv2.rectangle(img[i], (p1_x, p1_y), (p2_x, p2_y), range_color[i], thickness=range_thickness[i], lineType=cv2.LINE_8, shift=0)
            tmp = __class__.rotate_affine(tmp, int(range_rotation[i]), is_adjust_border=False, center=[p1_x, p1_y], color=color_init)
            list_imgs.append(tmp)
        return list_imgs
    
    @classmethod
    def draw_label(
        cls, height: int=None, width: int=None, iters: int=None, color_init=None, 
        imgs_label: List[np.ndarray]=None, imgs_mask: List[np.ndarray]=None, 
        range_scale: List[float]=[0.5, 2], range_rotation: List[int]=[-20, 20], is_fix_scale_ratio: bool=True
    ):
        assert isinstance(iters, int) and iters > 0
        assert isinstance(height, int)
        assert isinstance(width,  int)
        assert check_type_list(imgs_label, np.ndarray)
        assert check_type_list(range_scale, [int, float]) and len(range_scale) == 2 and range_scale[0] <= range_scale[1]
        img         = __class__.create_zero_imgs(height=height, width=width, iters=iters, color_init=color_init)
        bool_mask   = (imgs_mask is not None)
        mask        = np.zeros((iters, height, width)).astype(np.uint8) if bool_mask else None
        indexes     = np.random.randint(0, len(imgs_label), iters)
        imgs_label  = np.array(imgs_label + [np.zeros(0)], dtype=object) # Prevent unification with np.ndarray
        imgs_label  = imgs_label[indexes]
        if bool_mask:
            imgs_mask = np.array(imgs_mask + [np.zeros(0)], dtype=object) # Prevent unification with np.ndarray
            imgs_mask = imgs_mask[indexes]
        range_scale = np.arange(range_scale[0], range_scale[1], (range_scale[1] - range_scale[0])/100.0)
        if is_fix_scale_ratio:
            range_scale = range_scale[np.random.randint(0, range_scale.shape[0], iters)]
            imgs_label  = np.vectorize(lambda x, y: cv2.resize(x, dsize=None, fx=y, fy=y), otypes=[object])(imgs_label, range_scale)
            if bool_mask:
                imgs_mask = np.vectorize(lambda x, y: cv2.resize(x, dsize=None, fx=y, fy=y), otypes=[object])(imgs_mask, range_scale)
        else:
            range_scale = range_scale[np.random.randint(0, range_scale.shape[0], iters*2)].reshape(2, -1)
            imgs_label  = np.vectorize(lambda x, y, z: cv2.resize(x, dsize=None, fx=y, fy=z), otypes=[object])(imgs_label, range_scale[0], range_scale[1])
            if bool_mask:
                imgs_mask = np.vectorize(lambda x, y, z: cv2.resize(x, dsize=None, fx=y, fy=z), otypes=[object])(imgs_mask, range_scale[0], range_scale[1])
        range_rotation = np.arange(range_rotation[0], range_rotation[1]+1, 1)
        range_rotation = range_rotation[np.random.randint(0, range_rotation.shape[0], iters)]
        imgs_label     = np.vectorize(
            lambda x, y: __class__.rotate_affine(x, int(y), is_adjust_border=True, center=None, color=color_init), otypes=[object]
        )(imgs_label, range_rotation)
        if bool_mask:
            imgs_mask = np.vectorize(
                lambda x, y: __class__.rotate_affine(x, int(y), is_adjust_border=True, center=None, color=0), otypes=[object]
            )(imgs_mask, range_rotation)
        loc_x = np.vectorize(lambda x: np.random.randint(0, width  - x.shape[1] - 1))(imgs_label)
        loc_y = np.vectorize(lambda x: np.random.randint(0, height - x.shape[0] - 1))(imgs_label)
        for i in np.arange(iters):
            p1_x, p1_y = loc_x[i], loc_y[i]
            img[i, p1_y:p1_y+imgs_label[i].shape[0], p1_x:p1_x+imgs_label[i].shape[1], :] = imgs_label[i]
            if bool_mask:
                mask[i, p1_y:p1_y+imgs_mask[i].shape[0], p1_x:p1_x+imgs_mask[i].shape[1]] = imgs_mask[i]
        if bool_mask: mask = (mask > 0)
        return img, mask, indexes


