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
            for x in ["height", "width", "path"]: assert x in canvas
        assert labels is None or check_type_list(labels, dict)
        assert check_type_list(procs, dict)
        for x in procs:
            assert "type" in x and x["type"] in ["text", "line", "circle", "ellipse", "rectangle", "label", "dest"]
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
            self.canvas      = partial(
                self.create_canvas_from_image, path=self.backgrounds, 
                height=canvas.get("height"), width=canvas.get("width")
            )
        self.labels      = []
        self.labels_name = []
        for dict_label in copy.deepcopy(labels):
            assert isinstance(dict_label.get("name"), str)
            paths = glob.glob(dict_label["path"])
            if len(paths) > 1: assert "mask" not in dict_label
            for path in paths:
                dictwk = copy.deepcopy(dict_label)
                dictwk["img"] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if "mask" in dictwk:
                    dictwk["mask"] = cv2.imread(dictwk["mask"])
                else:
                    dictwk["mask"] = __class__.get_mask(dictwk["img"], thre=dictwk.get("thre"))
                if dictwk["img"].shape[-1] == 4:
                    dictwk["img"] = dictwk["img"][:, :, :3]
                self.labels.append(dictwk)
                self.labels_name.append(dictwk["name"])
        self.labels      = np.array(self.labels, dtype=object)
        self.labels_name = np.array(self.labels_name, dtype=object)
        self.set_process(canvas["height"], canvas["width"])
    
    def copy(self):
        obj = copy.deepcopy(self)
        labels = []
        for x in self.labels:
            dictwk = copy.deepcopy(x)
            dictwk["img"]  = dictwk["img" ].copy()
            dictwk["mask"] = dictwk["mask"].copy()
            labels.append(x)
        obj.labels = np.array(labels, dtype=object)
        return obj
    
    def set_process(self, height, width):
        _procs, _procs_label, _procs_dest = [], [], []
        for x in self.procs:
            _type = x["type"]
            del x["type"]
            if   _type == "text":      _procs.append(partial(self.draw_text,            height=height, width=width, color_init=self.color_init, **x))
            elif _type == "line":      _procs.append(partial(self.draw_shape_line,      height=height, width=width, color_init=self.color_init, **x))
            elif _type == "circle":    _procs.append(partial(self.draw_shape_circle,    height=height, width=width, color_init=self.color_init, **x))
            elif _type == "ellipse":   _procs.append(partial(self.draw_shape_ellipse,   height=height, width=width, color_init=self.color_init, **x))
            elif _type == "rectangle": _procs.append(partial(self.draw_shape_rectangle, height=height, width=width, color_init=self.color_init, **x))
            elif _type in ["label", "dest"]:
                assert "group" in x and check_type_list(x["group"], str)
                ndf_bool = np.isin(self.labels_name, x["group"])
                del x["group"]
                if _type == "label":
                    _procs_label.append(partial(
                        self.draw_label, height=height, width=width, color_init=self.color_init, n_merge=None,
                        imgs_label=[x["img"] for x in self.labels[ndf_bool]], 
                        imgs_mask=[x["mask"] for x in self.labels[ndf_bool]],
                        imgs_name=self.labels_name[ndf_bool],
                        **x
                    ))
                elif _type == "dest":
                    _procs_dest.append(partial(
                        self.draw_label, height=height, width=width, color_init=self.color_init, 
                        imgs_label=[x["img"] for x in self.labels[ndf_bool]],
                        imgs_mask=[x["mask"] for x in self.labels[ndf_bool]],
                        imgs_name=self.labels_name[ndf_bool],
                        **x
                    ))
        self.procs_draw  = _procs
        self.procs_label = _procs_label
        self.procs_dest  = _procs_dest

    def create_image(self, filepath: str, is_save: bool=False):
        img  = self.canvas()
        coco = CocoManager()
        adds = []
        for proc in self.procs_draw:
            adds += proc()
        adds = np.stack(adds).astype(np.uint8)
        mask = self.get_mask_except_color(adds, self.color_init)
        for proc in self.procs_dest:
            dest, dest_mask, _ = proc()
            adds = np.concatenate([adds, dest],      axis=0).astype(np.uint8)
            mask = np.concatenate([mask, dest_mask], axis=0).astype(bool)
        mask_duplication = np.zeros((img.shape[0], img.shape[1])).astype(bool)
        for i_proc in np.random.permutation(np.arange(len(self.procs_label), dtype=int)):
            proc = self.procs_label[i_proc]
            labels_img, labels_mask, labels_name = proc()
            for label_img, label_mask, label_name in zip(labels_img, labels_mask, labels_name):
                if mask_duplication[label_mask].sum() > 0: continue
                mask_duplication[label_mask] = True
                ndf_y, ndf_x = np.where(label_mask)
                if len(ndf_y) > 0 and len(ndf_x) > 0:
                    bbox = [int(ndf_x.min()), int(ndf_y.min()), int(ndf_x.max()-ndf_x.min()), int(ndf_y.max()-ndf_y.min())]
                    coco.add(
                        filepath, img.shape[0], img.shape[1], bbox,
                        label_name, segmentations=mask_from_bool_to_polygon(label_mask.astype(np.uint8))
                    )
                    adds = np.concatenate([adds, label_img. reshape(1, *label_img. shape)], axis=0)
                    mask = np.concatenate([mask, label_mask.reshape(1, *label_mask.shape)], axis=0)
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
    def create_canvas_from_image(cls, path: str=None, height: int=None, width: int=None):
        if isinstance(path, list):
            path = path[np.random.randint(0, len(path))]
        assert isinstance(path, str)
        if height is not None:
            assert isinstance(height, int)
            assert isinstance(width,  int)
        img  = cv2.imread(path)
        if height is not None:
            h, w = img.shape[:2]
            h    = np.random.randint(0, h - height) if (h - height) > 0 else 0
            w    = np.random.randint(0, w - width ) if (w - width ) > 0 else 0
            img  = img[h:h+height, w:w+width]
        return img

    @classmethod
    def rotate_affine(cls, img: np.ndarray, angle: int, is_adjust_border: bool=False, center=None, color=[0, 0, 0]):
        assert center is None or (check_type_list(center, int) and len(center) == 2)
        if isinstance(color, int): color = [color, color, color]
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
        assert isinstance(img,  np.ndarray) and len(img.shape) == 3 and img.shape[-1] in [3, 4]
        if thre is not None:
            __class__.check_color_range(thre)
            mask = []
            if len(thre) == 2: thre = [thre, thre, thre]
            for i, (x, y) in enumerate(thre):
                mask.append(((img[:, :, i] >= x) & (img[:, :, i] <= y)).astype(np.uint8))
            mask = np.stack(mask)
            mask = (mask.sum(axis=0) == 3).astype(np.uint8) * 255
        else:
            if img.shape[-1] == 4:
                # if image has RGBA ch,     "cv2.imread(x, cv2.IMREAD_UNCHANGED)" function read image with (height, width, BGRA).
                # if image has only RGB ch, "cv2.imread(x, cv2.IMREAD_UNCHANGED)" function read image with (height, width, BGR).
                mask = ((img[:, :, -1] > 0) * 255).astype(np.uint8)
            else:
                mask = (np.ones(img.shape[:2]) * 255).astype(np.uint8)
        return mask
    
    @classmethod
    def get_mask_except_color(cls, img: np.ndarray, color: List[int]):
        mask = []
        if len(img.shape) == 3:
            for i, x in enumerate(color):
                mask.append(img[:, :, i] == x)
        else:
            for i, x in enumerate(color):
                mask.append(img[:, :, :, i] == x)
        mask = ~(np.stack(mask).astype(int).sum(axis=0) == 3)
        return mask

    @classmethod
    def img_stack(cls, list_imgs: List[np.ndarray], color: List[int]):
        img  = np.stack(list_imgs).astype(np.uint8)
        mask = __class__.get_mask_except_color(img, color)
        for i, _mask in enumerate(mask):
            if i == 0: continue
            img[0][_mask] = img[i][_mask]
        return img[0].astype(np.uint8)
    
    @classmethod
    def create_zero_imgs(cls, height: int=None, width: int=None, iters: int=None, color_init: List[int]=[255, 255, 255]):
        img = np.zeros((iters, height, width, 3), dtype=np.uint8)
        for i, x in enumerate(color_init): img[:, :, :, i] = x
        return img.astype(np.uint8)

    @classmethod
    def draw_text(
        cls, height: int=None, width: int=None, iters: int=None, color_init=None, chars: Union[str, List[str]]=None, n_connect: int=1,
        range_scale: List[int]=[0.5, 2.0], range_thickness: List[int]=[1, 3], range_color: Union[List[int], List[List[int]]]=[255, 255], 
        range_rotation: List[int]=[-20, 20], is_PIL: bool=False, font_pil: str=None, is_hanko: bool=False, padding_loc: Union[int, List[int]]=0, n_merge: int=1
    ):
        assert isinstance(iters, int) and iters >= 0
        if iters == 0: return []
        assert isinstance(height, int)
        assert isinstance(width,  int)
        if isinstance(chars, str): chars = list(chars)
        assert check_type_list(chars, str)
        assert isinstance(n_connect, int) and n_connect > 0
        assert check_type_list(range_scale, [int, float]) and len(range_scale)    == 2 and range_scale[0]    <= range_scale[1]
        assert check_type_list(range_rotation, int)       and len(range_rotation) == 2 and range_rotation[0] <= range_rotation[1]
        if is_PIL:
            if font_pil is None: font_pil=f"/{kkannotation.__path__[0]}/font/ipaexg.ttf"
            assert isinstance(font_pil, str)
        else: assert check_type_list(range_thickness, int) and len(range_thickness) == 2 and range_thickness[0] <= range_thickness[1]
        if not isinstance(padding_loc, list): padding_loc = [padding_loc, padding_loc]
        assert check_type_list(padding_loc, int) and len(padding_loc) == 2 and sum([x < 0 for x in padding_loc]) == 0
        assert isinstance(n_merge, int) and n_merge >= 1 and (iters % n_merge == 0)
        range_color     = __class__.convert_color_range(range_color, iters=iters).tolist()
        img             = __class__.create_zero_imgs(height=height, width=width, iters=iters, color_init=color_init)
        chars           = np.array(chars).copy()
        indexes         = np.random.randint(0, chars.shape[0], (n_connect, iters))
        chars           = np.vectorize(lambda *x: "".join(x))(*[chars[index] for index in indexes])   
        range_scale     = np.arange(range_scale[0], range_scale[1], (range_scale[1] - range_scale[0])/100.0)
        range_scale     = range_scale[np.random.randint(0, range_scale.shape[0], iters)]
        range_thickness = np.arange(range_thickness[0], range_thickness[1], 1)
        range_thickness = range_thickness[np.random.randint(0, range_thickness.shape[0], iters)]
        range_rotation  = np.arange(range_rotation[0], range_rotation[1]+1, 1)
        range_rotation  = range_rotation[np.random.randint(0, range_rotation.shape[0], iters)]
        loc_x           = np.random.randint(0 + padding_loc[0], width  - padding_loc[0],  iters).tolist()
        loc_y           = np.random.randint(0 + padding_loc[1], height - padding_loc[1], iters).tolist()
        list_imgs       = []
        list_imgs_tmp   = []
        for i in np.arange(iters):
            loc = [loc_x[i], loc_y[i]]
            if is_hanko:
                tmp_h   = __class__.draw_hanko(chars[i], font_pil, basesize=int(range_scale[i]), color=tuple(range_color[i]), color_init=color_init)
                tmp     = img[i]
                tmp_h_w = min(tmp_h.shape[0], width  - loc[0])
                tmp_h_h = min(tmp_h.shape[1], height - loc[1])
                tmp[loc[1]:loc[1]+tmp_h_h, loc[0]:loc[0]+tmp_h_w, :] = tmp_h[:tmp_h_h, :tmp_h_w, :]
            elif is_PIL:
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
            if n_merge > 1:
                list_imgs_tmp.append(tmp)
                if len(list_imgs_tmp) >= n_merge:
                    list_imgs.append(__class__.img_stack(list_imgs_tmp, color_init))
                    list_imgs_tmp = []
            else:
                list_imgs.append(tmp)
        return list_imgs
    
    @classmethod
    def draw_hanko(cls, text: str, font_pil: str, basesize: int=50, is_vertical: bool=True, thickness: int=2, color=None, color_init=None, padding: int=1):
        if basesize % 2 == 1: basesize += 1
        text     = text.replace("\n", "")
        if is_vertical:
            img  = __class__.create_zero_imgs(height=(basesize * len(text)), width=basesize, iters=1, color_init=color_init)[0]
            text = "\n".join(list(text))
        else:
            img  = __class__.create_zero_imgs(height=basesize, width=(basesize * len(text)), iters=1, color_init=color_init)[0]
        tmp      = cv2pil(img)
        draw     = ImageDraw.Draw(tmp)
        fontPIL  = ImageFont.truetype(font=font_pil, size=basesize)
        draw.text(
            xy=(0, 0),
            text=text,
            fill=color,
            font=fontPIL
        )
        tmp = tmp.resize((basesize, basesize))
        tmp = pil2cv(tmp)
        img = __class__.create_zero_imgs(height=(basesize * 2), width=(basesize * 2), iters=1, color_init=color_init)[0]
        img[basesize//2:basesize//2+tmp.shape[0], basesize//2:basesize//2+tmp.shape[1], :] = tmp
        r   = int((((basesize//2) ** 2) * 2) ** (1/2))
        img = cv2.circle(img, (basesize, basesize), r, color, thickness=thickness)
        margin = max(basesize - r - padding, 1)
        return img[margin:-margin, margin:-margin, :]

    @classmethod
    def draw_shape_line(
        cls, height: int=None, width: int=None, iters: int=None, color_init=None, range_thickness: List[int]=[1, 3], 
        range_color: Union[List[int], List[List[int]]]=[255, 255], n_lines: int=1, n_merge: int=1
    ):
        assert isinstance(iters, int) and iters >= 0
        if iters == 0: return []
        assert isinstance(height, int)
        assert isinstance(width,  int)
        assert check_type_list(range_thickness, int) and len(range_thickness) == 2 and range_thickness[0] <= range_thickness[1]
        assert isinstance(n_merge, int) and n_merge >= 1 and (iters % n_merge == 0)
        range_color      = __class__.convert_color_range(range_color, iters=iters).tolist()
        img              = __class__.create_zero_imgs(height=height, width=width, iters=iters, color_init=color_init)
        range_thickness  = np.arange(range_thickness[0], range_thickness[1], 1)
        range_thickness  = range_thickness[np.random.randint(0, range_thickness.shape[0], iters)]
        loc_x            = np.random.randint(0, width,  iters*2).reshape(-1, 2).tolist()
        loc_y            = np.random.randint(0, height, iters*2).reshape(-1, 2).tolist()
        mean             = (height + width) / 2
        range_space      = np.random.randint(mean//100, mean//20, iters)
        list_imgs        = []
        list_imgs_tmp    = []
        for i in np.arange(iters):
            p1_x, p1_y = loc_x[i][0], loc_y[i][0]
            p2_x, p2_y = loc_x[i][1], loc_y[i][1]
            tmp = img[i]
            for j in range(n_lines):
                tmp = cv2.line(tmp, (p1_x, p1_y+(j*range_space[i])), (p2_x, p2_y+(j*range_space[i])), tuple(range_color[i]), thickness=range_thickness[i], lineType=cv2.LINE_8, shift=0)
            if n_merge > 1:
                list_imgs_tmp.append(tmp)
                if len(list_imgs_tmp) >= n_merge:
                    list_imgs.append(__class__.img_stack(list_imgs_tmp, color_init))
                    list_imgs_tmp = []
            else:
                list_imgs.append(tmp)
        return list_imgs

    @classmethod
    def draw_shape_circle(
        cls, height: int=None, width: int=None, iters: int=None, color_init=None, range_thickness: List[int]=[1, 3], 
        range_color: Union[List[int], List[List[int]]]=[255, 255], range_radius: List[int]=[1, 3], n_merge: int=1
    ):
        assert isinstance(iters, int) and iters >= 0
        if iters == 0: return []
        assert isinstance(height, int)
        assert isinstance(width,  int)
        assert check_type_list(range_thickness, int) and len(range_thickness) == 2 and range_thickness[0] <= range_thickness[1]
        assert check_type_list(range_radius,    int) and len(range_radius)    == 2 and range_radius[0]    <= range_radius[1]
        assert isinstance(n_merge, int) and n_merge >= 1 and (iters % n_merge == 0)
        range_color      = __class__.convert_color_range(range_color, iters=iters).tolist()
        img              = __class__.create_zero_imgs(height=height, width=width, iters=iters, color_init=color_init)
        range_thickness  = np.arange(range_thickness[0], range_thickness[1], 1)
        range_thickness  = range_thickness[np.random.randint(0, range_thickness.shape[0], iters)]
        range_radius     = np.arange(range_radius[0], range_radius[1], 1)
        range_radius     = range_radius[np.random.randint(0, range_radius.shape[0], iters)]
        loc_x            = np.random.randint(0, width,  iters).tolist()
        loc_y            = np.random.randint(0, height, iters).tolist()
        list_imgs        = []
        list_imgs_tmp    = []
        for i in np.arange(iters):
            p1_x, p1_y = loc_x[i], loc_y[i]
            tmp = cv2.circle(img[i], (p1_x, p1_y), range_radius[i], range_color[i], thickness=range_thickness[i])
            if n_merge > 1:
                list_imgs_tmp.append(tmp)
                if len(list_imgs_tmp) >= n_merge:
                    list_imgs.append(__class__.img_stack(list_imgs_tmp, color_init))
                    list_imgs_tmp = []
            else:
                list_imgs.append(tmp)
        return list_imgs

    @classmethod
    def draw_shape_ellipse(
        cls, height: int=None, width: int=None, iters: int=None, color_init=None, range_thickness: List[int]=[1, 3], 
        range_color: Union[List[int], List[List[int]]]=[255, 255], range_rotation: List[int]=[-20, 20],
        range_scale: List[int]=[100, 200], n_merge: int=1
    ):
        assert isinstance(iters, int) and iters >= 0
        if iters == 0: return []
        assert isinstance(height, int)
        assert isinstance(width,  int)
        assert check_type_list(range_thickness, int) and len(range_thickness) == 2 and range_thickness[0] <= range_thickness[1]
        assert check_type_list(range_rotation, int)  and len(range_rotation)  == 2 and range_rotation[0]  <= range_rotation[1]
        assert check_type_list(range_scale, int)     and len(range_scale)     == 2 and range_scale[0]     <= range_scale[1]
        assert isinstance(n_merge, int) and n_merge >= 1 and (iters % n_merge == 0)
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
        list_imgs_tmp    = []
        for i in np.arange(iters):
            p1_x, p1_y = loc_x[i], loc_y[i]
            tmp = cv2.ellipse(img[i], ((p1_x, p1_y), (range_scale[i][0], range_scale[i][1]), range_rotation[i]), range_color[i], thickness=range_thickness[i])
            if n_merge > 1:
                list_imgs_tmp.append(tmp)
                if len(list_imgs_tmp) >= n_merge:
                    list_imgs.append(__class__.img_stack(list_imgs_tmp, color_init))
                    list_imgs_tmp = []
            else:
                list_imgs.append(tmp)
        return list_imgs

    @classmethod
    def draw_shape_rectangle(
        cls, height: int=None, width: int=None, iters: int=None, color_init=None, range_thickness: List[int]=[1, 3], 
        range_color: Union[List[int], List[List[int]]]=[255, 255], range_rotation: List[int]=[-20, 20],
        range_scale: List[int]=[100, 200], n_merge: int=1
    ):
        assert isinstance(iters, int) and iters >= 0
        if iters == 0: return []
        assert isinstance(height, int)
        assert isinstance(width,  int)
        assert check_type_list(range_thickness, int) and len(range_thickness) == 2 and range_thickness[0] <= range_thickness[1]
        assert check_type_list(range_rotation, int)  and len(range_rotation)  == 2 and range_rotation[0]  <= range_rotation[1]
        assert check_type_list(range_scale, int)     and len(range_scale)     == 2 and range_scale[0]     <= range_scale[1]
        assert isinstance(n_merge, int) and n_merge >= 1 and (iters % n_merge == 0)
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
        list_imgs_tmp    = []
        ndf              = np.ones(iters).astype(int)
        ndf[np.random.randint(0, 2, iters).astype(bool)] = -1
        for i in np.arange(iters):
            p1_x, p1_y = loc_x[i], loc_y[i]
            p2_x, p2_y = p1_x + ndf[i] * range_scale[i][0], p1_y + ndf[i] * range_scale[i][1]
            tmp = cv2.rectangle(img[i], (p1_x, p1_y), (p2_x, p2_y), range_color[i], thickness=range_thickness[i], lineType=cv2.LINE_8, shift=0)
            tmp = __class__.rotate_affine(tmp, int(range_rotation[i]), is_adjust_border=False, center=[p1_x, p1_y], color=color_init)
            if n_merge > 1:
                list_imgs_tmp.append(tmp)
                if len(list_imgs_tmp) >= n_merge:
                    list_imgs.append(__class__.img_stack(list_imgs_tmp, color_init))
                    list_imgs_tmp = []
            else:
                list_imgs.append(tmp)
        return list_imgs
    
    @classmethod
    def draw_label(
        cls, height: int=None, width: int=None, iters: int=None, color_init=None, 
        imgs_label: List[np.ndarray]=None, imgs_mask: List[np.ndarray]=None, imgs_name: np.ndarray=None,
        range_noise: Union[List[int], List[List[int]]]=None, range_scale: List[float]=[0.5, 2], 
        range_rotation: List[int]=[-20, 20], is_fix_scale_ratio: bool=True, padding_loc: Union[int, List[int]]=0, n_merge: int=1
    ):
        assert isinstance(iters, int) and iters > 0
        assert isinstance(height, int)
        assert isinstance(width,  int)
        assert check_type_list(imgs_label, np.ndarray)
        assert check_type_list(imgs_mask,  np.ndarray)
        assert isinstance(imgs_name, np.ndarray)
        assert check_type_list(range_scale, [int, float]) and len(range_scale) == 2 and range_scale[0] <= range_scale[1]
        if not isinstance(padding_loc, list): padding_loc = [padding_loc, padding_loc]
        assert check_type_list(padding_loc, int) and len(padding_loc) == 2 and sum([x < 0 for x in padding_loc]) == 0
        if n_merge is None: n_merge = 1
        assert isinstance(n_merge, int) and n_merge >= 1 and (iters % n_merge == 0)
        img         = __class__.create_zero_imgs(height=height, width=width, iters=iters, color_init=color_init)
        mask        = np.zeros((iters, height, width)).astype(np.uint8)
        indexes     = np.random.randint(0, len(imgs_label), iters)
        imgs_label  = np.array([x.copy() for x in imgs_label] + [np.zeros(0)], dtype=object) # Prevent unification with np.ndarray
        imgs_label  = imgs_label[indexes]
        imgs_mask   = np.array([x.copy() for x in imgs_mask ] + [np.zeros(0)], dtype=object) # Prevent unification with np.ndarray
        imgs_mask   = imgs_mask[indexes]
        range_scale = np.arange(range_scale[0], range_scale[1], (range_scale[1] - range_scale[0])/100.0)
        if is_fix_scale_ratio:
            range_scale = range_scale[np.random.randint(0, range_scale.shape[0], iters)]
            imgs_label  = np.vectorize(lambda x, y: cv2.resize(x, dsize=None, fx=y, fy=y), otypes=[object])(imgs_label, range_scale)
            imgs_mask   = np.vectorize(lambda x, y: cv2.resize(x, dsize=None, fx=y, fy=y), otypes=[object])(imgs_mask,  range_scale)
        else:
            range_scale = range_scale[np.random.randint(0, range_scale.shape[0], iters*2)].reshape(2, -1)
            imgs_label  = np.vectorize(lambda x, y, z: cv2.resize(x, dsize=None, fx=y, fy=z), otypes=[object])(imgs_label, range_scale[0], range_scale[1])
            imgs_mask   = np.vectorize(lambda x, y, z: cv2.resize(x, dsize=None, fx=y, fy=z), otypes=[object])(imgs_mask,  range_scale[0], range_scale[1])
        range_rotation = np.arange(range_rotation[0], range_rotation[1]+1, 1)
        range_rotation = range_rotation[np.random.randint(0, range_rotation.shape[0], iters)]
        imgs_label     = np.vectorize(
            lambda x, y: __class__.rotate_affine(x, int(y), is_adjust_border=True, center=None, color=color_init), otypes=[object]
        )(imgs_label, range_rotation)
        imgs_mask = np.vectorize(
            lambda x, y: __class__.rotate_affine(x, int(y), is_adjust_border=True, center=None, color=0), otypes=[object]
        )(imgs_mask, range_rotation)
        loc_x = np.vectorize(lambda x: np.random.randint(0 + padding_loc[0], width  - x.shape[1] - 1 - padding_loc[0]))(imgs_label)
        loc_y = np.vectorize(lambda x: np.random.randint(0 + padding_loc[1], height - x.shape[0] - 1 - padding_loc[1]))(imgs_label)
        for i in np.arange(iters):
            p1_x, p1_y = loc_x[i], loc_y[i]
            img[ i, p1_y:p1_y+imgs_label[i].shape[0], p1_x:p1_x+imgs_label[i].shape[1], :] = imgs_label[i]
            mask[i, p1_y:p1_y+imgs_mask[ i].shape[0], p1_x:p1_x+imgs_mask[ i].shape[1]]    = imgs_mask[i]
        mask = (mask > 0)
        if range_noise is not None:
            assert (check_type_list(range_noise, int) and len(range_noise) == 2) or (check_type_list(range_noise, list, int) and len(range_noise) == 3)
            if len(range_noise) == 2:
                noise = np.random.randint(*range_noise, (iters, height, width, 1))
                noise = np.concatenate([noise, noise, noise], axis=-1)
            else:
                noise = [np.random.randint(x_min, x_max, (iters, height, width, 1)) for x_min, x_max in range_noise]
                noise = np.concatenate(noise, axis=-1)
            img = (img.astype(np.int16) + noise.astype(np.int16))
            img[img < 0]    = 0
            img[img >= 256] = 255
            img = img.astype(np.uint8)
        if n_merge > 1:
            # in this case, indexes does not make sense
            list_img, list_mask, tmp, tmp_m = [], [], None, None
            for i, (_img, _mask) in enumerate(zip(img, mask)):
                if tmp is None:
                    tmp, tmp_m = _img, _mask.astype(np.int16)
                else:
                    tmp[_mask] = _img[_mask]
                    tmp_m      = tmp_m + _mask.astype(np.int16)
                if (i + 1) % n_merge == 0:
                    list_img. append(tmp.copy())
                    list_mask.append(tmp_m.copy())
                    tmp, tmp_m = None, None
            img  = np.stack(list_img).astype(np.uint8)
            mask = np.stack(list_mask).astype(bool)
        return img, mask, imgs_name[indexes]
