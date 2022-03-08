from typing import List, Union
import numpy as np
import cv2

# local package
from kkannotation.util.com import check_type_list


__all__ = [
    "SymbolEmbedding"
]


class SymbolEmbedding:
    def __init__(
        self, height: int, width: int, range_noise: Union[List[int], List[List[int]]]=[255, 256],
        range_scale: List[int]=[0.5, 2.0], 
        range_thickness: List[int]=[1, 3], range_color: Union[List[int], List[List[int]]]=[255, 255]
    ):
        pass

    def create_image(self):
        img = self.create_canvas(self.height, self.width, range_noise=self.range_noise)
        img = self.draw_text_with_numpy(
            img, self.chars_numpy, self.iters, range_scale=self.range_scale, 
            range_thickness=self.range_thickness, range_color=self.range_color
        )
        return img

    @classmethod
    def create_canvas(cls, height: int, width: int, range_noise: Union[List[int], List[List[int]]]=[255, 256]):
        assert isinstance(height, int)
        assert isinstance(width,  int)
        __class__.check_color_range(range_noise)
        img = np.zeros((height, width, 3)).astype(np.uint8)
        if len(range_noise) == 2:
            noise = np.random.randint(*range_noise, (height, width)).astype(np.uint8)
            noise = noise.reshape(*noise.shape, 1).repeat(3, axis=-1)
            img   = noise
        else:
            list_noise = []
            for x in range_noise:
                list_noise.append(np.random.randint(*x, (height, width)).astype(np.uint8).reshape(height, width, 1))
            img = np.concatenate(list_noise, axis=-1)
        return img

    @classmethod
    def draw_text_with_numpy(
        cls, img: np.ndarray, chars: List[str], iters: int, range_scale: List[int]=[0.5, 2.0], 
        range_thickness: List[int]=[1, 3], range_color: Union[List[int], List[List[int]]]=[255, 255]
    ):
        assert isinstance(img, np.ndarray)
        assert check_type_list(chars, str)
        assert check_type_list(range_scale, float)   and len(range_scale)     == 2 and range_scale[0]     < range_scale[1]
        assert check_type_list(range_thickness, int) and len(range_thickness) == 2 and range_thickness[0] < range_thickness[1]
        __class__.check_color_range(range_color)
        img              = img.copy()
        height, width, _ = img.shape
        chars            = np.array(chars).copy()
        chars            = chars[np.random.randint(0, chars.shape[0], iters)]
        range_scale      = np.arange(range_scale[0], range_scale[1], (range_scale[1] - range_scale[0])/100.0)
        range_scale      = range_scale[np.random.randint(0, range_scale.shape[0], iters)]
        range_thickness  = np.arange(range_thickness[0], range_thickness[1]+1, 1)
        range_thickness  = range_thickness[np.random.randint(0, range_thickness.shape[0], iters)]
        loc_x            = np.random.randint(0, width,  iters)
        loc_y            = np.random.randint(0, height, iters)
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
        for i in np.arange(iters):
            img = cv2.putText(
                img,
                text=chars[i],
                org=(loc_x[i], loc_y[i]),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=range_scale[i], 
                color=tuple(range_color[i]), 
                thickness=range_thickness[i],
                lineType=cv2.LINE_AA,
            )
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

