import os
from typing import List, Tuple, Union
import numpy as np
import cv2

# local package
from kkannotation.util.com import check_type, correct_dirpath, makedirs
from kkannotation.util.logger import set_logger
logger = set_logger(__name__)


__all__ = [
    "Streamer",
]


class Streamer:
    def __init__(self, src: Union[str, int]):
        """
        Params::
            src:
                string or integer.
                if integer, use your PC Camera.
        Usage::
            if string
            >>> from kkannotation.streamer import Streamer
            >>> video = Streamer("hogehoge.mp4")
            >>> video.play()
            if integer
            >>> from kkannotation.streamer import Streamer
            >>> video = Streamer(0)
            >>> video.play()
        """
        assert check_type(src, [str, int])
        logger.info(f"open src: {src}", color=["BOLD", "GREEN"])
        self.cap = cv2.VideoCapture(src)
        self.src = src

    def __str__(self):
        return f"{self.__class__.__name__}({self.src})"

    def __len__(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __getitem__(self, index) -> np.ndarray:
        frame_id = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        is_next, frame = self.cap.read()
        if is_next == False: raise IndexError
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        return frame

    def __del__(self):
        self.cap.release()

    def __iter__(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self

    def __next__(self):
        is_next, frame = self.cap.read()
        if is_next == False: raise StopIteration()
        return frame

    def close(self):
        self.__del__()
        logger.info(f"close {self}", color=["BOLD", "GREEN"])

    def shape(self) -> (int, int):
        """
        Return:: (height, width)
        """
        width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (height, width)

    def get_fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS)

    def play(self):
        for frame in self:
            cv2.imshow('__window', frame)
            if cv2.waitKey(30) & 0xff == 27: # quit with ESQ key
                break

    def save_images(self, outdir: str, step: int = 1, max_images: int=None, exist_ok: bool=False, remake: bool=False):
        assert isinstance(step, int) and step > 0
        if max_images is not None:
            assert isinstance(max_images, int) and max_images > 0
        outdir = correct_dirpath(outdir)
        bname  = os.path.basename(self.src) if isinstance(self.src, str) else f"camera{self.src}"
        count  = 0
        maxlen = int(np.log10(len(self))) + 1
        makedirs(outdir, exist_ok=exist_ok, remake=remake)
        for i, frame in enumerate(self):
            if i % step == 0:
                filename = f"{outdir}{bname}.{str(i).zfill(maxlen)}.png"
                cv2.imwrite(filename, frame)
                count += 1
                if max_images is not None and count > max_images:
                    logger.warning(f"number of saved images are reached max count: {max_images}")
                    break

