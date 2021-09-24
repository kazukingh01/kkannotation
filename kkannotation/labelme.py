import os, json, datetime
from typing import List
import pandas as pd
import numpy as np
import cv2

# local package
from kkannotation.coco import CocoManager
from kkannotation.util.com import check_type_list, correct_dirpath, makedirs, get_file_list
from kkannotation.util.logger import set_logger
logger = set_logger(__name__)


__all__ = [
    "Labelme2Coco",
]


class Labelme2Coco:
    def __init__(
        self, dirpath_json: str, dirpath_img: str, 
        categories_name: List[str], keypoints: List[str] = None, 
        keypoints_belong: dict=None, skelton: List[List[str]] = None
    ):
        """
        Params::
            dirpath_json: labelme dirpath contains jsons.
            dirpath_img:  labelme dirpath contains images.
        Usage::
            >>> from kkannotation.labelme import Labelme2Coco
            >>> labelme2coco = Labelme2Coco(
                    dirpath_json="./json", dirpath_img="./img",
                    categories_name=["dog", "cat"],
                    keypoints=["eye_left", "eye_right", "nose", "mouth"],
                    keypoints_belong={
                        "dog": ["eye_left", "eye_right", "nose", "mouth"],
                        "cat": ["eye_left", "eye_right", "nose", "mouth"],
                    },
                    skelton=[["eye_left", "eye_right"], ["nose", "mouht"]]
                )
            >>> labelme2coco.to_coco("coco.json")
        """
        assert isinstance(dirpath_json, str)
        assert isinstance(dirpath_img,  str)
        assert check_type_list(categories_name, str)
        if keypoints is not None:
            assert check_type_list(keypoints, str)
            assert isinstance(keypoints_belong, dict)
            assert skelton is None or check_type_list(skelton, list, str)
        self.dirpath_json          = correct_dirpath(dirpath_json)
        self.dirpath_img           = correct_dirpath(dirpath_img)
        self.categories_name       = categories_name
        self.index_categories_name = {x:i for i, x in enumerate(categories_name)}
        self.keypoints             = keypoints
        self.keypoints_belong      = keypoints_belong
        self.index_keypoints       = {x:i for i, x in enumerate(keypoints)} if keypoints is not None else {}
        self.skelton               = skelton

    def read_json(self, json_path: str) -> pd.DataFrame:
        """
        Create a DataFrame that is compatible with CocoManager.
        """
        logger.info(f"read json file: {json_path}")
        fjson = json.load(open(json_path))
        if fjson.get("imagePath") is None: return pd.DataFrame()
        fname = self.dirpath_img + os.path.basename(fjson["imagePath"])
        img   = cv2.imread(fname)
        # labelme json to dataframe
        df = pd.DataFrame()
        df_json = pd.DataFrame(fjson["shapes"])
        if (df_json.columns == "shape_type").sum() == 0: return pd.DataFrame()
        for i, (label, points, shape_type) in enumerate(df_json[df_json["shape_type"].isin(["polygon","rectangle"])][["label","points","shape_type"]].values):
            if self.index_categories_name.get(label) is None: continue
            x1, y1, x2, y2 = None, None, None, None
            x_min, y_min, width, height, segs, area, kpts = None, None, None, None, None, None, None
            # bbox
            if   shape_type == "rectangle":
                [[x1, y1], [x2, y2]] = points
                if x1 > x2: x2, x1 = points[0][0], points[1][0]
                if y1 > y2: y2, y1 = points[0][1], points[1][1]
                x_min, y_min, width, height = int(x1), int(y1), int(x2-x1), int(y2-y1)
            # bbox and segmentation
            elif shape_type == "polygon":
                ndf = np.zeros_like(img).astype(np.uint8)
                ndf = cv2.polylines(ndf, [np.array(points).reshape(-1,1,2).astype(np.int32)], True, (255,0,0))
                contours = cv2.findContours(ndf[:, :, 0], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
                x_min, y_min, width, height = cv2.boundingRect(contours[0])
                x_min, y_min, width, height = int(x_min), int(y_min), int(width), int(height)            
                segs = [np.array(points).reshape(-1).astype(int).tolist()]
                area = cv2.contourArea(contours[0])
                x1, y1, x2, y2 = x_min, y_min, x_min + width, y_min + height           
            # keypoint
            if self.keypoints is not None:
                list_kpt  = self.keypoints_belong[label]
                dfwk      = df_json[(df_json["shape_type"] == "point") & (df_json["label"].isin(list_kpt))].copy()
                dfwk["x"] = dfwk["points"].map(lambda x: x[0][0])
                dfwk["y"] = dfwk["points"].map(lambda x: x[0][1])
                dfwk      = dfwk.loc[((dfwk["x"] >= x1) & (dfwk["x"] <= x2) & (dfwk["y"] >= y1) & (dfwk["y"] <= y2)), :]
                ndf       = np.zeros(0)
                for keyname in self.keypoints:
                    dfwkwk = dfwk[dfwk["label"] == keyname]
                    if dfwkwk.shape[0] > 0:
                        sewk = dfwkwk.iloc[0]
                        ndf = np.append(ndf, (sewk["x"], sewk["y"], 2,))
                    else:
                        ndf = np.append(ndf, (0, 0, 0,))
                kpts = ndf.reshape(-1).tolist()
            se = CocoManager.to_series(
                fname, img.shape[0], img.shape[1],
                (x_min, y_min, width, height), 0, 0, self.index_categories_name.get(label), label,
                super_category_name=None, segmentations=segs, area=area,
                keypoints=kpts, category_name_kpts=self.keypoints,
            )
            df = df.append(se, ignore_index=True)
        return df

    def to_coco_manager(self) -> CocoManager:
        df = pd.DataFrame()
        for x in get_file_list(self.dirpath_json, regex_list=[r"\.json"]):
            dfwk = self.read_json(x)
            df   = pd.concat([df, dfwk], ignore_index=True, sort=False, axis=0)
        coco = CocoManager()
        coco.df_json = df.copy()
        coco.re_index()
        return coco

    def save_coco_file(self, save_path: str, save_images_path: str=None, exist_ok: bool=False, remake: bool=False):
        coco = self.to_coco_manager()
        coco.save(save_path, save_images_path=save_images_path, exist_ok=exist_ok, remake=remake)
