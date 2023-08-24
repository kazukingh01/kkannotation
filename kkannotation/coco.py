import json, os, shutil, datetime
import pandas as pd
import numpy as np
import cv2
from typing import List, Union
from tqdm import tqdm

# local package
from kkannotation.util.image import draw_annotation, fit_resize
from kkannotation.util.com import check_type, check_type_list, correct_dirpath, makedirs
from kkannotation.util.logger import set_logger
logger = set_logger(__name__)


__all__ = [
    "coco_info",
    "CocoManager"
]


COCO_INFO_TMP_DESCROPTION  = "my coco dataset."
COCO_INFO_TMP_URL          = "http://test"
COCO_INFO_TMP_VERSION      = "1.0"
COCO_INFO_TMP_CONTRIBUTION = "Test"
COCO_INFO_TMP_LICENSE_NAME = "test license"


def coco_info(
    description: str  = COCO_INFO_TMP_DESCROPTION,
    url: str          = COCO_INFO_TMP_URL,
    version: str      = COCO_INFO_TMP_VERSION,
    year: str         = datetime.datetime.now().strftime("%Y"), 
    contributor: str  = COCO_INFO_TMP_CONTRIBUTION,
    date_created: str = datetime.datetime.now().strftime("%Y/%m/%d")
):
    info = {}
    info["description"]  = description
    info["url"]          = url
    info["version"]      = version
    info["year"]         = year
    info["contributor"]  = contributor
    info["date_created"] = date_created
    return info


class CocoManager:
    """
    Coco format manager with pandas DataFrame
    Usage::
        >>> from kkannotation.coco import CocoManager
        >>> coco = CocoManager()
        >>> coco.add_json("./coco.json", root_dir="./img/")
        >>> coco.draw_annotations(0)
    """

    def __init__(self, src: Union[str, dict]=None, root_dir: str=None):
        self.df_json    = pd.DataFrame()
        self.json       = {} # Save the most recent one
        self.coco_info  = {}
        self.dict_paths = {"shape": -1}
        self._dict_imgpath = {}
        self._dict_ann     = {}
        self._dict_cat     = {}
        self._list_se      = []
        self.initialize()
        if src is not None:
            self.add_json(src=src, root_dir=root_dir)
    
    def __len__(self):
        return len(self.df_json["images_coco_url"].unique())
    
    def __getitem__(self, index):
        if self.dict_paths["shape"] != self.df_json.shape[0]:
            self.dict_paths["shape"] = self.df_json.shape[0]
            self.dict_paths["paths"] = np.sort(self.df_json["images_coco_url"].unique())
        return self.df_json.loc[self.df_json["images_coco_url"] == self.dict_paths["paths"][index]].copy()

    def initialize(self):
        """
        images_id                                                                    0
        images_coco_url                                       ./traindata/000000.0.png
        images_date_captured                                       2020-07-30 16:07:12
        images_file_name                                                  000000.0.png
        images_flickr_url                                                  http://test
        images_height                                                             1400
        images_license                                                               0
        images_width                                                              1400
        annotations_id                                                               0
        annotations_area                                                         11771
        annotations_bbox             [727.0316772460938, 596.92626953125, 124.13433...
        annotations_category_id                                                      0
        annotations_image_id                                                         0
        annotations_iscrowd                                                          0
        annotations_keypoints        [834, 855, 2, 821, 649, 2, 780, 615, 2, 750, 6...
        annotations_num_keypoints                                                    5
        annotations_segmentation     [[773, 602, 772, 603, 771, 603, 770, 603, 769,...
        licenses_id                                                                  0
        licenses_name                                                     test license
        licenses_url                                                       http://test
        categories_id                                                                0
        categories_keypoints         [kpt_a, kpt_cb, kpt_c, kpt_cd, kpt_e, kpt_b, k...
        categories_name                                                           hook
        categories_skeleton                   [[0, 1], [1, 2], [2, 3], [3, 4], [5, 6]]
        categories_supercategory                                                  hook
        """
        self.df_json = pd.DataFrame(
            columns=[
                'images_id', 'images_coco_url', 'images_date_captured',
                'images_file_name', 'images_flickr_url', 'images_height',
                'images_license', 'images_width', 'annotations_id', 'annotations_area',
                'annotations_bbox', 'annotations_category_id', 'annotations_image_id',
                'annotations_iscrowd', 'annotations_keypoints',
                'annotations_num_keypoints', 'annotations_segmentation', 'licenses_id',
                'licenses_name', 'licenses_url', 'categories_id',
                'categories_keypoints', 'categories_name', 'categories_skeleton',
                'categories_supercategory'
            ]
        )
        self.coco_info = coco_info()
    
    @classmethod
    def to_series(
        cls, imgpath: str, height: int, width: int, 
        bbox: (Union[int, float], Union[int, float], Union[int, float], Union[int, float]),
        image_id: int, annotation_id: int, category_id: int,
        category_name: str, super_category_name: str=None,
        segmentations: List[List[Union[int, float]]]=None, area: float=None,
        keypoints: List[float]=None,
        category_name_kpts: List[str]=None,
    ) -> pd.Series:
        """
        Create a coco format pd.Series and add inner list.
        Params::
            imgpath: image path
            bbox: [x_min, y_min, width, height]
            category_name: class name
            super_category_name: if you want to define, set super category name. default is None
            segmentations: [[x11, y11, x12, y12, ...], [x21, y21, x22, y22, ...], ..]
            keypoints: [x1, y1, vis1, x2, y2, vis2, ...]
            category_name_kpts: ["kpt1", "kpt2", ...]
        """
        # Check type
        assert isinstance(imgpath, str)
        assert isinstance(height, int) and height > 0
        assert isinstance(width, int) and width > 0
        assert check_type_list(bbox, [int, float]) and sum([(x >= 0) for x in bbox]) == 4
        assert isinstance(image_id, int) and image_id >= 0
        assert isinstance(annotation_id, int) and annotation_id >= 0
        assert isinstance(category_id, int) and category_id >= 0
        assert isinstance(category_name, str)
        if super_category_name is not None: assert isinstance(super_category_name, str)
        if area                is not None: assert check_type(area, [int, float]) and area > 0
        if category_name_kpts  is not None: assert check_type_list(category_name_kpts, str)
        if keypoints           is not None:
            assert check_type_list(keypoints, float)
            assert len(category_name_kpts) == (len(keypoints) // 3)
            keypoints = [round(x, 2) for x in keypoints]
        if segmentations       is not None:
            assert check_type_list(segmentations, list, [int, float])
            segmentations = [[round(x, 1) for x in y] for y in segmentations]
        se = pd.Series(dtype=object)
        se["images_id"]                 = image_id
        se["images_file_name"]          = os.path.basename(imgpath)
        se["images_coco_url"]           = imgpath
        se["images_date_captured"]      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        se["images_flickr_url"]         = COCO_INFO_TMP_URL
        se["images_height"]             = height
        se["images_width"]              = width
        se["images_license"]            = 0
        se["annotations_id"]            = annotation_id
        se["annotations_bbox"]          = [round(x, 2) for x in bbox]
        se["annotations_area"]          = int(se["annotations_bbox"][-2] * se["annotations_bbox"][-1]) if area is None else round(area, 2)
        se["annotations_category_id"]   = category_id
        se["annotations_image_id"]      = image_id
        se["annotations_iscrowd"]       = 0
        se["annotations_keypoints"]     = keypoints if keypoints is not None else []
        se["annotations_num_keypoints"] = len(keypoints)//3 if keypoints is not None else 0
        se["annotations_segmentation"]  = segmentations if segmentations is not None else []
        se["licenses_id"]               = 0
        se["licenses_name"]             = COCO_INFO_TMP_LICENSE_NAME
        se["licenses_url"]              = COCO_INFO_TMP_URL
        se["categories_id"]             = category_id
        se["categories_keypoints"]      = category_name_kpts if category_name_kpts is not None else []
        se["categories_name"]           = category_name
        se["categories_skeleton"]       = []
        se["categories_supercategory"]  = category_name if super_category_name is None else super_category_name
        return se

    def add(
        self, imgpath: str, height: int, width: int, 
        bbox: (float, float, float, float), 
        category_name: str, super_category_name: str=None,
        segmentations: List[List[float]]=None,
        keypoints: List[float]=None,
        category_name_kpts: List[str]=None,
    ):
        # Check already exist
        if imgpath       not in self._dict_imgpath: self._dict_imgpath[imgpath]   = len(self._dict_imgpath)
        if category_name not in self._dict_cat:     self._dict_cat[category_name] = len(self._dict_cat)
        self._dict_ann[imgpath] = (self._dict_ann[imgpath] + 1) if imgpath in self._dict_ann else 0
        se = self.to_series(
            imgpath, height, width, bbox, 
            self._dict_imgpath[imgpath], self._dict_ann[imgpath], self._dict_cat[category_name],
            category_name=category_name, super_category_name=super_category_name, segmentations=segmentations,
            keypoints=keypoints, category_name_kpts=category_name_kpts
        )
        self._list_se.append(se)
    
    def concat_added(self):
        if len(self._list_se) > 0:
            self.df_json = pd.concat(self._list_se, ignore_index=True, sort=False, axis=1).T
        self.re_index()

    @classmethod
    def coco_json_to_df(cls, src: Union[str, dict]):
        """
        convert coco format dict to dataframe.
        Params::
            src:
                coco file path or dictionary with coco format.
        """
        assert check_type(src, [str, dict])
        json_coco = {}
        if   isinstance(src, str):  json_coco = json.load(open(src))
        elif isinstance(src, dict): json_coco = src
        df_img = pd.DataFrame(json_coco["images"])
        df_ann = pd.DataFrame(json_coco["annotations"])
        df_lic = pd.DataFrame(json_coco["licenses"])
        df_cat = pd.DataFrame(json_coco["categories"])
        df_img.columns = ["images_"+x      for x in df_img.columns]
        df_ann.columns = ["annotations_"+x for x in df_ann.columns]
        df_lic.columns = ["licenses_"+x    for x in df_lic.columns]
        df_cat.columns = ["categories_"+x  for x in df_cat.columns]
        df = pd.merge(df_img, df_ann, how="left", left_on="images_id",               right_on="annotations_image_id")
        df = pd.merge(df, df_lic,     how="left", left_on="images_license",          right_on="licenses_id")
        df = pd.merge(df, df_cat,     how="left", left_on="annotations_category_id", right_on="categories_id")
        if (df.columns == "images_license").sum() == 0: df["images_license"] = 0
        return df

    def check_index(self):
        """
        Check if the file name is the same but the path is different.
        """
        se = self.df_json.groupby("images_file_name")["images_coco_url"].apply(lambda x: x.unique())
        if (se.apply(lambda x: len(x) > 1)).sum() > 0:
            logger.warning(f"same file name: [{se[se.apply(lambda x: len(x) > 1)].index.values}]")

    def re_index(
        self, keypoints: List[str]=None, skeleton: List[List[str]]=None
    ):
        """
        Organize the index.
        Params::
            keypoints:
                If there is a keypoint name that you want to keep when organizing, specify it.
        """
        # Check
        if keypoints is not None:
            assert check_type_list(keypoints, str)
            assert skeleton is not None and check_type_list(skeleton, list, str)
        # Add any missing columns.
        for name, default_value in zip(
            ["categories_keypoints", "categories_skeleton", "licenses_name", "licenses_url", "annotations_segmentation", "annotations_keypoints"], 
            [[], [], np.nan, np.nan, [], []]
        ):
            if (self.df_json.columns == name).sum() == 0:
                self.df_json[name] = [default_value for _ in np.arange(self.df_json.shape[0])]
        # image id
        dictwk = {x:i  for i, x in enumerate(np.sort(self.df_json["images_coco_url"].unique()))}
        self.df_json["images_id"]            = self.df_json["images_coco_url"].map(dictwk)
        self.df_json["annotations_image_id"] = self.df_json["images_coco_url"].map(dictwk)
        # license id
        self.df_json["licenses_name"] = self.df_json["licenses_name"].fillna(COCO_INFO_TMP_LICENSE_NAME)
        self.df_json["licenses_url"]  = self.df_json["licenses_url"]. fillna(COCO_INFO_TMP_URL)
        dictwk = {x:i  for i, x in enumerate(np.sort(self.df_json["licenses_name"].unique()))}
        self.df_json["images_license"] = self.df_json["licenses_name"].map(dictwk)
        self.df_json["licenses_id"]    = self.df_json["licenses_name"].map(dictwk)
        # category id
        if self.df_json.columns.isin(["categories_supercategory"]).sum() == 0:
            self.df_json["categories_supercategory"] = self.df_json["categories_name"].copy()
        self.df_json["__work"] = (self.df_json["categories_supercategory"].astype(str) + "_" + self.df_json["categories_name"].astype(str)).copy()
        dictwk = {x:i  for i, x in enumerate(np.sort(self.df_json["__work"].unique()))}
        self.df_json["annotations_category_id"] = self.df_json["__work"].map(dictwk)
        self.df_json["categories_id"]           = self.df_json["__work"].map(dictwk)
        self.df_json = self.df_json.drop(columns=["__work"])
        # annotations id ( equal df's index )
        self.df_json = self.df_json.reset_index(drop=True)
        self.df_json["annotations_id"] = self.df_json.index.values
        # concat keypoint
        if keypoints is not None:
            self.df_json["annotations_keypoints"] = self.df_json.apply(
                lambda x: np.array(
                    [x["annotations_keypoints"][np.where(np.array(x["categories_keypoints"]) == y)[0].min()*3:np.where(np.array(x["categories_keypoints"]) == y)[0].min()*3+3]
                    if y in x["categories_keypoints"] else [0,0,0] for y in keypoints]
                ).reshape(-1).tolist(), axis=1
            )
            self.df_json["categories_keypoints"] = self.df_json["categories_keypoints"].apply(lambda x: keypoints)
            skeleton = np.array(skeleton).copy()
            for i, x in enumerate(keypoints):
                skeleton[skeleton == x] = i
            skeleton = skeleton.astype(int)
            self.df_json["categories_skeleton"] = self.df_json["categories_skeleton"].apply(lambda x: skeleton.tolist())
        # convert type
        for x in [
            "images_id", "images_height", "images_width", "images_license", 
            "annotations_id", "annotations_category_id", "annotations_image_id", "annotations_iscrowd",
            "annotations_num_keypoints", "licenses_id", "categories_id"
        ]:
            self.df_json[x] = self.df_json[x].astype(int)
        # classes for drawing
        self._classes = self.df_json["categories_name"].unique()

    def organize_segmentation(self, min_point: int=1):
        """
        When using Segmentation annotations for augmentation, the
        If you have a weird annotation, you will get an error, so you need to add a function
        """
        df = []
        for i in tqdm(range(self.df_json.shape[0])):
            se = self.df_json.iloc[i].copy()
            polygons = []
            assert check_type_list(se["annotations_segmentation"], list, [int, float])
            for polygon in se["annotations_segmentation"]:
                if len(polygon) < min_point: continue
                img = np.zeros((int(se["images_height"]), int(se["images_width"]), 3), dtype=np.uint8)
                # Draw segmentation by connecting lines.
                img = cv2.polylines(img, [np.array(polygon).reshape(-1,1,2).astype(np.int32)], True, (255,255,255))
                # Draw a line and then get the outermost contour
                contours, _ = cv2.findContours(img[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                polygon = contours[0].reshape(-1).astype(int).tolist()
                if len(polygon) < min_point: continue
                polygons.append(polygon)
            se["annotations_segmentation"] = polygons
            df.append(se)
        self.df_json = pd.concat(df, axis=1).T
        self.df_json = self.df_json.loc[self.df_json["annotations_segmentation"].apply(lambda x: len(x)) > 0]
        self.re_index()

    def add_json(self, src: Union[str, dict], root_dir: str=None, is_add_df: bool=True):
        """
        add coco file.
        Params::
            src:
                coco file path or dictionary with coco format.
            root_dir:
                directory path name.
                Replace coco url if root_dir is specified.
        """
        assert check_type(src, [str, dict])
        json_coco = None
        if   type(src) == str:  json_coco = json.load(open(src))
        elif type(src) == dict: json_coco = src
        self.json = json_coco.copy()
        if json_coco.get("licenses") is None or len(json_coco["licenses"]) == 0:
            json_coco["licenses"] = [{'url': COCO_INFO_TMP_URL, 'id': 0, 'name': COCO_INFO_TMP_LICENSE_NAME}]
        try:
            self.coco_info = json_coco["info"]
        except KeyError:
            logger.warning("'src' file or dictionary is not found 'info' key. so, skip this src.")
            self.coco_info = coco_info()
        df = self.coco_json_to_df(json_coco)
        if root_dir is not None:
            df["images_coco_url"] = correct_dirpath(root_dir) + df["images_file_name"]
        if is_add_df:
            self.df_json = pd.concat([self.df_json, df], axis=0, ignore_index=True, sort=False)
            self.check_index()
            self.re_index()
        return df
    
    def add_jsons(self, paths: List[str], root_dir: str=None):
        assert check_type_list(paths, str)
        assert root_dir is None or isinstance(root_dir, str)
        list_df = [self.add_json(x, root_dir=root_dir, is_add_df=False) for x in paths]
        self.df_json = pd.concat(list_df, axis=0, ignore_index=True, sort=False)
        self.check_index()
        self.re_index()
    
    def remove_annotations_not_files(self, root_dir: str=None):
        """
        Delete records that do not have images_coco_url.
        Params::
            root_dir:
                If root_dir is specified, then check rot_dir + images_file_name.
        """
        assert root_dir is None or isinstance(root_dir, str)
        list_target = []
        if root_dir is None:
            for x in self.df_json["images_coco_url"].unique():
                if os.path.exists(x):
                    list_target.append(x)
        else:
            root_dir = correct_dirpath(root_dir)
            for x in self.df_json["images_file_name"].unique():
                if os.path.exists(root_dir + x):
                    list_target.append(x)
        self.df_json = self.df_json.loc[self.df_json["images_file_name"].isin(list_target), :]
    
    def remove_keypoints(self, list_key_names: List[str], list_key_skeleton:List[List[str]]):
        """
        Params::
            list_key_names: list of keypoints' names you want to remain
        """
        df = self.df_json.copy()
        df["annotations_keypoints"] = df[["annotations_keypoints","categories_keypoints"]].apply(
            lambda x: np.array(x["annotations_keypoints"]).reshape(-1, 3)[
                np.isin(np.array(x["categories_keypoints"]), list_key_names)
            ].reshape(-1).tolist(), axis=1
        ).copy()
        df["annotations_num_keypoints"] = df["annotations_keypoints"].apply(lambda x: (np.array(x).reshape(-1, 3)[::, 2] > 0).sum())
        df["categories_keypoints"]      = df["categories_keypoints" ].apply(lambda x: np.array(x)[np.isin(np.array(x), list_key_names)].tolist()).copy()
        df["categories_skeleton"]       = df["categories_keypoints" ].apply(lambda x: [np.where(np.isin(np.array(x), _x))[0].tolist() for _x in list_key_skeleton])
        self.df_json = df.copy()

    def concat_segmentation(self, category_name: str):
        """
        Combine the segmentation of the class specified in category_name by images_coco_url.
        """
        df = self.df_json.copy()
        df = df[df["categories_name"] == category_name]
        df_rep = pd.DataFrame()
        for _, dfwk in df.groupby("images_coco_url"):
            se = dfwk.iloc[0].copy()
            list_seg = []
            for segs in dfwk["annotations_segmentation"].values:
                for seg in segs:
                    list_seg.append([int(x) for x in seg])
            se["annotations_segmentation"] = list_seg
            bbox = np.concatenate(list_seg).reshape(-1, 2)
            se["annotations_bbox"] = [bbox[:, 0].min(), bbox[:, 1].min(), bbox[:, 0].max() - bbox[:, 0].min(), bbox[:, 1].max() - bbox[:, 1].min()]
            se["annotations_bbox"] = [int(x) for x in se["annotations_bbox"]]
            se["annotations_area"] = int(dfwk["annotations_area"].sum())
            df_rep = df_rep.append(se, ignore_index=True, sort=False)
        df = self.df_json.copy()
        df = df[df["categories_name"] != category_name]
        df = pd.concat([df, df_rep], axis=0, ignore_index=True, sort=False)
        self.df_json = df.copy()
        self.re_index()
    
    def copy_annotation(self, catname_copy_from: str, catname_copy_to: str, colnames: List[str]):
        """
        images_coco_url Copy the unit category name XXX to another category name XXX.
        """
        df = self.df_json.copy()
        df_to   = df[df["categories_name"] == catname_copy_to  ].copy()
        df_from = df[df["categories_name"] == catname_copy_from].groupby("images_coco_url").first() # if image has some instance, get first annotation.
        df_from.columns = [x + "_from" for x in df_from.columns]
        df_from = df_from.reset_index()
        for colname in colnames:
            df_to = pd.merge(df_to, df_from[["images_coco_url", colname+"_from"]].copy(), how="left", on="images_coco_url")
            df_to[colname] = df_to[colname + "_from"].copy()
            if df_to[colname].isna().sum() > 0: raise Exception(f'{colname} has nan')
            df_to = df_to.drop(columns=[colname + "_from"])
        df = df[df["categories_name"] != catname_copy_to]
        df = pd.concat([df, df_to], axis=0, ignore_index=True, sort=False)
        self.df_json = df.copy()
        self.re_index()

    @classmethod
    def __to_str_coco_format(cls, df_json: pd.DataFrame) -> str:
        assert isinstance(df_json, pd.DataFrame)
        json_dict = {"info": coco_info()}
        for _name in ["images", "annotations", "licenses", "categories"]:
            df = df_json.loc[:, df_json.columns.str.contains("^"+_name+"_", regex=True)].copy()
            df.columns = df.columns.str[len(_name+"_"):]
            df = df.fillna("%%null%%")
            json_dict[_name] = df.groupby("id").first().reset_index().apply(lambda x: x.to_dict(), axis=1).to_list()
        strjson = json.dumps(json_dict)
        return strjson.replace('"%%null%%"', 'null')

    def to_str_coco_format(self) -> str:
        return self.__to_str_coco_format(self.df_json.copy())

    def save(self, filepath: str, save_images_path: str=None, exist_ok: bool=False, remake: bool=False):
        """
        save coco format json file.
        Params::
            filepath: coco file path.
            save_images_path:
                if not None, copy images images_coco_url path file.

        """
        assert isinstance(filepath, str)
        assert save_images_path is None or isinstance(save_images_path, str)
        if save_images_path is not None:
            save_images_path = correct_dirpath(save_images_path)
            makedirs(save_images_path, exist_ok=exist_ok, remake=remake)
            df = self.df_json.groupby(["images_file_name","images_coco_url"]).size().reset_index()
            df["images_file_name"] = df.groupby("images_file_name")["images_file_name"].apply(
                lambda x: pd.Series([os.path.basename(y)+"."+str(i)+"."+y.split(".")[-1] for i, y in enumerate(x)]) if x.shape[0] > 1 else x
            ).values.copy()
            self.df_json["images_file_name"] = self.df_json["images_coco_url"].map(df.set_index("images_coco_url")["images_file_name"].to_dict())
            for x, y in df[["images_coco_url", "images_file_name"]].values:
                shutil.copy2(x, save_images_path+y)
            self.df_json["images_coco_url"] = save_images_path + self.df_json["images_file_name"]
            self.re_index()
        with open(filepath, "w") as f:
            f.write(self.to_str_coco_format())
    
    def draw_annotations(
        self, src: Union[int, str], imgpath: str=None, is_draw_name: bool=False, is_show: bool=True, save_path: str=None, resize: int=None,
    ) -> np.ndarray:
        assert check_type(src, [int, str])
        assert save_path is None or isinstance(save_path, str)
        assert resize is None or isinstance(resize, int) and resize > 100
        df = None
        if   isinstance(src, int): df = self[src]
        elif isinstance(src, str): df = self.df_json[self.df_json["images_file_name"] == src].copy()
        imgpath = df["images_coco_url"].iloc[0] if imgpath is None else correct_dirpath(imgpath) + df["images_file_name"].iloc[0]
        img     = cv2.imread(imgpath)
        if img is None:
            logger.raise_error(f"img file: {imgpath} is not exist.")
        for i in np.arange(df.shape[0]):
            se  = df.iloc[i]
            if hasattr(self, "_classes"):
                color_id = int(np.where(self._classes == se["categories_name"])[0][0])
            else:
                color_id = None
            img = draw_annotation(
                img, bbox=se["annotations_bbox"], 
                catecory_name=(se['categories_name'] if is_draw_name else None),
                segmentations=se["annotations_segmentation"],
                keypoints=se["annotations_keypoints"],
                keypoints_name=(se['categories_keypoints'] if is_draw_name else None),
                keypoints_skeleton=(se['categories_skeleton'] if is_draw_name else None),
                color_id=color_id
            )
        if resize is not None:
            img = fit_resize(img, "y", resize)
        if is_show:
            cv2.imshow("sample", img)
            cv2.waitKey(0)
        if save_path is not None:
            cv2.imwrite(save_path, img)
        return img
    
    def output_labelme_files(self, outdir: str, root_dir: str=None, exist_ok: bool=False, remake: bool=False, version: str="4.5.10"):
        outdir = correct_dirpath(outdir)
        if root_dir is not None: root_dir = correct_dirpath(root_dir)
        makedirs(outdir, exist_ok=exist_ok, remake=remake)
        for i in range(len(self)):
            df = self[i]
            out_filename = ".".join(df["images_file_name"].iloc[0].split(".")[:-1]) + ".json"
            path_image   = df["images_coco_url"].iloc[0] if root_dir is None else root_dir + df["images_file_name"].iloc[0]
            dict_labelme = {
                "version": version, "flags": {}, "shapes": [],
                "imagePath": path_image, "imageData": None,
                "imageHeight": int(df["images_height"].iloc[0]),
                "imageWidth" : int(df["images_width"].iloc[0]),
            }
            for j in range(df.shape[0]):
                se = df.iloc[j]
                dictwk = {"label": str(se["categories_name"]), "group_id": None, "flags": {}}
                if len(se["annotations_segmentation"]) > 0:
                    # segmentation
                    dictwk["shape_type"] = "polygon"
                    dictwk["points"] = np.array(se["annotations_segmentation"]).reshape(-1, 2).astype(np.int32).tolist()
                else:
                    # bbox
                    x_min, y_min, width, height = [int(x) for x in se["annotations_bbox"]]
                    dictwk["shape_type"] = "rectangle"
                    dictwk["points"] = [[x_min, y_min], [x_min + width, y_min + height]]
                dict_labelme["shapes"].append(dictwk)
                if len(se["categories_keypoints"]) > 0:
                    # keypoint
                    ndf = np.array(se["annotations_keypoints"]).reshape(-1, 3)
                    for id_key, (x, y, v) in enumerate(ndf):
                        if v > 0:
                            dictwk = {"label": str(se["annotations_keypoints"][id_key]), "group_id": None, "flags": {}}
                            dictwk["shape_type"] = "point"
                            dictwk["points"] = [x, y]
                            dict_labelme["shapes"].append(dictwk)
            with open(outdir + out_filename, 'w') as f:
                json.dump(dict_labelme, f, indent=4)

    def save_draw_annotations(self, outdir: str, imgpath: str=None, is_draw_name: bool=False, exist_ok: bool=False, remake: bool=False):
        outdir = correct_dirpath(outdir)
        makedirs(outdir, exist_ok=exist_ok, remake=remake)
        for i in range(len(self)):
            img   = self.draw_annotations(i, imgpath=imgpath, is_draw_name=is_draw_name, is_show=False)
            fname = self[i]["images_file_name"].iloc[0]
            cv2.imwrite(outdir + fname, img)
    
    def scale_bbox(self, padding: Union[int, float, List[int], List[float]], scale_by_name: Union[str, List[str]]=None):
        if isinstance(padding, int):   padding = [padding, ] * 4
        if isinstance(padding, float): padding = [padding, ] * 4
        if isinstance(scale_by_name, str): scale_by_name = [scale_by_name, ]
        assert scale_by_name is None or check_type_list(scale_by_name, str)
        assert check_type_list(padding, int) or check_type_list(padding, float)
        assert len(padding) == 4
        df_json = []
        for i in range(self.df_json.shape[0]):
            se = self.df_json.iloc[i].copy()
            if scale_by_name is not None and se["categories_name"] in scale_by_name:
                df_json.append(se)
                continue
            # bbox
            img_h,  img_w  = se["images_height"], se["images_width"]
            x1, y1,  w,  h = se["annotations_bbox"]
            x1, y1, x2, y2 = x1, y1, x1 + w, y1 + h
            diff_x1, diff_y1, diff_x2, diff_y2 = padding
            if isinstance(diff_x1, float):
                diff_x1 = w * (diff_x1 - 1.0)
                diff_y1 = h * (diff_y1 - 1.0)
                diff_x2 = w * (diff_x2 - 1.0)
                diff_y2 = h * (diff_y2 - 1.0)
            x1, y1, x2, y2 = x1 - diff_x1, y1 - diff_y1, x2 + diff_x2, y2 + diff_y2
            if x1 < 0:     x1 = 0
            if x1 > img_w: x1 = img_w
            if x2 < 0:     x2 = 0
            if x2 > img_w: x2 = img_w
            if y1 < 0:     y1 = 0
            if y1 > img_h: y1 = img_h
            if y2 < 0:     y2 = 0
            if y2 > img_h: y2 = img_h
            se["annotations_bbox"] = [x1, y1, x2 - x1, y2 - y1]
            df_json.append(se)
        df_json = pd.concat(df_json, axis=1).T
        self.df_json = df_json
        self.re_index()

    def crop_bbox(self, outdir: str, crop_by_name: Union[str, List[str]]=None, exist_ok: bool=False, remake: bool=False, remove_size: int=None):
        assert isinstance(outdir, str)
        if isinstance(crop_by_name, str): crop_by_name = [crop_by_name, ]
        assert crop_by_name is None or check_type_list(crop_by_name, str)
        assert remove_size is None or (isinstance(remove_size, int) and remove_size > 0)
        df      = self.df_json.copy()
        if crop_by_name is not None:
            df_crop = df.loc[(df["categories_name"].isin(crop_by_name))].copy()
        else:
            df_crop = df.copy()
        outdir  = correct_dirpath(outdir)
        makedirs(outdir, exist_ok=exist_ok, remake=remake)
        df_json = []
        for image_path, (crop_x, crop_y, crop_w, crop_h) in df_crop[["images_coco_url", "annotations_bbox"]].values:
            crop_x, crop_y, crop_w, crop_h = int(crop_x), int(crop_y), int(crop_w), int(crop_h)
            images_file_name = f"{os.path.basename(image_path)}.{crop_x}_{crop_y}_{crop_w}_{crop_h}.png"
            images_coco_url  = outdir + images_file_name
            df_ann = df.loc[(df["images_coco_url"] == image_path)].copy()
            # image
            img = cv2.imread(image_path)
            img = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w, :]
            cv2.imwrite(images_coco_url, img)
            for i_ann in range(df_ann.shape[0]):
                # annotations
                se = df_ann.iloc[i_ann].copy()
                se["images_coco_url"]  = images_coco_url
                se["images_file_name"] = images_file_name
                se["images_height"]    = crop_h
                se["images_width"]     = crop_w
                # bbox
                x1, y1,  w,  h = se["annotations_bbox"]
                x1, y1, x2, y2 = x1, y1, x1 + w, y1 + h
                x1, y1, x2, y2 = x1 - crop_x, y1 - crop_y, x2 - crop_x, y2 - crop_y
                if x1 < 0:      x1 = 0
                if x1 > crop_w: x1 = crop_w
                if x2 < 0:      x2 = 0
                if x2 > crop_w: x2 = crop_w
                if y1 < 0:      y1 = 0
                if y1 > crop_h: y1 = crop_h
                if y2 < 0:      y2 = 0
                if y2 > crop_h: y2 = crop_h
                se["annotations_bbox"] = [x1, y1, x2 - x1, y2 - y1]
                if len(se["annotations_keypoints"]) > 0:
                    # keypoint
                    ndf_kpt = np.array(se["annotations_keypoints"])
                    ndf_kpt = ndf_kpt.reshape(-1, 3)
                    ndf_kpt[:, 0] -= crop_x
                    ndf_kpt[:, 1] -= crop_y
                    ndf_kpt[(ndf_kpt[:, 0] < 0     ), 2] = 0 # point with out of window, vis to 0
                    ndf_kpt[(ndf_kpt[:, 0] > crop_w), 2] = 0 # point with out of window, vis to 0
                    ndf_kpt[(ndf_kpt[:, 1] < 0     ), 2] = 0 # point with out of window, vis to 0
                    ndf_kpt[(ndf_kpt[:, 1] > crop_h), 2] = 0 # point with out of window, vis to 0
                    ndf_kpt[(ndf_kpt[:, 2] == 0), 0]     = 0 # vis == 0, point to 0
                    ndf_kpt[(ndf_kpt[:, 2] == 0), 1]     = 0 # vis == 0, point to 0
                    se["annotations_keypoints"] = [float(x) for x in ndf_kpt.reshape(-1)]
                if len(se["annotations_segmentation"]) > 0:
                    # segmentation
                    segs = []
                    for seg in se["annotations_segmentation"]:
                        assert len(seg) > 0
                        ndf_seg = np.array(seg).reshape(-1, 2)
                        ndf_seg[:, 0] -= crop_x
                        ndf_seg[:, 1] -= crop_y
                        ndf_seg[(ndf_seg[:, 0] < 0     ), 0] = 0
                        ndf_seg[(ndf_seg[:, 0] > crop_w), 0] = crop_w
                        ndf_seg[(ndf_seg[:, 1] < 0     ), 1] = 0
                        ndf_seg[(ndf_seg[:, 1] > crop_h), 1] = crop_h
                        segs.append([int(x) for x in ndf_seg.reshape(-1)])
                    se["annotations_segmentation"] = segs
                df_json.append(se)
        df_json = pd.concat(df_json, axis=1).T
        if remove_size is not None:
            # remove small bbox
            ndf     = np.stack(df_json["annotations_bbox"].values)
            boolwk  = ((ndf[:, 2] <= remove_size) | (ndf[:, 3] <= remove_size))
            df_json = df_json.loc[~boolwk]
        coco    = __class__()
        coco.df_json = df_json
        coco.re_index()
        coco.save(outdir + "/coco.json")
        return coco

    def split_validation_data(self, path_json_train: str, path_json_valid: str, size: float=0.1):
        df = self.df_json.copy()
        ndf_fname = df["images_file_name"].unique()
        ndf_fname = np.random.permutation(ndf_fname)
        size = int(ndf_fname.shape[0] * size)
        df_train = df[df["images_file_name"].isin(ndf_fname[:-size ])].copy()
        df_valid = df[df["images_file_name"].isin(ndf_fname[ -size:])].copy()
        # train
        self.df_json = df_train
        self.re_index()
        self.save(path_json_train)
        # train
        self.df_json = df_valid
        self.re_index()
        self.save(path_json_valid)
        # back it previous df
        self.df_json = df
