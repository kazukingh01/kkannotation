from kkannotation.labelme import Labelme2Coco

"""
Usage labelme::
    >>> virtualenv labelme # Using python-opencv and labelme at the same time will break the environment.
    >>> source ./labelme/bin/activate
    >>> pip install labelme
    >>> labelme --output json/ img/
"""

if __name__ == "__main__":
    labelme2coco = Labelme2Coco(
        dirpath_json="./json", dirpath_img="./img",
        categories_name=["dog", "cat"],
        keypoints=["eye_left", "eye_right", "nose", "mouth"],
        keypoints_belong={
            "dog": ["eye_left", "eye_right", "nose", "mouth"],
            "cat": ["eye_left", "eye_right", "nose", "mouth"],
        },
    )
    labelme2coco.save_coco_file("coco.json")
