# Labelme json to Coco json format

```bash
python ./labelme2coco.py
```

You can change parameters in script.
```python
labelme2coco = Labelme2Coco(
    dirpath_json="./json", dirpath_img="../img",
    categories_name=["dog", "cat"],
    keypoints=["eye_left", "eye_right", "nose", "mouth"],
    keypoints_belong={
        "dog": ["eye_left", "eye_right", "nose", "mouth"],
        "cat": ["eye_left", "eye_right", "nose", "mouth"],
    },
)
```
