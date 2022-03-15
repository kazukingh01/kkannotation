from kkannotation.coco import CocoManager


if __name__ == "__main__":
    coco = CocoManager()
    coco.add_json("./coco.json", root_dir="../img/")
    coco.draw_annotations(0, is_draw_name=True, is_show=True)
    """
    >>> coco[0]
    images_id        images_coco_url images_date_captured images_file_name  ...                categories_keypoints  categories_name  categories_skeleton  categories_supercategory
    0          0  ./img/img_dog_cat.jpg  2021-09-21 13:09:36  img_dog_cat.jpg  ...  [eye_left, eye_right, nose, mouth]              dog                   []                       dog
    1          0  ./img/img_dog_cat.jpg  2021-09-21 13:09:36  img_dog_cat.jpg  ...  [eye_left, eye_right, nose, mouth]              cat                   []                       cat
    >>> coco.df_json.iloc[0]
    images_id                                                                    0
    images_coco_url                                          ./img/img_dog_cat.jpg
    images_date_captured                                       2021-09-21 13:09:36
    images_file_name                                               img_dog_cat.jpg
    images_flickr_url                                                  http://test
    images_height                                                              398
    images_license                                                               0
    images_width                                                               710
    annotations_id                                                               0
    annotations_area                                                       39059.5
    annotations_bbox                                           [119, 83, 238, 269]
    annotations_category_id                                                      1
    annotations_image_id                                                         0
    annotations_iscrowd                                                          0
    annotations_keypoints        [0.0, 0.0, 0.0, 288.57055214723925, 124.766871...
    annotations_num_keypoints                                                    4
    annotations_segmentation     [[200, 132, 240, 93, 264, 84, 296, 83, 321, 95...
    licenses_id                                                                  0
    licenses_name                                                     test license
    licenses_url                                                       http://test
    categories_id                                                                1
    categories_keypoints                        [eye_left, eye_right, nose, mouth]
    categories_name                                                            dog
    categories_skeleton                                                         []
    categories_supercategory                                                   dog
    Name: 0, dtype: object
    """
