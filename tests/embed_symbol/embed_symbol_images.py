import json
import numpy as np
import cv2
from kkannotation.symbolemb import SymbolEmbedding


if __name__ == "__main__":
    with open("config1.json") as f: config = json.load(f)
    emb = SymbolEmbedding(**config)
    img, coco = emb.create_image("./test.png", is_save=True)
    cv2.imshow("test", img)
    cv2.waitKey(0)
