import json
import numpy as np
import cv2
from kkannotation.symbolemb import SymbolEmbedding


if __name__ == "__main__":
    with open("config.json") as f: config = json.load(f)
    emb = SymbolEmbedding(**config)
    img = emb.create_image()
    cv2.imshow("test", img)
    cv2.waitKey(0)

