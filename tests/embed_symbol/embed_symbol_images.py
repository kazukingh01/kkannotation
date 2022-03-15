import json, argparse
import cv2
from kkannotation.symbolemb import SymbolEmbedding


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config file. ex) --config ./config_rand.json", required=True)
    parser.add_argument("--path",   type=str, help="output path. ex) --config ./test.png",         required=False, default="./test.png")
    args = parser.parse_args()

    with open(args.config, mode="r") as f:
        config = json.load(f)
    emb = SymbolEmbedding(**config)
    img, coco = emb.create_image(args.path, is_save=True)
    cv2.imshow("test", img)
    cv2.waitKey(0)
