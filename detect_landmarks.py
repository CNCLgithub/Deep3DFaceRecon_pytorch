import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import glob
from tqdm import tqdm
from options.test_options import TestOptions
from facenet_pytorch import MTCNN
from PIL import ImageDraw
from util.image_folder import ImageFolderWithPaths


def output_pth(pth):
    # grab the relative path
    base, f = os.path.split(pth)
    out_dir = os.path.join(base, "detections")
    out_pth = os.path.splitext(os.path.join(out_dir, f))[0]
    return out_dir, out_pth


def draw_landmarks(img, landmarks, r=8, color=(255, 0, 0)):
    draw = ImageDraw.Draw(img)
    for i, landmark in enumerate(landmarks.astype(int)):
        x, y = landmark[0], landmark[1]
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, width=5)
    return img


def main(opt, ds_folder="custom_images", make_overlays=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load data
    ds = ImageFolderWithPaths(ds_folder,
                              transform=transforms.PILToTensor())
    dataloader = DataLoader(ds, num_workers=2, batch_size=1)

    # create face/landmark detector
    detector = MTCNN(device=device)

    for img, _, img_pth in tqdm(dataloader):
        out_dir, out_pth = output_pth(img_pth[0])

        if os.path.isfile(out_pth+".txt"):
            continue

        os.makedirs(out_dir, exist_ok=True)

        img = img.squeeze(0)
        _, _, keypoints = detector.detect(
            img.permute(1, 2, 0).to(device), landmarks=True
        )

        if keypoints is None:
            continue

        keypoints = keypoints[0]
        np.savetxt(out_pth+".txt", keypoints, fmt="%10.5f")

        if make_overlays:
            img = transforms.functional.to_pil_image(img.cpu())
            img = draw_landmarks(img, keypoints)
            img.save(out_pth + ".png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    opt = TestOptions().initialize(parser)
    opt.add_argument(
        "--make_overlays", action="store_true", help="whether to save annotated images"
    )
    opt = opt.parse_args()
    main(opt, opt.img_folder, opt.make_overlays)
