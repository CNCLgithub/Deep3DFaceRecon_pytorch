import argparse
import pandas as pd
import os
import numpy as np
import os.path as osp
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader, DatasetFolder
from torchvision import transforms
import torchvision.transforms.functional as TF
from options.test_options import TestOptions
from models import create_model
from util.visualizer import MyVisualizer
from util.load_mats import load_lm3d
from util.preprocess import align_img


LAYERS_DEF = [
    ["id"],
    ["tex"],
    ["exp"],
    ["id", "exp"],
    ["id", "tex"],
    ["exp", "tex"],
    ["id", "exp", "tex"],
]

class Rotation:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)

class ImageFolderFX(DatasetFolder):
    def __init__(self, root, lm3d_std, transform = None,
                 target_transform = None,
                 loader = default_loader,
                 is_valid_file = None,
                 inverted = False,
                 lm_transform = None):
        super().__init__(root, loader,
            (".jpg",) if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.lm_transform = lm_transform
        self.inverted = inverted
        self.lm3d_std = lm3d_std

    def __getitem__(self, index):
        img_path, target = self.samples[index]

        img_name = osp.basename(img_path)
        lm_name = img_name.replace(".jpg", ".txt")
        lm_path = osp.join(osp.dirname(img_path), "detections", lm_name)

        img = self.loader(img_path)
        W,H = img.size
        lm = np.loadtxt(lm_path).astype(np.float32).reshape([-1,2])
        lm[:, -1] = H - 1 - lm[:, -1]

        _, img, lm, _ = align_img(img, lm, self.lm3d_std)

        if self.inverted:
            # rotate landmarks
            lm[:,0] = W - 1 - lm[:,0]
            lm[:,1] = H - 1 - lm[:,1]
            # rotate image
            img = Rotation(180*int(self.inverted))(img)

        if self.transform is not None:
            img = self.transform(img)
            if self.lm_transform is not None:
                lm = self.lm_transform(lm)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target, img_name, lm

def set_arch_params(opt):
    """
    Read the architecture from the train_opt.txt
    """
    expr_dir = osp.join(opt.checkpoints_dir, opt.name)
    file_name = osp.join(expr_dir, "train_opt.txt")
    with open(file_name, "r") as f:
        options = [word for word in f.read().split()]
    override_keys = ["net_recon", "use_last_fc"]
    for key in override_keys:
        setattr(opt, key, options[options.index(key+":")+1])

def main(rank, opt, img_folder):
    set_arch_params(opt)

    # model
    device = torch.device(rank)
    torch.cuda.set_device(device)
    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()

    # visualizer
    visualizer = MyVisualizer(opt)

    # data
    lm3d_std = load_lm3d(opt.bfm_folder)
    transform = transforms.ToTensor()

    results = []

    for inverted in [False, True]:
        ds = ImageFolderFX(opt.img_folder, lm3d_std,
                        inverted=inverted,
                        transform=transform,
                        lm_transform=transform)
        ds_name = osp.basename(osp.normpath(opt.img_folder))
        ds_suffix = "-inverted" if inverted else ""
        dataloader = DataLoader(ds, batch_size=64, pin_memory=True)

        # extract features
        for i, (imgs, labels, img_names, lms) in enumerate(tqdm(dataloader)):
            model.set_input({"imgs": imgs, "lms": lms})
            with torch.no_grad():
                model.test()
            visuals = model.get_current_visuals()
            visualizer.display_current_results(
                visuals, 0, opt.epoch, dataset=ds_name+ds_suffix,
                save_results=True, count=i, add_image=False)
            coeffs_dict = {key: value.cpu().numpy()
                        for key, value in model.pred_coeffs_dict.items()}

            for layer_def in LAYERS_DEF:
                layer_name = "+".join(layer_def)
                activations = []
                for layer in layer_def:
                    activations.append(coeffs_dict[layer])
                activations = np.concatenate(activations, 1)
                results.append(pd.DataFrame.from_dict({
                    "gt_id": labels.numpy(),
                    "img_name": img_names,
                    "layer": layer_name,
                    "dataset": ds_name,
                    "method": "EIG",
                    "inverted": inverted,
                    "model_name": opt.name,
                    "activations": list(activations),
                }))

    df = pd.concat(results, ignore_index=True)
    df_pth = osp.join(visualizer.img_dir, f"{ds_name}.json")
    df.to_json(df_pth)

if __name__ == "__main__":
    opt = TestOptions().parse()
    main(0, opt, opt.img_folder)
