"""This script is the data preparation script for Deep3DFaceRecon_pytorch
"""

import json
import os
import glob
import numpy as np
import argparse
from tqdm import tqdm
from util.detect_lm68 import detect_68p,load_lm_graph
from util.skin_mask import get_skin_mask
from util.generate_list import check_list, write_list
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='datasets', help='root directory for training data')
parser.add_argument('--img_folder', nargs="+", required=True, help='folders of training images')
parser.add_argument('--mode', type=str, default='train', help='train or val')
parser.add_argument('--split_file', type=str,
                    help="json file that lists all images to be considered")
parser.add_argument('--part_num', type=int, help="how many partitions to use")
parser.add_argument('--part_id', type=int, help="partition id")
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def data_prepare(folder_list,mode,split_file=None,part_num=None,part_id=None):

    lm_sess,input_op,output_op = load_lm_graph('./checkpoints/lm_model/68lm_detector.pb') # load a tensorflow version 68-landmark detector

    if split_file is not None:
        with open(split_file) as f:
            names_dict = json.load(f)
        assert len(folder_list) == 1
        names_dict = names_dict[mode]
        img_folders = list(names_dict.keys())
    else:
        img_folders = folder_list

    if part_num is not None and part_id is not None:
        #partition the folders into 10 parts
        img_folders = list(np.array_split(img_folders, part_num)[part_id])

    for img_folder in tqdm(img_folders):
        if split_file is not None:
            img_names = names_dict[img_folder]
            img_folder = os.path.join(folder_list[0], img_folder)
        else:
            img_names = [i for i in sorted(os.listdir(
                img_folder)) if 'jpg' in i or 'png' in i or 'jpeg' in i or 'PNG' in i]

        detect_68p(img_folder,img_names,lm_sess,input_op,output_op) # detect landmarks for images
        get_skin_mask(img_folder,img_names) # generate skin attention mask for images

    # create files that record path to all training data
    msks_list = []
    for img_folder in tqdm(img_folders):
        if split_file is not None:
            img_folder = os.path.join(folder_list[0], img_folder)
        path = os.path.join(img_folder, 'mask')
        msks_list += ['/'.join([img_folder, 'mask', i]) for i in sorted(os.listdir(path)) if 'jpg' in i or
                                                    'png' in i or 'jpeg' in i or 'PNG' in i]

    imgs_list = [i.replace('mask/', '') for i in msks_list]
    lms_list = [i.replace('mask', 'landmarks') for i in msks_list]
    lms_list = ['.'.join(i.split('.')[:-1]) + '.txt' for i in lms_list]

    lms_list_final, imgs_list_final, msks_list_final = \
        check_list(lms_list, imgs_list, msks_list) # check if the path is valid
    print(f"{len(lms_list_final)}/{len(lms_list)} processed files survived.")
    save_name = '' if part_id is None else f"{part_id:02d}_"
    write_list(lms_list_final, imgs_list_final, msks_list_final,
               mode=mode, save_name=save_name) # save files

if __name__ == '__main__':
    print('Datasets:',opt.img_folder)
    data_prepare([os.path.join(opt.data_root,folder) for folder in opt.img_folder],
                 opt.mode,opt.split_file,opt.part_num,opt.part_id)
