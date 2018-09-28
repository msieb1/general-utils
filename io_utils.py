import os
import functools
import imageio
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, TensorDataset
from torch import Tensor
from torch.autograd import Variable
import logging

import sys
import matplotlib.pyplot as plt
import pickle
from ipdb import set_trace


def distance(x1, x2):
    diff = torch.abs(x1 - x2)
    return torch.pow(diff, 2).sum(dim=1)

def view_image(frame):
    # For debugging. Shows the image
    # Input shape (3, 299, 299) float32
    img = Image.fromarray(np.transpose(frame * 255, [1, 2, 0]).astype(np.uint8))
    img.show()

def write_to_csv(values, keys, filepath):
    if  not(os.path.isfile(filepath)):
        with open(filepath, 'w', newline='') as csvfile:
            filewriter = csv.writer(csvfile)
            filewriter.writerow(keys)
            filewriter.writerow(values)
    else:
        with open(filepath, 'a', newline='') as csvfile:
            filewriter = csv.writer(csvfile)
            filewriter.writerow(values)


def ensure_folder(folder):
    path_fragments = os.path.split(folder)
    joined = '.'
    for fragment in path_fragments:
        joined = os.path.join(joined, fragment)
        if not os.path.exists(joined):
            os.mkdir(joined)

def resize_frame(frame, out_size):
    image = Image.fromarray(frame)
    image = image.resize(out_size)
    scaled = np.array(image, dtype=np.float32) / 255
    return np.transpose(scaled, [2, 0, 1])

def write_video(file_name, path, frames):
    imageio.mimwrite(os.path.join(path, file_name), frames, fps=60)

def read_video(filepath, frame_size):
    imageio_video = imageio.read(filepath)
    snap_length = len(imageio_video) 
    frames = np.zeros((snap_length, 3, *frame_size))
    resized = map(lambda frame: resize_frame(frame, frame_size), imageio_video)
    for i, frame in enumerate(resized):
        frames[i, :, :, :] = frame
    return frames

def read_extracted_video(filepath, frame_size):
    try:
        files = sorted(os.listdir(filepath), key=lambda x: int(x.split('.')[0]))
    except:
        files = sorted(os.listdir(filepath), key=lambda x: int(x.split('_')[-1].split('.')[0]))

    all_images = [file for file in files if file.endswith('.jpg')]
    snap_length = len(all_images) 
    frames = []
    for i, filename in enumerate(all_images):
        frame = plt.imread(os.path.join(filepath, filename))
        frames.append(frame)
    return frames

def read_extracted_rcnn_results(filepath, frame_size):
    try:
        files = sorted(os.listdir(filepath), key=lambda x: int(x.split('.')[0]))
    except:
        files = sorted(os.listdir(filepath), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    all_files = [file for file in files if file.endswith('.pkl')]
    snap_length = len(all_files) 
    all_results = []
    for i, filename in enumerate(all_files):
        with open(os.path.join(filepath, filename), 'rb') as fb:
            all_results.append(pickle.load(fb))    
    return all_results

def read_caption(filepath):
    try:
        with open(filepath, 'r') as fp:
            caption = fp.readline()
        return caption
    except:
        print("{} does not exist".format(filepath))
        return None

def ls_directories(path):
    return next(os.walk(path))[1]

# def ls(path):
#     # returns list of files in directory without hidden ones.
#     return sorted([p for p in os.listdir(path) if p[0] != '.' and (p[-4:] == '.mp4' or p[-4] == '.mov')], key=lambda x: int(x.split('_')[0] + x.split('.')[0].split('view')[1]))
#     # randomize retrieval for every epoch?

def ls(path):
    # returns list of files in directory without hidden ones.
    return sorted([p for p in os.listdir(path) if p[0] != '.' and (p[-4:] == '.mp4' or p[-4:] == '.mov')], key=lambda x: int(x.split('_')[0]))
    # rand

def ls_unparsed_txt(path):
    return sorted([p for p in os.listdir(path) if p[0] != '.' and p[-5] != 'd' and p.endswith('.txt')], key=lambda x: int(x.split('.')[0]))


def ls_npy(path):
    # returns list of files in directory without hidden ones.
    return sorted([p for p in os.listdir(path) if p[0] != '.' and p[-4:] == '.npy'], key=lambda x: x.split('.')[0])
    # rand

def ls_txt(path):
    return sorted([p for p in os.listdir(path) if p[0] != '.' and p.endswith('.txt')], key=lambda x: x.split('.')[0])

def ls_view(path, view):
    # Only lists video files
    return sorted([p for p in os.listdir(path) if p[0] != '.' and (p.endswith(str(view) + '.mp4'))], key=lambda x: int(x.split('_')[0]))

def ls_extracted(path):
     # returns list of folders in directory without hidden ones.
    return sorted([p for p in os.listdir(path) if (p[0] != '.' and p != 'debug') ], key=lambda x: int(x.split('_')[0]))


##### Logger #####
class Logger(object):
    def __init__(self, logfilename):
        logging.basicConfig(filename=logfilename, level=logging.DEBUG, filemode='a')

    def info(self, *arguments):
        print(*arguments)
        message = " ".join(map(repr, arguments))
        logging.info(message)




