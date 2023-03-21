from apng import APNG
import os
from os import listdir
from os.path import isfile, join
import csv


def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")


def load_csv():
    _path_csv = r"small-sample/data.csv"
    _dataset_folders = []
    _labels_dic = {}
    with open(_path_csv, newline='') as csvfile:
        file = csv.reader(csvfile, delimiter=',')
        next(file)
        for row in file:
            if not row[0] in _labels_dic:
                _labels_dic[row[0]] = []
                _dataset_folders.append(row[0])
            _labels_dic[row[0]].append(row[-1])
    return _labels_dic, _dataset_folders


def create_folder_segment_images():
    _label, _dataset = load_csv()
    type_of_images = "1"  # 1 segmentation images, 2 dvs images
    path_clips = "small-sample/clips/"
    path_output = r"/small-sample/output"

    n = 0
    rawfiles = [f for f in listdir(path_clips) if isfile(join(path_clips, f)) and "apng" in f]
    fixfiles=[]
    for file in rawfiles:
        fixfiles.append(file[:-5])

    for i in range(len(rawfiles)):
        label_list= _label[fixfiles[i][:-2]]
        if fixfiles[i][-1] == type_of_images:
            continue
        if not os.path.exists(os.path.join(path_output, fixfiles[i][:-2])):
            os.mkdir(os.path.join(path_output, fixfiles[i][:-2]))
            n = 0

        im = APNG.open(os.path.join(path_clips, rawfiles[i]))
        for (png, control), label in zip(im.frames,label_list):
            label=int(str2bool(label))
            png.save(os.path.join(path_output, fixfiles[i][:-2], "{n}-{label}.png".format(n=n, label=label)))
            n += 1

create_folder_segment_images()