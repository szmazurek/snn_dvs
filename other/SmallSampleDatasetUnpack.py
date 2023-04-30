import numpy as np
from apng import APNG
import os
from os import listdir
from os.path import isfile, join
import csv
import shutil

def _str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def _load_csv(_path_csv):
    _dataset_folders = []
    _labels_dic = {}
    with open(_path_csv, newline='') as csvfile:
        file = csv.reader(csvfile, delimiter=',')
        next(file)
        for row in file:
            if not row[0] in _labels_dic:
                _labels_dic[row[0]] = []
            _labels_dic[row[0]].append(row[-1])
    return _labels_dic


def _load_id_filtered(_path_csv):
    _dataset_folders = []
    _dic_id = []
    with open(_path_csv, newline='') as csvfile:
        file = csv.reader(csvfile, delimiter=',')
        next(file)
        for row in file:
            if not row[0] in _dic_id:
                _dic_id.append(row[0])
    return _dic_id


def create_csv_simple(_path_old_csv, _path_new_csv):
    _labels_dic = _load_csv(_path_old_csv)
    with open(_path_new_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(("id", "frame.idx", "frame.pedestrian.is_crossing"))
        for key, values in _labels_dic.items():
            i = 0
            for v in values:
                writer.writerow((key, i, v))
                i += 1
    file.close()


def create_csv_simple_filtered(_path_old_csv, _path_new_csv, _csv_of_filtered_keys):
    _labels_dic = _load_csv(_path_old_csv)
    _dic_id = _load_id_filtered(_csv_of_filtered_keys)
    with open(_path_new_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(("id", "frame.idx", "frame.pedestrian.is_crossing"))
        for key, values in _labels_dic.items():
            if key in _dic_id:
                i = 0
                for v in values:
                    writer.writerow((key, i, v))
                    i += 1
    file.close()


def upack_apng_images(_path_clips=r"D:\datasets\big-sample\clips",
                      _type_of_images="1", _path_output=r"D:\datasets\big-sample\segAll",
                      _path_csv=r"D:\datasets\big-sample\data.csv"):
    _label = _load_csv(_path_csv)

    for key in _label:
        _label[key] = list(filter(lambda x: x != "", _label[key]))

    print(len(_label["8a7905f2-6777-4ec7-afff-038a49e1bce3-0"]))
    n = 0
    _raw_files = [f for f in listdir(_path_clips) if isfile(join(_path_clips, f)) and "apng" in f]
    _no_extension_files = []
    for _raw_file in _raw_files:
        _no_extension_files.append(_raw_file[:-5])

    for i in range(len(_raw_files)):

        try:
            _label_list = _label[_no_extension_files[i][:-2]]
        except:
            #print(_no_extension_files[i][:-2])
            continue

        if _no_extension_files[i][-1] == _type_of_images:
            continue

        if not os.path.exists(os.path.join(_path_output, _no_extension_files[i][:-2])):
            os.mkdir(os.path.join(_path_output, _no_extension_files[i][:-2]))

            n = 0

        im = APNG.open(os.path.join(_path_clips, _raw_files[i]))
        print(_no_extension_files[i])
        for (png, control), label in zip(im.frames, _label_list):
            label = int(_str2bool(label))
            png.save(
                os.path.join(_path_output, _no_extension_files[i][:-2], "{n}-{label}.png".format(n=n, label=label)))
            n += 1
        print(n)
        #print(i)


def delete_duplicated_folders(path_to_delete_duplicated_folders, path_to_template):
    _folders_to_clear = listdir(path_to_delete_duplicated_folders)
    _folders_template = listdir(path_to_template)
    print("There was:", len(_folders_to_clear), "folders")

    for f in _folders_template:
        try:
            shutil.rmtree(os.path.join(path_to_delete_duplicated_folders, f))
        except:
            pass
    print("Now, there is:", len(listdir(path_to_delete_duplicated_folders)), "folders")

def create_new_csv_based_on_folders(path_to_folder, path_to_base_csv, path_to_new_csv):
    _folders_list = listdir(path_to_folder)
    _labels_dic = _load_csv(path_to_base_csv)

    with open(path_to_new_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(("id", "frame.idx", "frame.pedestrian.is_crossing"))
        for key, values in _labels_dic.items():
            if key in _folders_list:
                i = 0
                for v in values:
                    writer.writerow((key, i, v))
                    i += 1

#  type_of_images: "1"-dvs or "2"-seg
#  path_clips = r"D:\ProjectsPython\SpikingJelly\small-sample\clips"
#  path_output = r"D:\datasets\test"
#  path_csv = r"D:\ProjectsPython\SpikingJelly\small-sample/data.csv"


#path_clips = r"D:\datasets\small-sample\clips"
path_clips = r"D:\datasets\huge-sample\clips"
#path_csv = r"D:\datasets\big-sample\predictionDataset\filtered_data_small.csv"
path_csv = r"D:\datasets\huge-sample\data.csv"
path_csv_new = r"D:\datasets\huge-sample\data1.csv"


path_output = r"D:\datasets\huge-sample\ach"
path_dir = r"D:\datasets\big-sample\segAll"
upack_apng_images(_type_of_images="1", _path_clips=path_clips, _path_csv=path_csv_new, _path_output=path_output)
#create_csv_simple(_path_old_csv=r"D:\datasets\big-sample\data.csv", _path_new_csv=r"D:\datasets\big-sample\new.csv")
#create_csv_simple_filtered(_path_old_csv=r"D:\datasets\small-sample\data.csv", _path_new_csv=r"D:\datasets\big-sample\newfiltersmall.csv",_csv_of_filtered_keys=r"D:\datasets\big-sample\predictionDataset\id_good_examples.csv")
#delete_duplicated_folders(path_output,path_dir)

#create_new_csv_based_on_folders(path_output, path_csv, path_csv_new)

# 8a7905f2-6777-4ec7-afff-038a49e1bce3-0-2