import numpy as np
from apng import APNG
import os
from os import listdir
from os.path import isfile, join
import csv
import shutil
import cv2

class DataPrepeation:

    def _id_number(self,filename):
        number = filename.split("-")[0]
        return int(number)

    def _id_label(self,filename):
        number_ext = filename.split("-")[1]
        number = number_ext.split(".")[0]
        return int(number)

    def _str2bool(self, v):
        return v.lower() in ("yes", "true", "t", "1")

    def _load_csv(self, path_csv):
        labels_dic = {}
        with open(path_csv, newline='') as csvfile:
            file = csv.reader(csvfile, delimiter=',')
            next(file)
            for row in file:
                if not row[0] in labels_dic:
                    labels_dic[row[0]] = []
                labels_dic[row[0]].append(row[-1])
        return labels_dic

    def _load_id_filtered(self, path_csv):
        dic_id = []
        with open(path_csv, newline='') as csvfile:
            file = csv.reader(csvfile, delimiter=',')
            next(file)
            for row in file:
                if not row[0] in dic_id:
                    dic_id.append(row[0])
        return dic_id

    def create_csv_simple(self, path_base_csv, path_new_csv):
        labels_dic = self._load_csv(path_base_csv)
        with open(path_new_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(("id", "frame.idx", "frame.pedestrian.is_crossing"))
            for key, values in labels_dic.items():
                i = 0
                for v in values:
                    writer.writerow((key, i, v))
                    i += 1
        file.close()

    def create_csv_based_on_filtered_csv(self, path_main_csv, csv_of_filtered_keys, path_to_new_csv):
        labels_dic = self._load_csv(path_main_csv)
        dic_id = self._load_id_filtered(csv_of_filtered_keys)
        with open(path_new_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(("id", "frame.idx", "frame.pedestrian.is_crossing"))
            for key, values in labels_dic.items():
                if key in dic_id:
                    i = 0
                    for v in values:
                        writer.writerow((key, i, v))
                        i += 1
        file.close()

    def delete_duplicated_folders(self, path_to_delete_duplicated_folders, path_to_template):
        folders_to_clear = listdir(path_to_delete_duplicated_folders)
        folders_template = listdir(path_to_template)
        print("There was:", len(folders_to_clear), "folders")

        for f in folders_template:
            try:
                shutil.rmtree(os.path.join(path_to_delete_duplicated_folders, f))
            except:
                pass
        print("Now, there is:", len(listdir(path_to_delete_duplicated_folders)), "folders")

    def create_new_csv_based_on_folders(self, folder_with_filtered_dic, path_to_main_csv, path_to_new_csv):
        _folders_list = listdir(folder_with_filtered_dic)
        _labels_dic =self._load_csv(path_to_main_csv)

        with open(path_to_new_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(("id", "frame.idx", "frame.pedestrian.is_crossing"))
            for key, values in _labels_dic.items():
                if key in _folders_list:
                    i = 0
                    for v in values:
                        writer.writerow((key, i, v))
                        i += 1

    def unpack_apng(self, path_clips=r"D:\datasets\big-sample\clips",
                          type_of_images="1", path_output=r"D:\datasets\big-sample\segAll",
                          path_csv=r"D:\datasets\big-sample\data.csv"):
        label = self._load_csv(_path_csv)
        for key in label:
            label[key] = list(filter(lambda x: x != "", label[key]))

        raw_files = [f for f in listdir(path_clips) if isfile(join(path_clips, f)) and "apng" in f]
        no_extension_files = []

        for file in raw_files:
            base_name, ext = os.path.splitext(file)
            base_name_shortened = base_name[:-2]
            format_type = base_name[-1]

            if format_type == type_of_images:
                continue

            try:
                label_list = label[base_name_shortened]
            except:
                continue

            new_folder_name = os.path.join(path_output, base_name_shortened)
            if not os.path.exists(new_folder_name):
                os.mkdir(new_folder_name)

            im = APNG.open(os.path.join(path_clips, file))
            n = 0
            for (png, control), label in zip(im.frames, label_list):
                label = int(self._str2bool(label))
                png.save(
                    os.path.join(path_output, new_folder_name, f'{frame_id}-{label_int}.png'))
                n += 1

    def unpack_mp4(self, path_clips=r"D:\datasets\big-sample\clips", path_csv=r"D:\datasets\big-sample\data.csv",
                   path_output=r"D:\datasets\big-sample\segAll"):
        label = self._load_csv(path_csv)
        for key in label:
            label[key] = list(filter(lambda x: x != "", label[key]))

        files_raw = [f for f in listdir(path_clips) if isfile(join(path_clips, f)) and "mp4" in f]
        for file in files_raw:
            base_name, ext = os.path.splitext(file)
            base_name_shortened = base_name[:-2]
            try:
                label_list = label[base_name_shortened]
            except:
                continue

            new_folder_name = os.path.join(path_output, base_name_shortened)
            if not os.path.exists(new_folder_name):
                os.mkdir(new_folder_name)

            cap = cv2.VideoCapture(os.path.join(path_clips, file))

            if not cap.isOpened():
                print("Error opening video file")
            frame_id = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    label_int = int(self._str2bool(label_list[frame_id]))
                    cv2.imwrite(
                        os.path.join(path_output, new_folder_name, f'{frame_id}-{label_int}.jpg'), frame)
                    frame_id += 1
                else:
                    break

    def unpack_mp4_based_on_spike_dataset(self, path_clips, path_spike_dataset, path_output):

        files_raw = [f for f in listdir(path_clips) if isfile(join(path_clips, f)) and "mp4" in f]
        spikes_folders = [f for f in listdir(path_spike_dataset)]

        for file in files_raw:
            base_name, ext = os.path.splitext(file)
            base_name_shortened = base_name[:-2]

            new_folder_name = os.path.join(path_output, base_name_shortened)
            if not base_name_shortened in spikes_folders:
                continue

            if not os.path.exists(new_folder_name):
                os.mkdir(new_folder_name)

            spikes_files = [f for f in listdir(os.path.join(path_spike_dataset, base_name_shortened))]
            spikes_files = sorted(spikes_files, key=self._id_number)

            cap = cv2.VideoCapture(os.path.join(path_clips, file))
            if not cap.isOpened():
                print("Error opening video file")
            frame_id = 0
            while cap.isOpened():
                ret, frame = cap.read()

                if ret:
                    label_int = self._id_label(spikes_files[frame_id])
                    resized_frame= cv2.resize(frame,(400,150))
                    cv2.imwrite(
                        os.path.join(path_output, new_folder_name, f'{frame_id}-{label_int}.jpg'), resized_frame)
                    frame_id += 1
                else:
                    break

#  type_of_images: "1"-dvs or "2"-seg
#  path_clips = r"D:\ProjectsPython\SpikingJelly\small-sample\clips"
#  path_output = r"D:\datasets\test"
#  path_csv = r"D:\ProjectsPython\SpikingJelly\small-sample/data.csv"


#path_clips = r"D:\datasets\small-sample\clips"
path_clips = r"D:\datasets\huge-sample\clips"
#path_csv = r"D:\datasets\big-sample\predictionDataset\filtered_data_small.csv"
path_csv = r"D:\datasets\huge-sample\data.csv"
#path_csv_new = r"D:\datasets\huge-sample\data1.csv"

path_spike_dataset= r"D:\datasets\huge-sample\predictionDataset\dataset_prediction"

path_output = r"D:\datasets\huge-sample\mp4"
path_dir = r"D:\datasets\big-sample\segAll"
#upack_apng_images(type_of_images="1", _path_clips=path_clips, _path_csv=path_csv_new, _path_output=path_output)
d = DataPrepeation()
d.unpack_mp4_based_on_spike_dataset(path_clips, path_spike_dataset, path_output)

