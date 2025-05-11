import os
import shutil
from pathlib import Path
from src.utils import WORK_DIR

# #WORK_DIR = '/home/group/self_improving/experiments/mixing'
# WORK_DIR = '/groups/gcd50678/mixing'
# #WORK_DIR = 'drive/MyDrive/mixing'

def process_dataset(txt_file, downloaded_data_path, output_folder):
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        input_path = line.strip()
        final_folder_name = "_".join(x for x in input_path.split('/')[:-1])
        filename = input_path.split('/')[-1]
        output_class_folder = os.path.join(output_folder, final_folder_name)

        if not os.path.exists(output_class_folder):
            os.makedirs(output_class_folder)

        full_input_path = os.path.join(downloaded_data_path, input_path)
        output_file_path = os.path.join(output_class_folder, filename)
        # print(final_folder_name, filename, output_class_folder, full_input_path, output_file_path)
        # exit()
        shutil.copy(full_input_path, output_file_path)
        if i % 100 == 0:
            print(f"Processed {i}/{len(lines)} images")

downloaded_data_path = os.path.join(WORK_DIR, 'datasets/dtd/images')
process_dataset(os.path.join(WORK_DIR, 'datasets/dtd/labels/train1.txt'), downloaded_data_path, os.path.join(WORK_DIR, 'datasets/dtd/train'))
process_dataset(os.path.join(WORK_DIR,'datasets/dtd/labels/val1.txt'), downloaded_data_path, os.path.join(WORK_DIR, 'datasets/dtd/train'))
process_dataset(os.path.join(WORK_DIR, 'datasets/dtd/labels/test1.txt'), downloaded_data_path, os.path.join(WORK_DIR, 'datasets/dtd/val'))
