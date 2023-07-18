import glob
import os
import re


file_path = 'data/grasp-anything'
grasp_files = glob.glob(os.path.join(file_path, 'positive_grasp', '*.pt'))
grasp_files.sort()

image_files = os.listdir(os.path.join(file_path, 'image'))
count = 0
for grasp_file in grasp_files:
    rgb_file = re.sub(r"_\d{1}\.pt", ".jpg", grasp_file)
    rgb_file = rgb_file.split('/')[-1]
    if rgb_file not in image_files:
        count += 1
        os.remove(grasp_file)

print(count)