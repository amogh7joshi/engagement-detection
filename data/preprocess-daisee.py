#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys
import fnmatch
import subprocess

import cv2

# Preprocessing Script for the DAiSEE Engagement Dataset.

datadir = os.path.join(os.path.dirname(__file__), 'dataset', 'daisee-dataset', 'Dataset')

def split_video():
    def split(video_file, image_name_prefix, destination_path):
        return subprocess.run(
            'ffmpeg -i "' + os.path.join(destination_path, video_file) + '" ' + image_name_prefix + '%d.jpg -hide_banner',
            shell = True, cwd = destination_path
        )

    # Get Train, Validation, and Test Datasets.
    i = 0
    for dataset in list(os.walk(datadir))[0][1]:
        users = os.listdir(os.path.join(datadir, dataset))
        # Get Each Individual User.
        for user in users:
            # Watch out for .DS_Store on MacOS.
            if sys.platform == "darwin" and user == ".DS_Store": continue
            videos = os.listdir(os.path.join(datadir, dataset, user))
            # Get Each Individual Video.
            for video in videos:
                # Watch out for .DS_Store on MacOS.
                if sys.platform == "darwin" and video == ".DS_Store": continue
                clip = os.listdir(os.path.join(datadir, dataset, user, video))# - [0]
                try:
                    if clip[0].endswith((".jpg", ".png")) or clip[1].endswith((".jpg", ".png")):
                        print(f"Skipping Video {video} --> Already Extracted")
                        continue
                except IndexError: clip = clip[0]

                # Extraction
                print(f"Video {clip} Extraction Beginning... ", end = ' ')
                # print(clip)
                # print(clip[:-4])
                # print(os.path.join(datadir, dataset, user, clip[:-4]))
                split(clip, video, os.path.join(datadir, dataset, user, video))
                i += 1; print(f"Video {clip} Extraction Complete!")

# Utility Function to remove unnecessary files.
# USE WITH CAUTION --> It will delete ALL images from directories.
# Should only be used as for intermediate cleaning or if absolutely necessary, a complete image removal function.
def remove_files():
    for dirpath, dirnames, filenames in os.walk(os.path.join(os.path.dirname(__file__), 'dataset', 'daisee-dataset')):
        for file in filenames:
            if fnmatch.fnmatch(file, '*.jpg'):
                os.remove(os.path.join(dirpath, file))

if __name__ == '__main__':
    split_video()

