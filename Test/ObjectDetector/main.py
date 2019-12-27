# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os

from os import path
from imutils import paths
from ObjectDetector.ObjectDetector import ObjectDetector
from shutil import copyfile


def check_img(folder):
    img = None

    if path.isdir(folder):
        img = list(paths.list_images(folder))

        if len(img) >= 1:
            i = 0

            for im in img:
                copyfile(im, "DB_RESULT/image_" + str(i) + ".jpg")
                i += 1

            img = list(paths.list_images("DB_RESULT"))

        else:
            img = None
            print("No Image Detected")
    else:
        print("No Folder Detected : " + str(folder))

    return img


def check_video(folder):
    video = None

    if path.isdir(folder):
        video = list(paths.list_files(folder))

        if len(video) < 1:
            video = None
            print("No Video Detected")
    else:
        print("No Folder Detected :" + folder)

    return video


if __name__ == "__main__":
    ap = argparse.ArgumentParser("main.py")
    ap.add_argument("-i", "--images", help="images folder", default="IMAGE_TO_DETECT")
    ap.add_argument("-v", "--videos", help="videos folder", default="VIDEO_TO_DETECT")
    args = ap.parse_args()

    # print(args.images)
    # print(args.videos)

    img = check_img(args.images)
    video = check_video(args.videos)

    # print(img)
    # print(video)

    if img is not None:
        cpt = 0
        t1 = time.time()

        for image in img:
            cpt += 1
            print("\n[INFOS] Proccessing : " + str(cpt) + " / " + str(len(img)))

            t3 = time.time()
            obj = ObjectDetector(image=image)
            obj.run()
            t4 = time.time()

            t_total = float("{0:.2f}".format(t4 - t3))
            print("[INFOS] Time Processing: " + str(t_total) + " second")

        t2 = time.time()
        t_total = float("{0:.2f}".format(t2 - t1))
        print("\n[INFOS] Proccessing Finished : " + str(t_total) + " second")
