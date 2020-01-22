# !/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Jordan BERTIEAUX"
__copyright__ = "Copyright 2019, Facial Recognition"
__credits__ = ["Jordan BERTIEAUX"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Jordan BERTIEAUX"
__email__ = "jordan.bertieaux@std.heh.be"
__status__ = "Production"


# ===========================================================================
#         Definition of Import
# ===========================================================================
from Helper.Colors import Colors
from Helper.Serializer import Serializer
from FaceDetector.FaceDetector import FaceDetector
from Helper.PATH import PATH
from tqdm import tqdm

import glob
import pandas as pd
import cv2
import time
import os


# ===========================================================================
#         Definition of class ExtractFaces
# ===========================================================================
class ExtractFaces:
    def __init__(self):
        self._color = Colors()
        self._serializer = Serializer()

    # ===========================================================================
    #         Function of main
    # ===========================================================================
    def run(self):
        self._color.printing("info", "[LOADING] Quantifying faces...")

        # Get list of Folder
        # print(glob.glob("IMAGE_DB_RAW/*"))
        data = self._format_data(glob.glob("IMAGE_DB_RAW/*"))

        self._color.printing("info", "[INFO] Create Folders of databases...")

        for name in data.name:
            if not os.path.isdir("Data/IMAGE_DB/" + name):
                os.mkdir("Data/IMAGE_DB/" + name)

        fd = FaceDetector(prob_thresh=float(0.5), nms_thres=float(0.1), lw=int(3),
                          model="/home/zerocool/PycharmProjects/FacialRecognizionTFE/Test/FaceRecognizerV4.0/Data/Model/hr_res101.weight")

        cpt = 0
        t1 = time.time()
        for img_path in data.image:
            # print(img_path)
            self._color.printing("info", "[PROCESSING] Extract Faces {}/{}".format(cpt + 1, len(data.image)))
            fd.ExtractFace(cv2.imread(img_path), "Data/IMAGE_DB/" + str(data.name[cpt]) + "/result_" + str(cpt))
            cpt += 1

        self._color.printing("info", "[INFO] Remove file in IMG_DB_RAW...")
        os.system("rm -rfv IMAGE_DB_RAW/*")

        self._color.printing("success", "[SUCCESS] Extraction Completed in " + str(round(time.time()-t1, 4)) + " s\n")

        # Cleanning RAM
        del data
        del fd
        del cpt
        del t1

        # Saving Images
        self._saving()

    # ===========================================================================
    #         Create the Data Frame with Panda
    # ===========================================================================
    """
    @:parameter train_path = Path from glog (UNIX LIKE)
    """
    def _format_data(self, train_paths):
        data = pd.DataFrame(columns=['image', 'label', 'name'])

        for i, train_path in tqdm(enumerate(train_paths)):
            name = train_path.split("/")[-1]
            images = glob.glob(train_path + "/*")
            for image in images:
                data.loc[len(data)] = [image, i, name]

            del name
            del images

        # print(data)
        return data

    def _saving(self):
        # Get list of Folder
        self._serializer.saving_data(self._format_data(glob.glob("Data/IMAGE_DB/*")))

        # print(data)
        # self._serializer.saving_faces(self._faces)
