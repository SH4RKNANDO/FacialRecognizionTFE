# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===========================================================================
#           Definition of Import
# ===========================================================================
# *-----------------------*
# | Import faces Detector |
# *-----------------------*
# from FaceDetector.FaceDetector import FaceDetector
from FaceDetector.FaceDetectorDNN import FaceDetectorDNN
from FaceDetector.FaceDetectorHaar import FaceDetectorHaar
from FaceDetector.FaceDetectorHog import FaceDetectorHoG
from FaceDetector.FaceDetectorMMOD import FaceDetectorMMOD
from FaceDetector.FaceDetectorTINY import FaceDetectorTINY
from FaceDetector import tiny_face_model

# *------------------------*
# | Import Object Detector |
# *------------------------*
from ObjectDetector.ObjectDetector import ObjectDetector

# *------------------------*
# | Import Python Library  |
# *------------------------*
from configparser import ConfigParser
from imutils import paths
from os import path
import argparse
import re
import cv2
import time
import numpy as np
import os
import dlib

# ===========================================================================
#           Infos developer
# ===========================================================================
__author__ = "Jordan BERTIEAUX"
__copyright__ = "Copyright 2020, Facial Recognition"
__credits__ = ["Jordan BERTIEAUX"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Jordan BERTIEAUX"
__email__ = "jordan.bertieaux@std.heh.be"
__status__ = "Production"

# =========================================== < HELPERS FUNCTION > ====================================================


# =============================*
# | Convert String to Boolean  |
# *============================*
def _convert_boolean(string):
    if re.match('(y|Y|Yes|yes|True|true)', string):
        return True
    else:
        return False


def _top():
    os.system("clear")
    print("\n*-----------------------------------------------------*")
    print("| __author__ = Jordan BERTIEAUX                       |")
    print("| __copyright__ = Copyright 2020, Facial Recognition  |")
    print("| __credits__ = [Jordan BERTIEAUX]                    |")
    print("| __license__ = GPL                                   |")
    print("| __version__ = 1.0                                   |")
    print("| __maintainer__ = Jordan BERTIEAUX                   |")
    print("| __email__ = jordan.bertieaux@std.heh.be             |")
    print("| __status__ = Production                             |")
    print("*-----------------------------------------------------*\n")


# =========================================== < DETECTOR FUNCTION > ===================================================


# ==========================================*
# | Create Object Detector From config.ini  |
# *=========================================*
def create_object_detector():
    config = ConfigParser()
    config.read('Data/Config/detector.ini')
    config = config['object']
    NET = None

    if path.isfile(config['yolo_labels_path']):
        LABELS = open(config['yolo_labels_path']).read().strip().split("\n")
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    else:
        raise Exception("Error : LabelPath no such file or directory : {0}".format(config['yolo_labels_path']))

    if path.isfile(config['yolo_weights_path']) and path.isfile(config['yolo_config_path']):
        NET = cv2.dnn.readNetFromDarknet(config['yolo_config_path'], config['yolo_weights_path'])

    elif not path.isfile(config['yolo_weights_path']):
        raise Exception("Error : weightsPath no such file or directory : {0}".format(config['yolo_weights_path']))

    elif not path.isfile(config['yolo_config_path']):
        raise Exception("Error : config_path no such file or directory : {0}".format(config['yolo_config_path']))

    obj = ObjectDetector(float(config['confidence']), float(config['threshold']), [LABELS, COLORS, NET],
                         _convert_boolean(config['yolo_show_percent']), _convert_boolean(config['yolo_override_ZM']),
                         config['detect_pattern'])
    # clean the RAM
    del config
    del LABELS
    del COLORS
    del NET
    return obj


# *=======================================*
# | Create Face Detector From config.ini  |
# *=======================================*
def create_face_detector():
    config = ConfigParser()
    config.read('Data/Config/detector.ini')
    config = config['General']
    face_detector = None

    # ELIF FACEDETECTOR == Tiny
    if config['face_detector_process'] == "Tiny":
        config = ConfigParser()
        config.read('Data/Config/detector.ini')
        config = config['FaceDetectorTiny']

        if path.isfile(config['Tiny_Face_detection_model']):
            face_detector = FaceDetectorTINY(MAX_INPUT_DIM=config['MAX_INPUT_DIM'],
                                             prob_thresh=float(config['prob_thresh']),
                                             nms_thres=float(config['nms_tresh']),
                                             lw=int(config['lw']),
                                             model=tiny_face_model.Model(config['Tiny_Face_detection_model']))
        else:
            raise Exception(
                "[ERROR] MMOD Model no such file or directory : ".format(config['Tiny_Face_detection_model']))

    # IF FACEDETECTOR == DNN
    elif config['face_detector_process'] == "DNN":
        config = ConfigParser()
        config.read('Data/Config/detector.ini')
        config = config['FaceDetectorDNN']

        if path.isfile(config['modelFile']) and path.isfile(config['configFile']):
            face_detector = FaceDetectorDNN(float(config['conf_threshold']),
                                            config['process_model'],
                                            config['modelFile'],
                                            config['configFile'])
        else:
            if not path.isfile(config['modelFile']):
                raise Exception("[ERROR] No such file or Directory : {0}".format(config['modelFile']))
            elif not path.isfile(config['configFile']):
                raise Exception("[ERROR] No such file or Directory : {0}".format(config['configFile']))

    # ELIF FACEDETECTOR == HaarCascade
    elif config['face_detector_process'] == "Haar":
        config = ConfigParser()
        config.read('Data/Config/detector.ini')
        config = config['FaceDetectorHaar']

        if path.isfile(config['haarcascade_frontalface_default']):

            face_detector = FaceDetectorHaar(int(config['max_multiscale']),
                                             float(config['min_multiscale']),
                                             cv2.CascadeClassifier(config['haarcascade_frontalface_default']))
        else:
            raise Exception("[ERROR] HaarCasecade Model No such file or Directory : ".format(config['haarcascade_frontalface_default']))

    # ELIF FACEDETECTOR == MMOD
    elif config['face_detector_process'] == "MMOD":
        config = ConfigParser()
        config.read('Data/Config/detector.ini')
        config = config['FaceDetectorMMOD']

        if path.isfile(config['cnn_face_detection_model_v1']):
            face_detector = FaceDetectorMMOD(dlib.cnn_face_detection_model_v1(config['cnn_face_detection_model_v1']))
        else:
            raise Exception("[ERROR] MMOD Model no such file or directory : ".format(config['cnn_face_detection_model_v1']))

    # ELIF FACEDETECTOR == HoG
    elif config['face_detector_process'] == "HoG":
        face_detector = FaceDetectorHoG()

    # ELIF FACEDETECTOR == ERROR
    else:
        raise Exception("ERROR No FaceDetector Selected into detector.ini")

    del config
    return face_detector


# ============================================= < MAIN FUNCTION > =====================================================

if __name__ == "__main__":
    _top()
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--eventpath', help='Path of Events Directory', default='IMAGE_TO_DETECT')
    args, u = ap.parse_known_args()
    args = vars(args)

    # *=============================*
    # |  Read the ini config file   |
    # *=============================*
    print("[INFOS ] Reading detector config ...")
    config = ConfigParser()
    config.read('Data/Config/detector.ini')
    config = config['General']

    if path.isdir(args['eventpath']):
        images = list(paths.list_images(args['eventpath']))
        cpt = 0

        if len(images) < 1:
            print("No Images Detected")
        else:
            # *=============================*
            # | Create Face/object detector |
            # *=============================*
            print("[INFOS ] Loading object detector ...")
            object_detector = create_object_detector()
            print("[INFOS ] Loading face detector ...")
            face_detector = create_face_detector()

            for image in images:
                start = time.time()
                cpt += 1

                print("\nProcessing Image {0}/{1}".format(cpt, len(images)))
                object_detector.image_path = image
                yolo_result = object_detector.run()

    #             if yolo_result == 'person' and _convert_boolean(config['use_facial_recognizion']):
    #                 print("Recognizing Process...")
    #                 frame = face_detector.detectFace(cv2.imread(image))
    #                 if frame[1] > 0 or frame2[1] > 0 or frame3[1] > 0:
    #                     print(image)
    #                     cv2.imshow("DEBUG", frame[0])
    #                     cv2.waitKey(0)
    #                     cv2.destroyAllWindows()
    #                 else:
    #                     print('No Faces Detected')
    #             elif yolo_result == 'car' and _convert_boolean(config['use_alpr']):
    #                 print("CAR AND ALPR")
    #                 # TODO ALPR RECOGNIZING
    #             else:
    #                 os.system("rm -rfv " + image)
    #                 print("Nothing Detected")
    #
    #             print("Processing Time {0} Seconds".format(round(time.time()-start, 2)))
    else:
        print("Folder Image Not Found : " + str(args['eventpath']))
