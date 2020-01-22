# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import time
from configparser import ConfigParser
from os import path

import cv2
import dlib
import numpy as np
# *-----------------------*
# | Import Python Library |
# *-----------------------*
from imutils import paths

from FaceDetector.ExtractFaces import ExtractFaces
# ===========================================================================
#           Definition of Import
# ===========================================================================
# *-----------------------*
# | Import faces Detector |
# *-----------------------*
from FaceDetector.FaceDetectorDNN import FaceDetectorDNN
from FaceDetector.FaceDetectorHaar import FaceDetectorHaar
from FaceDetector.FaceDetectorHoG import FaceDetectorHoG
from FaceDetector.FaceDetectorMMOD import FaceDetectorMMOD
from FaceDetector.FaceDetectorTINY import FaceDetectorTiny
from Helper.Colors import Colors
from ObjectDetector.ObjectDetector import ObjectDetector
from Recognizer.Recognizer import Recognizer

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

# *============================*
# | Convert String to Boolean  |
# *============================*
def _convert_boolean(string):
    if re.match('(y|Y|Yes|yes|True|true)', string):
        return True
    else:
        return False


# =======================*
# | Affichage des infos  |
# *======================*
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


# =========================================
# Saving and get the old state
# =========================================
def _Saving(total):
    # *===================================*
    # |   Saving the pre processing File  |
    # *===================================*
    f = open("processing.dat", "w")
    f.write(str(total))
    f.close()
    del f


def _Reading():
    totalSkip = 0
    # *=====================================*
    # | if File exist try to read the file  |
    # | and get the last processus          |
    # *=====================================*
    if os.path.isfile("processing.dat"):
        f = open("processing.dat", "r")
        totalSkip = int(f.read())
        f.close()
        del f
    return totalSkip


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
            face_detector = FaceDetectorTiny(prob_thresh=float(config['prob_thresh']),
                                             nms_thres=float(config['nms_tresh']),
                                             lw=int(config['lw']),
                                             model=str(config['Tiny_Face_detection_model']))
        else:
            raise Exception(
                "[ERROR] Tiny Model no such file or directory : " + str(config['Tiny_Face_detection_model']))

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
    ap = argparse.ArgumentParser()
    ap.add_argument('-e', '--eventpath', help='Directory of Events', default='IMAGE_TO_DETECT')
    ap.add_argument('-i', '--imgdb', help='Directory Image to Train', default='IMAGE_DB_RAW')
    ap.add_argument('-v', '--verbose', help='Print infos', default='True')
    args, u = ap.parse_known_args()
    args = vars(args)
    del ap

    verbose = _convert_boolean(str(args['verbose']))

    # *=============================*
    # |  Read the ini config file   |
    # *=============================*
    if verbose:
        Colors.print_infos("[INFOS] Reading detector config ...")

    # *================================*
    # |  Get if use FacialRecognizing  |
    # *================================*
    config = ConfigParser()
    config.read('Data/Config/detector.ini')
    config = config['General']
    use_FacialRecognizer = _convert_boolean(config['use_facial_recognizion'])
    use_ALPR = _convert_boolean(config['use_alpr'])
    del config

    # *=====================================*
    # |  Get list of pictures into folders  |
    # *=====================================*
    images = list(paths.list_images(str(args['eventpath'])))

    t1 = time.time()
    # ===========================*
    # | check files to TRAIN     |
    # |            AND           |
    # | LAUNCH TRAINNING PROCESS |
    # *==========================*
    if len(list(paths.list_images(str(args['imgdb'])))) > 0:
        if verbose:
            Colors.print_sucess("[NEW] New Image Detected Run Analyse...\n")
        fd = ExtractFaces()
        fd.run()
        del fd
    else:
        if verbose:
            Colors.print_infos("[INFO] Nothing to Train now")

    # =========================*
    # | check files to Detect  |
    # |         AND            |
    # | Launch Infos Extractor |
    # *========================*
    if len(images) > 0:

        if verbose:
            Colors.print_sucess("[NEW] New Image(s) Detected Run Recognizing...\n")
            Colors.print_infos("[INFOS] Loading object and Face detector ...")

        # *=============================*
        # | Create Face/object detector |
        # *=============================*
        object_detector = create_object_detector()
        face_detector = create_face_detector()

        if verbose:
            Colors.print_sucess("[SUCCESS] Object and Face detector Loaded !")

        cpt = 0
        # *=================================*
        # | Foreach image in list of images |
        # *=================================*
        for img in images:

            if verbose:
                Colors.print_infos("\n[PROCESSING] Processing Image {0}/{1}".format(cpt + 1, len(images)))
                Colors.print_infos("\n[PROCESSING] Loading Recognizer...")

            reco = Recognizer()

            if verbose:
                Colors.print_sucess("\n[SUCCESS] Recognizer Loaded !\n")

            # *=============================*
            # | Running the Object Detector |
            # *=============================*
            yolo_result = object_detector.run(img)

            if re.match('person', yolo_result) and use_FacialRecognizer:

                if verbose:
                    Colors.print_infos("[INFOS] Person Was detected !")

                # *==================*
                # | Extract the Face |
                # *==================*
                if verbose:
                    Colors.print_infos("[PROCESSING] Running Extract Face Process...")

                result = face_detector.detectFace(frame=cv2.imread(img))
                faces = result[0]
                refined_bbox = result[1]
                del result

                if verbose:
                    Colors.print_sucess("\n[PROCESSING] " + str(len(faces)) + " Face Detected ")

                if len(faces) > 0:
                    if verbose:
                        Colors.print_infos("[PROCESSING] Running Facial Recognizing...\n")

                    result = reco.run(faces, face_detector, cv2.imread(img), refined_bbox)

                    if verbose:
                        Colors.print_sucess("\n[SUCESS] Detected Person: " + str(result[1]) + " \n")

                    # cv2.imshow("DEBUG FaceRecognizing", result[0])
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

            elif yolo_result == 'car' and use_ALPR:
                print("CAR AND ALPR")
                # TODO ALPR RECOGNIZING
            else:
                # os.system("rm -rfv " + image)
                print("Nothing Detected")

    if _convert_boolean(str(args['verbose'])):
        Colors.print_sucess("\n[SUCCESS] Finished with Total processing time : " + str(round(time.time()-t1, 3)) + " s")
    del args
    del images
    del t1

