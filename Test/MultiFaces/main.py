from FaceDetector.FaceDetector import FaceDetector
from imutils import paths
import numpy as np
import cv2
import os


# Show Divised Image
def Show_img(frame):
    top = np.hstack([frame[0], frame[1]])
    bottom = np.hstack([frame[2], frame[3]])
    combined = np.vstack([top, bottom])
    print("Please tape any Touch for continue")
    cv2.imshow("Face Detection Comparison", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    fd = FaceDetector()
    imgPath = list(paths.list_images("IMAGE_TO_DETECT"))

    if len(imgPath) > 1:

        for img in imgPath:
            # print(img)

            img_read = cv2.imread(img)
            vframe = []

            print("----------")
            print("HoG Method")
            print("----------")
            vframe.append(cv2.resize(fd.detectFaceDlibHog(img_read), (640, 480), interpolation=cv2.INTER_LINEAR))
            print("")

            print("----------")
            print("DNN Method")
            print("----------")
            vframe.append(cv2.resize(fd.detectFaceOpenCVDnn(img_read), (640, 480), interpolation=cv2.INTER_LINEAR))
            print("")

            print("----------")
            print("HaarCascade Method")
            print("----------")
            vframe.append(cv2.resize(fd.detectFaceOpenCVHaar(img_read), (640, 480), interpolation=cv2.INTER_LINEAR))
            print("")

            print("----------")
            print("MMOD Method")
            print("----------")
            vframe.append(cv2.resize(fd.detectFaceDlibMMOD(img_read), (640, 480), interpolation=cv2.INTER_LINEAR))
            print("")
            Show_img(vframe)
    else:
        print("No images found into IMAGE_TO_DETECT folder")
