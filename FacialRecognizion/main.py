from imutils import paths
from FaceDetector.FaceDetectorHaar import FaceDetectorHaar
from imageio import imsave
import cv2


def check_new_files_db():
    if len(list(paths.list_images('IMAGE_DB'))) > 0:
        return True
    else:
        return False


def check_file_to_detect():
    if len(list(paths.list_images('IMAGE_TO_DETECT'))) > 0:
        return True
    else:
        return False


# Show Divised Image
def Show_img(frame):
    cv2.imshow("Face Detection Comparison", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Show Divised Image
def Save_img(frame, name):
    imsave(frame, name)
    # cv2.imshow("Face Detection Comparison", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    if check_file_to_detect():
        fd = FaceDetectorHaar()
        imgPath = list(paths.list_images("IMAGE_TO_DETECT"))
        cpt = 0

        for img in imgPath:
            cpt += 1
            img_read = cv2.imread(img)
            # cv2.resize(img_read, (1280, 720), interpolation=cv2.INTER_LINEAR)

            frame = fd.detectFaceOpenCVHaar(img_read)

            print("Processing {0} / {1} ".format(cpt, len(imgPath)))
            if frame is not None:
                cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
                Save_img("IMG_DB_RESULT/IMG_{0}_.jpg".format(int(cpt)), frame)
            else:
                print("No Face Detected !")
