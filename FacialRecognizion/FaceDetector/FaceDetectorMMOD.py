import cv2
import dlib


class FaceDetectorMMOD:
    def __init__(self):
        self._cnnFaceDetector = dlib.cnn_face_detection_model_v1("Data/Model/mmod_human_face_detector.dat")

    def detectFaceDlibMMOD(self, frame, inHeight=300, inWidth=0):
        frameDlibMMOD = frame.copy()
        frameHeight = frameDlibMMOD.shape[0]
        frameWidth = frameDlibMMOD.shape[1]
        if not inWidth:
            inWidth = int((frameWidth / frameHeight) * inHeight)

        scaleHeight = frameHeight / inHeight
        scaleWidth = frameWidth / inWidth

        frameDlibMMODSmall = cv2.resize(frameDlibMMOD, (inWidth, inHeight))

        frameDlibMMODSmall = cv2.cvtColor(frameDlibMMODSmall, cv2.COLOR_BGR2RGB)
        faceRects = self._cnnFaceDetector(frameDlibMMODSmall, 0)

        # print(frameWidth, frameHeight, inWidth, inHeight)
        bboxes = []

        cv2.putText(frameDlibMMOD, "OpenCV MMOD", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)

        if len(faceRects) < 1:
            return None
        for faceRect in faceRects:
            cvRect = [int(faceRect.rect.left() * scaleWidth), int(faceRect.rect.top() * scaleHeight),
                      int(faceRect.rect.right() * scaleWidth), int(faceRect.rect.bottom() * scaleHeight)]
            bboxes.append(cvRect)
            cv2.rectangle(frameDlibMMOD, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0),
                          int(round(frameHeight / 150)), 4)

        return frameDlibMMOD
