import cv2


class FaceDetectorHaar:
    def __init__(self):
        # OpenCV HAAR
        self._faceCascade = cv2.CascadeClassifier('Data/Model/haarcascade_frontalface_default.xml')

    def detectFaceOpenCVHaar(self, frame, inHeight=300, inWidth=0):
        frameOpenCVHaar = frame.copy()
        frameHeight = frameOpenCVHaar.shape[0]
        frameWidth = frameOpenCVHaar.shape[1]
        if not inWidth:
            inWidth = int((frameWidth / frameHeight) * inHeight)

        scaleHeight = frameHeight / infireHeight
        scaleWidth = frameWidth / inWidth

        frameOpenCVHaarSmall = cv2.resize(frameOpenCVHaar, (inWidth, inHeight))
        frameGray = cv2.cvtColor(frameOpenCVHaarSmall, cv2.COLOR_BGR2GRAY)

        faces = self._faceCascade.detectMultiScale(frameGray)
        # faces = self._faceCascade.detectMultiScale(frameGray, 1.3, 3)
        # faces = self._faceCascade.detectMultiScale(frameGray, 3, 5)

        bboxes = []
        cv2.putText(frameOpenCVHaar, "OpenCV HaarCascade", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3,
                    cv2.LINE_AA)

        if len(faces) < 1:
            return None
        else:
            for (x, y, w, h) in faces:
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                cvRect = [int(x1 * scaleWidth), int(y1 * scaleHeight), int(x2 * scaleWidth), int(y2 * scaleHeight)]
                bboxes.append(cvRect)
                cv2.rectangle(frameOpenCVHaar, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0),
                              int(round(frameHeight / 150)), 4)
            return frameOpenCVHaar

    def create_rect(self, name, bbox, frame, frameHeight):
        cpt = 0

        for cvRect in bbox:
            cv2.putText(frame, name[cpt], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.rectangle(frame, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0),
                          int(round(frameHeight / 150)), 4)
            cpt += 1
