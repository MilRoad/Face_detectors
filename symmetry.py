from imutils import face_utils
import imutils
import dlib
import cv2
import os

class SymmetryLines:
    def detect(self, photo):
        # construct the argument parser and parse the arguments
        # ap = argparse.ArgumentParser()
        # ap.add_argument("-p", "--shape-predictor", required=True, help="shapepredictor.bz2")
        # ap.add_argument("-i", "--image", required=True, help=photo)
        # args = vars(ap.parse_args())

        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        global shape
        shapepredictor = 'shape_predictor_68_face_landmarks.dat'
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(shapepredictor)

        # load the input image, resize it, and convert it to grayscale
        image = cv2.imread(photo)
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale image
        rects = detector(gray, 1)

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # show the face number
            # cv2.putText(image, "Лицо".format(i + 1), (x - 10, y - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        # show the output image with the face detections + facial landmarks

        if shape.size > 0:
            firstPoint = (int((shape[37][0]+shape[38][0])/2), shape[38][1]-50)
            secondPoint = (int((shape[40][0]+shape[41][0])/2), shape[40][1]+50)
            cv2.line(image, firstPoint, secondPoint, (0, 255, 0))
            firstPoint = (int((shape[43][0]+shape[44][0])/2), shape[43][1]-50)
            secondPoint = (int((shape[46][0]+shape[47][0])/2), shape[46][1]+50)
            cv2.line(image, firstPoint, secondPoint, (0, 255, 0))
            firstPoint = (shape[27][0], shape[27][1])
            secondPoint = (shape[8][0], shape[8][1])
            cv2.line(image, firstPoint, secondPoint, (110, 255, 110))
        cv2.imshow("Линии симметрии", image)
        cv2.waitKey(0)

if __name__ == '__main__':
    file_directory = 'photo'
    images = [os.path.join(file_directory, f'photo{i}.jpg') for i in range(1, 8)]

    matching_instance = SymmetryLines()

    for image in images:
        print(f'detecting {image}')
        matching_instance.detect(image)
