import numpy as np
import cv2

def main():
    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier('haarcascade/face.xml')
    smile_cascade = cv2.CascadeClassifier('haarcascade/smile.xml')

    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        smiles = smile_cascade.detectMultiScale(gray, 1.8, 5)
        faces = face_cascade.detectMultiScale(gray, 1.8, 5)
        if (len(faces) == 1) and (len(smiles) > 0):
            x,y,a,b = faces[0][0], faces[0][1], faces[0][2], faces[0][3]
            for smile in smiles:
                x2,y2,a2,b2 = smile[0], smile[1], smile[2], smile[3]
                if ((x2>x) and (y2>y) and (a>a2) and (b>b2)):
                    cv2.rectangle(frame,(x2,y2),(x2+a2,y2+b2),(255,0,0),2)
                    roi_gray = gray[y2:y2+b2, x2:x2+a2]
                    roi_color = frame[y2:y2+b2, x2:x2+a2]
                    break

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
