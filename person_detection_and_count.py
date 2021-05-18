import cv2
import datetime
import imutils
import numpy as np
from centroid_tracker import CentroidTracker
from imutils.object_detection import non_max_suppression
import smtplib
import time
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
toadd="ms2030914@gmail.com@gmail.com"
myadd="mk1263790@gmail.com"
Subject="Security Alert"
msg=MIMEMultipart()
msg["Subject"]=Subject
msg["From"]=myadd
msg["To"]=toadd

protopath = "persondetection.prototxt"
modelpath = "persondetection.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
           "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
           "tvmonitor"]

tracker = CentroidTracker(maxDisappeared= 20, maxDistance= 90)

def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))

def main():
    cap = cv2.VideoCapture('test_video.mp4')

    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0
    lpc_count = 0
    opc_count = 0
    object_id_list = []

    while True:
        ret, frame = cap.read()

        frame = imutils.resize(frame, width=400)
        total_frames = total_frames + 1

        (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        detector.setInput(blob)
        person_detections = detector.forward()
        rects = []

        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue
                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box)


        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)

        objects = tracker.update(rects)
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = "Person {}".format(objectId)
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255),1)
            if objectId not in object_id_list:
                object_id_list.append(objectId)

        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)

        fps_text = "FPS: {:.2f}".format(fps)

        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        lpc_count = len(objects)
        opc_count = len(object_id_list)

        lpc_txt = "LPC = {}".format(lpc_count)
        opc_txt = "OPC = {}".format(opc_count)

        cv2.putText(frame, lpc_txt, (5, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        cv2.putText(frame, opc_txt, (5, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        pick = non_max_suppression_fast(boundingboxes, 0.3)
        #
        for (xa, ya, wa, ha) in pick:
        #     cv2.rectangle(frame, (xa, ya), (xa + W, ya + H), (255, 255, 0), 2)
            cv2.imwrite("image.jpg", frame)
            pic = open("image.jpg", "rb")
            img = MIMEImage(pic.read())
            pic.close()
            msg.attach(img)
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(user="mk1263790@gmail.com", password="manisidhu11")
            server.sendmail(myadd, toadd, msg.as_string())
            server.quit()
            time.sleep(1)

        cv2.imshow("Desktop", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


main()
        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)

        fps_text = "FPS: {:.2f}".format(fps)

        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        lpc_count = len(objects)
        opc_count = len(object_id_list)

        lpc_txt = "LPC = {}".format(lpc_count)
        opc_txt = "OPC = {}".format(opc_count)

        cv2.putText(frame, lpc_txt, (5, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        cv2.putText(frame, opc_txt, (5, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        cv2.imshow("Desktop", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


main()
