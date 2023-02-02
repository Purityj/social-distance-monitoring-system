#Idea behind social distance detecting - detect a person
# and based on the bounding boxes of those persons, calculate the centroid of 2 people.
#if the distance between centroid person1 and person2 is less than a threshold value(required social distance), then
#mark their bounding boxes as red else mark their boxes as green

import cv2
import datetime
import imutils
import numpy as np
from centroidtracker import CentroidTracker #includes the tracker
import math
from itertools import combinations  #for creating combinations btw persons (say person 0,1,2,3, this would
#create a combination btw person 1 and 3 or 2 and 3 or 2, 4, 6)


#mention the path to your detection file -> protopath
#and the modelpath - path to your model
protopath = "model files/generic object detection model/MobileNetSSD_deploy.prototxt"
modelpath = "model files/generic object detection model/MobileNetSSD_deploy.caffemodel"

#initialize the detector and have it read from caffemodel
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

#if running your code through openVino env, include these 2 lines
#they simply say, use the Backend_inference for inferencing and use my cpu
# detector.setPreferableBackend(cv2.dnn.BACKEND_INFERENCE_ENGINE)
# detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#define the classes. Our model is a generic detection model; it can detect any object around us
#it can detect a car, cat, dog, person etc
#here we are mentioning all the classes it has and while inferencing, we'll only choose the one which has a person
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

#initialize the tracker
#maxDisappeared is how long the tracker should wait for a frame to appear in a certain location so that it assign it the same id
tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)

#non max suppression algorithm to remove noise like multiple bounding boxes around an object
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

#fps(frames per second) - how many frames your hardware is able to process in 1s. 30fps means the hardware is able to process 30 frames per second
#fps is dependednt on the type of h/w you have, good laptops have high fps, bad laptops have low fps

def main():
    # get the video file and store it in cap
    cap = cv2.VideoCapture('video/testvideo2.mp4')

    #calculating fps - params -> start and end time value
    #To get fps = total frames/(start time - end time)
    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0
    centroid_dict = dict()      #where the centroid will be saved at

    while True:
        #read frames from video file
        ret, frame = cap.read()

        #resize the frame
        frame = imutils.resize(frame, width=800)

        #increase total frames-> when going through the while loop, go to 1st frame, 2nd, etc
        total_frames = total_frames + 1

        #calculate the height and the width from the frame
        (H, W) = frame.shape[:2]

        # create a blob out of our image which will the used for infrencing
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        # pass this blob into our detector
        detector.setInput(blob)

        # get all the results from the detector
        # person_detections has all the results from our model file
        person_detections = detector.forward()

        #a list where all the coordinates are saved
        rects = []

        # iterate over all the detections
        for i in np.arange(0, person_detections.shape[2]):
            # calculate the confidence of the detection -> if the confidence is above threshold, we accept that result
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:  # 0.5 is the threshold
                # calculate the index of the detection
                idx = int(person_detections[0, 0, i, 1])

                # based on this index, check which class it belongs to. If it isn't a person, continue
                # iterating, if it is person, calculate the bounding box
                if CLASSES[idx] != "person":
                    continue
                # the bounding box has cordinates like start x & y
                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                # calculating the start x&y and end x&y
                (startX, startY, endX, endY) = person_box.astype("int")
                #all person bounding boxes cordinates are saved here
                rects.append(person_box)

        #apply the non max suppression algorithm here to remove all noise(unwanted bounding boxes) from the persons in the image/video
        boundingBoxes = np.array(rects)
        boundingBoxes = boundingBoxes.astype(int) #convert bounding boxes to integer
        rects = non_max_suppression_fast(boundingBoxes, 0.3)  #pass all the bounding boxes to nonmaxsuppression & give it a threshold value of 0.3

        #above rects has the correct bounding boxes(right coordinates for the objects) & doesnt have noise in them
        #pass the bounding boxes(cordinates) to the tracker which will keep track of the bounding boxes(objects inside the boundingboxes)
        #after the tracker has taken all the cordinates it returns objects id
        objects = tracker.update(rects)

        #objects variable contains the bounding box and the object id of an object
        for (objectId, bbox) in objects.items():
            #assign bounding box values, the cordinates, to x1,y1,x2,y2
            x1, y1, x2, y2 = bbox

            #convert them to integers
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            #calculating the centroids
            cX = int((x1 + x2)/2.0)
            cY = int((y1 + y2) / 2.0)

            #save the centroids and the bounding boxes of each person a dictionary
            centroid_dict[objectId] = (cX, cY, x1, y1, x2, y2)

            #display the object id given by the tracker
            #use cv2.putText to put the text(below string) to the frame
            text = "ID: {}".format(objectId)
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        #list of persons who come on the red zone
        red_zone_list = []
        #this for loop iterates contents of the dictionary
        for(id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
            #calculate centroid, taking centroid of p1 - centroid of p2
            dx, dy = p1[0] -p2[0], p1[1] - p2[1]

            #calculate the real distance
            distance = math.sqrt(dx * dx + dy * dy)

            #check if social dist(75) is observed, if not, put their id in a list
            if distance < 75.0:
                if id1 not in red_zone_list:
                    red_zone_list.append(id1)
                if id2 not in red_zone_list:
                    red_zone_list.append(id2)

        #iterate the centroids again and display those ids in green or red bounding boxes
        for id, box in centroid_dict.items():
            if id in red_zone_list:
                # display these cordinates using red rectangle/bounding boxes
                cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 0, 255), 2)
            else:
                # display these cordinates using green rectangle/bounding boxes
                cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2)



        #get fps end time
        fps_end_time = datetime.datetime.now()

        #get time difference
        time_diff = fps_end_time - fps_start_time

        #get fps
        #N/B: at the start the time diff will be very small, comp will interpret as zero.
        #since we can't divide by zero, use this if else so that if the time diff is zero we display nothing
        if time_diff.seconds == 0:
            fps = 0
        else:
            fps = (total_frames/time_diff.seconds)

        #display the fps value through text
        #first round it to 2 dps
        fps_text = "FPS: {:.2f}".format(fps)
        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 1)

        #display the video
        cv2.imshow('Application', frame)

        #stop the while loop
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

#after breaking destroy all windows
cv2.destroyAllWindows()

#call the main function
main()

