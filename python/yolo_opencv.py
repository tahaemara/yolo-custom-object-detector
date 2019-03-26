import cv2
import argparse
import numpy as np
import websocket
import json

imgsize = (700,700)

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', 
                help = 'path to config file', default="custom/yolov3-tiny.cfg")
ap.add_argument('-w', '--weights', 
                help = 'path to pre-trained weights', default="../backup/yolov3-tiny.backup")
ap.add_argument('-cl', '--classes', 
                help = 'path to objects.names',default="custom/objects.names")
args = ap.parse_args()

ap.add_argument('-s', '--server',
                help = 'yes to connect to server, no to not', default='0')


# Connect to the Rover control server through a websocket
if 'yes' in args.server:
    ws = websocket.create_connection("ws://192.168.0.107:9020/locomotion")
    WheelControl = {"l":0, "r":0}


# Get names of output layers, output for YOLOv3 is ['yolo_16', 'yolo_23']
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Send a message to the Server through the WebSocket telling the rover to turn in a certain way
def SendMsgToServer(message):
    ws.send(json.dumps(message))

# Load names classes
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)

#Generate color for each class randomly
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Define network from configuration file and load the weights from the given weights file
net = cv2.dnn.readNet(args.weights,args.config)

# Define video capture for default cam
cap = cv2.VideoCapture(0)


while cv2.waitKey(1) < 0 or False:

    hasframe, image = cap.read()
    image=cv2.resize(image, imgsize) 
    
    blob = cv2.dnn.blobFromImage(image, 1.0/255.0, imgsize, [0,0,0], True, crop=False)
    Width = image.shape[1]
    Height = image.shape[0]
    net.setInput(blob)
    
    outs = net.forward(getOutputsNames(net))
    
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    
    
    #print(len(outs))
    
    # In case of tiny YOLOv3 we have 2 output(outs) from 2 different scales [3 bounding box per each scale]
    # For normal normal YOLOv3 we have 3 output(outs) from 3 different scales [3 bounding box per each scale]
    
    # For tiny YOLOv3, the first output will be 507x6 = 13x13x18
    # 18=3*(4+1+1) 4 boundingbox offsets, 1 objectness prediction, and 1 class score.
    # and the second output will be = 2028x6=26x26x18 (18=3*6) 
    
    for out in outs: 
        #print(out.shape)
        for detection in out:
            
        #each detection  has the form like this [center_x center_y width height obj_score class_1_score class_2_score ..]
            scores = detection[5:]#classes scores starts from index 5
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    # apply non-maximum suppression algorithm on the bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        if x<imgsize[0]/3:
            WheelControl["l"] = -0.5
            WheelControl["r"] = 0.5
            print("left")
        elif x>imgsize[0]*0.6666:
            WheelControl["l"] = 0.5
            WheelControl["r"] = -0.5
            print("right")
        else:
            WheelControl["l"] = 0
            WheelControl["r"] = 0
            print("straight")
        
        SendMsgToServer(WheelControl)
   
    # Put efficiency information.
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(image, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 0, 0))
    
    # cv2.imshow(window_title, image)
