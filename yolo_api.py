import cv2
import numpy as np
import glob
import random
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load Yolo
net = cv2.dnn.readNet("yolov3_training_final.weights", "yolov3_testing.cfg")

# Name custom object
classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()
@app.route('/api/v1/resources/<path:pathToFile>', methods=['GET'])
def api_all(pathToFile):
    #images_path = glob.glob(r"C:yourpath\1608647733768.jpg") change images_path to run
    #get pathToFile from API
    image = pathToFile
    images_path = glob.glob(image)
    print("images_path from api:", pathToFile)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # loop through all the images
    for img_path in images_path:
        # Loading image
        img = cv2.imread(img_path)
        img = cv2.resize(img, None, fx=0.6, fy=0.6)
        height, width, channels = img.shape
        # Prepare image and convert it as an input image that can be fit into yolo network
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.6:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
		    # Draw a bounding box rectangle and label
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.4)
        data = []
        font = cv2.FONT_HERSHEY_PLAIN
        if len(indexes)>0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i],2))
                color = colors[i]
		    # Create bounding boxes around objects
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                cv2.putText(img, label + " " + confidence, (x, y - 20), font, 2, (255,255,255), 2)
		    # Show label and confidence
                print("Cake: ", label)
                print("Confidence: ", confidence)
                data.append(label)
    # Create a response with JSON 
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=3000)
