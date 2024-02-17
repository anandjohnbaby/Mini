import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Function to detect persons in the frame
def detect_person(frame):
    classes = []
    with open("coco.names", "r") as f:
        classes = f.read().splitlines()

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Class ID 0 is for person detection
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected_boxes = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            detected_boxes.append((x, y, x + w, y + h))

    return detected_boxes

# Function to calculate centroid of a bounding box
def calculate_centroid(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))  # Return centroid coordinates

# Load CCTV footage
cap = cv2.VideoCapture('not_intrider.webm')

# Previous centroids for tracking
prev_centroids = {}

# Process frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect persons in the frame
    detected_boxes = detect_person(frame)

    # Update the tracking using centroids
    current_centroids = {i: calculate_centroid(box) for i, box in enumerate(detected_boxes)}
    
    # Intruder detection logic
    if prev_centroids:
        for i, centroid in current_centroids.items():
            if i not in prev_centroids:
                print("Intruder detected!")  # New object detected, raise alert
                break
            else:
                prev_centroid = prev_centroids[i]
                # Your behavior analysis and alerting logic would go here
                # Example: check for suspicious behavior based on the tracked objects' movements
                # For simplicity, we'll just compare the current centroid with the previous one
                if np.linalg.norm(np.array(centroid) - np.array(prev_centroid)) > 50:  # Adjust threshold as needed
                    print("Intruder detected!")  # Movement deviation detected, raise alert
                    break

    # Update previous centroids for next frame
    prev_centroids = current_centroids

    # Draw bounding boxes
    for box in detected_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('CCTV', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Manually close OpenCV windows using cv2.waitKey()
cv2.destroyAllWindows()
cv2.waitKey(1)

