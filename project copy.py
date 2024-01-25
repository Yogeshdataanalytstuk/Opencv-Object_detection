import cv2
import imutils
import os
from ultralytics import YOLO
import easyocr
import re


# Function to calculate Intersection over Union (IoU)
model = YOLO("yolov8n.pt")
def calculate_iou(box1, box2):
    x1_int = max(box1[0], box2[0])
    y1_int = max(box1[1], box2[1])
    x2_int = min(box1[2], box2[2])
    y2_int = min(box1[3], box2[3])

    intersection_area = max(0, x2_int - x1_int + 1) * max(0, y2_int - y1_int + 1)
    intersection_area1 = max(0, x1_int - x2_int + 1) * max(0, y1_int - y2_int + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    box1_area1 = (box1[0] - box1[2] + 1) * (box1[1] - box1[3] + 1)
    box2_area1 = (box2[2] - box2[2] + 1) * (box2[1] - box2[3] + 1)

    iou = (intersection_area / float(box1_area + box2_area - intersection_area)) * 100
    iou1 = (intersection_area1 / float(box1_area1 + box2_area1 - intersection_area1)) * 100
    return max(iou, iou1)


# Function to process an image
def process(image_path):
    image = cv2.imread(image_path)

    # Detection and processing loop
    results = model(image, stream=True,)
    for r in results:
        boxes = r.boxes
        overs = []

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            classs = box.cls[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            classs = int(classs)

            if classs in [2, 3, 4,5,6, 7, 8]:
                over = [x1, y1, x2, y2]
                res = [0.000000000000000000001]
                if len(overs) >= 1:
                    for i in overs:
                        res.append(calculate_iou(i, over))
                overs.append(over)
                x3 = int((x1 + x2) / 2)
                y3 = int((y1 + y2) / 2)

                if max(res) <= 30:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.circle(image, (x3, y3), 3, (0, 0, 255))

                    crop = image[y1:y2, x1:x2]

                    reader = easyocr.Reader(['en'])
                    results = reader.readtext(crop)
                    for (bbox, text, prob) in results:
                        alphanumeric_text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  # Keep only letters, numbers, and spaces
                        if alphanumeric_text:  # Check if the resulting text is not empty
                            (top_left, top_right, bottom_right, bottom_left) = bbox
                            top_left = tuple(map(int, top_left))
                            bottom_right = tuple(map(int, bottom_right))
                            cv2.rectangle(crop, top_left, bottom_right, (0, 255, 0), 2)
                            cv2.putText(crop, alphanumeric_text, (top_left[0], top_left[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            #print(f"Text: {alphanumeric_text}, Probability: {prob:.2f}")

    # Display the result
    cv2.imshow('Original and Processed Images', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Process all JPG files in the 'DATA/' folder with extensions .jpg, .jpeg, and .webp
folder_path = 'DATA/'
image_extensions = ['.jpg', '.jpeg', '.webp']
jpg_files = [file for file in os.listdir(folder_path) if any(file.lower().endswith(ext) for ext in image_extensions)]

for jpg_file in jpg_files:
    file_path = os.path.join(folder_path, jpg_file)
    process(file_path)
