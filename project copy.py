import cv2
import os
from ultralytics import YOLO
import numpy as np
import easyocr
import re
import mahotas
from PIL import ImageFilter,Image



# Function to calculate Intersection over Union (IoU)
model = YOLO("yolov8n.pt")
import cv2
kernel = np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])

def count_white_black_pixels(image):
    # Check if the image is grayscale
    if len(image.shape) == 2:
        img_gray = image
    elif len(image.shape) == 3:
        # Convert the image to grayscale
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Handle unsupported number of channels
        raise ValueError("Unsupported number of channels in the input image")

    # Convert the grayscale image to a binary image
    _, img_binary = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)

    # Count the number of white and black pixels
    white_pixels = cv2.countNonZero(img_binary)
    black_pixels = img_binary.size - white_pixels

    return white_pixels, black_pixels
def img_resize(original,resized,lst):
    x=lst[0]
    y=lst[1]
    r=(int(x * (original.shape[1] / resized.shape[0])),int(y * (original.shape[0] / resized.shape[1])))
    return r

def words(crop):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(crop)
    alphanumeric_text=None
    for (bbox, text, prob) in results:
        alphanumeric_text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  # Keep only letters, numbers, and spaces
    return alphanumeric_text



            # print(f"Text: {alphanumeric_text}, Probability: {prob:.2f}")
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

def get_dominant_color(cropped_box):
    # Calculate the average color of the box
    hsv = cv2.cvtColor(cropped_box, cv2.COLOR_BGR2HSV)
    return hsv
def imageinpr(org_image):
    #cv2.imshow("org_image",org_image)
    target_size = (300, 300)
    image = cv2.resize(org_image, target_size)
    #cv2.imshow("image_re",image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 100, 205, cv2.THRESH_BINARY)
    image_pillow = Image.fromarray(thresh)
    imagefiltered = image_pillow.filter(ImageFilter.MaxFilter(5))
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    height, _ = image.shape[:2]
    contours = [c for c in contours if cv2.contourArea(c) <= 3000 and cv2.contourArea(c) >= 10]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    vechi_area=image.shape[0]*image.shape[1]
    con=0
    x1,y1,x2,y2=0,0,0,0
    for idx, c in enumerate(contours):

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(c)

        # Get top-left, top-right, bottom-left, bottom-right points
        top_left = (x, y)
        top_right = (x + w, y)
        bottom_left = (x, y + h)
        bottom_right = (x + w, y + h)
        cropped_box = image[y:y + h, x:x + w]
        area_of_marked=cropped_box.shape[0]*cropped_box.shape[1]

        x_i=image.shape[0]

        if y > (image.shape[0]/4) and y+h < ((x_i)-(x_i/8)):
            if area_of_marked<=(vechi_area/7) :
                if area_of_marked >= 500:

                    if w*1.8>= h:
                        if w<=h*5:
                            if con<=0:




                                x1,y1=top_left[0],top_left[1]
                                x2,y2=bottom_right[0],bottom_right[1]

                                con+=1
    points_in_resized_image=[[x1,y1],[x2,y2]]
    points_in_original_image=[img_resize(org_image,image,[x,y]) for x,y in points_in_resized_image]


    return points_in_original_image



def imagequality(crop,gray):

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = np.std(crop)
    gray_contrast = np.std(gray)
    colorfulness = np.mean(np.std(crop, axis=0))
    return [sharpness,contrast,gray_contrast,colorfulness]

def colorquality(image,gray):
    colorfulness = np.mean(np.std(image, axis=0))
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    mean_color = np.mean(image, axis=(0, 1))
    std_color = np.std(image, axis=(0, 1))
    return [colorfulness,hist,mean_color,std_color]
def texturequality(image,gray):
    textures = mahotas.features.haralick(gray).mean(axis=0)
    equalized = cv2.equalizeHist(gray)

    gamma = 1.5
    corrected = np.clip(image ** gamma, 0, 255).astype(np.uint8)
    noise = np.std(image - cv2.GaussianBlur(image, (0, 0), 5))
    return [textures,equalized,corrected,noise]

def highlight(image):
    image_float = image.astype(float)

    # Extract the intensity values for each channel
    blue_channel = image_float[:, :, 0]
    green_channel = image_float[:, :, 1]
    red_channel = image_float[:, :, 2]

    # Apply a gamma correction to enhance the highlights (adjust the gamma value as needed)
    gamma = 1.05
    blue_channel_highlight = np.power(blue_channel, 1 / gamma)
    green_channel_highlight = np.power(green_channel, 1 / gamma)
    red_channel_highlight = np.power(red_channel, 1 / gamma)

    # Combine the enhanced channels
    enhanced_image = np.stack((blue_channel_highlight, green_channel_highlight, red_channel_highlight), axis=-1)

    # Convert back to uint8 for display
    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)
    return(enhanced_image)
# Function to process an image
def process(image):
    frame=[]
    org=image
    target_size=(300, 300)
    value=[]
    frame_left=[]
    np_left=[]
    np_right=[]

    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray_frame)


    image = cv2.resize(image, target_size) #1st resize
    if mean_intensity > 127:
        dark_filtered_image = highlight(image)
        image=dark_filtered_image
    results = model(image, stream=True)#croped value
    for r in results:
        boxes = r.boxes
        overs = []
        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]#each vechicle box
            classs = box.cls[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            classs = int(classs)
            h1,w1=x2-x1,y2-y1

            if classs in [2, 3, 4, 5, 6, 7, 8]:
                over = [x1, y1, x2, y2]

                res = [0.000000000000000000001]
                if len(overs) >= 1:
                    for i in overs:
                        res.append(calculate_iou(i, over))
                overs.append(over)
                x3 = int((x1 + x2) / 2)
                y3 = int((y1 + y2) / 2)

                if max(res) <= 30:
                    frame_left.append([x1, y1])
                    crop = image[y1:y2, x1:x2]


                    crop = cv2.filter2D(src=crop, ddepth=-1, kernel=kernel)
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    proimage = imageinpr(crop)
                    np_left.append(proimage)
                    value.append(proimage)


        for i in range(len(frame_left)):
            left=np_left[i][0]
            right=np_left[i][1]
            l,m=frame_left[i]
            F=img_resize(org,image,[left[0] + l, left[1] + m])
            R=img_resize(org,image,[right[0] + l, right[1] + m])
            frame.append([F,R])






    return frame









# Process all JPG files in the 'DATA/' folder with extensions .jpg, .jpeg, and .webp
folder_path = 'DATA/'
image_extensions = ['.jpg', '.jpeg', '.webp']
jpg_files = [file for file in os.listdir(folder_path) if any(file.lower().endswith(ext) for ext in image_extensions)]

for jpg_file in jpg_files:
    file_path = os.path.join(folder_path, jpg_file)
    image = cv2.imread(file_path)
    values=process(image)
    for x,y in values:
        cv2.rectangle(image,x,y,(0,255,0),2)
    cv2.imshow("image",image)
    cv2.waitKey(0)


