from ultralytics import YOLO
import cv2
import math
import time
import datetime
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display
import numpy as np

# Get the current timestamp for output names
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# Define the file name with the timestamp
file_name = f'output_{timestamp}.jpg'
classnames = ['car', 'plate']
charclassnames = ['0','9','ب','د','ع','ع','ج','ق','ه','ن','س','1','م','ن','س','ص','ت','ط','و','ی','2'
                  ,'3','4','5','6','7','8']
source = "assets/video.mp4"
#load YOLOv8 model
model_object = YOLO("weights/best.pt")
model_char = YOLO("weights/yolov8n_char_new.pt")


cap = cv2.VideoCapture(source)
# Define the output video properties
output_videoname = f'output_{timestamp}.mp4'
output_imagename = f'output_{timestamp}.jpg'
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if total_frames > 1:
    video_writer = cv2.VideoWriter('output/' + output_videoname, fourcc, fps, (frame_width, frame_height))
    #do inference for video
    while cap.isOpened():
        success, img = cap.read()
        if success:
            #detect objects with yolov8s model
            tick = time.time()
            output = model_object(img, show=False, conf=0.7, stream=True)
            #extract bounding box and class names
            for i in output:
                bbox = i.boxes
                for box in bbox:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(img,(x1, y1), (x2, y2), (255, 0, 0), 3)
                    confs = math.ceil((box.conf[0]*100))/100
                    cls_names = int(box.cls[0])
                    if cls_names == 1:
                        cv2.putText(img, f'{confs}', (max(40, x2 + 5), max(40, y2 + 5)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 20, 255),thickness=1, lineType=cv2.LINE_AA)
                    elif cls_names == 0:
                        cv2.putText(img, f'{confs}', (max(40, x1), max(40, y1)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.6, color=(0, 20, 255),thickness=1, lineType=cv2.LINE_AA)
                    

                    #check plate to recognize characters with yolov8n model
                    if cls_names == 1:
                        char_display = []
                        #crop plate from frame
                        plate_img = img[ y1:y2, x1:x2]
                        #detect characters of plate with yolov8n model
                        plate_output = model_char(plate_img, conf=0.3)
                        tock_2 = time.time()
                        elapsed_time_2 = tock_2 - tick
                        
                        #extract bounding box and class names
                        bbox = plate_output[0].boxes.xyxy
                        cls = plate_output[0].boxes.cls
                        #make a dict and sort it from left to right to show the correct characters of plate
                        keys = cls.numpy().astype(int)
                        values =bbox[:, 0].numpy().astype(int)
                        dictionary = list(zip(keys, values))
                        sorted_list = sorted(dictionary, key=lambda x: x[1])
                        #convert all characters to a string
                        for i in sorted_list:
                            char_class = i[0]
                            char_display.append(charclassnames[char_class])
                        char_result = (''.join(char_display))
                        fps_text_2 = "FPS: {:.2f}".format(1/elapsed_time_2)
                        text_size, _ = cv2.getTextSize(fps_text_2, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            
                        #just show the correct characters in output
                        if len(char_display) == 8:
                            char_result = char_result[3:6] + char_result[6:8] + char_result[2] + char_result[0:2]
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            # Create a PIL image object from the NumPy array
                            pil_image = Image.fromarray(img_rgb)
                            # Create a drawing context
                            draw = ImageDraw.Draw(pil_image)
                            # Set the font
                            font = ImageFont.truetype("font/B-NAZANIN.TTF", size=32, encoding='unic')
                            # Reshape and reorder the text
                            reshaped_text = arabic_reshaper.reshape(char_result)
                            display_text = get_display(reshaped_text)
                            # Get the text size
                            text_size = draw.textsize(display_text, font=font)
                            # Draw the text
                            draw.text((max(40, x1 - 15), max(40, y1 - 5)), display_text, fill='red', font=font)

                            # Convert the PIL image to a NumPy array
                            numpy_array = np.array(pil_image)

                            # Convert the color space from RGB to BGR (OpenCV uses BGR by default)
                            bgr_image = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)

                        # else:
                        #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        #     # Create a PIL image object from the NumPy array
                        #     pil_image = Image.fromarray(img_rgb)
                        #     # Create a drawing context
                        #     draw = ImageDraw.Draw(pil_image)
                        #     # Set the font
                        #     font = ImageFont.truetype("font/B-NAZANIN.TTF", size=32, encoding='unic')
                        #     # Reshape and reorder the text
                        #     char_result = char_result[:3] + "گ" + char_result[6:]
                        #     reshaped_text = arabic_reshaper.reshape(char_result)
                        #     display_text = get_display(reshaped_text)
                        #     # Get the text size
                        #     text_size = draw.textsize(display_text, font=font)
                        #     # Draw the text
                        #     draw.text((max(40, x1 - 15), max(40, y1 - 5)), display_text, fill='red', font=font)

                        #     # Convert the PIL image to a NumPy array
                        #     numpy_array = np.array(pil_image)

                        #     # Convert the color space from RGB to BGR (OpenCV uses BGR by default)
                        #     bgr_image = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)

                    else:
                        bgr_image = img

                tock = time.time()
                elapsed_time = tock - tick
                fps_text = "FPS: {:.2f}".format(1/elapsed_time)
                text_size, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                fps_text_loc = (frame_width - text_size[0] - 10, text_size[1] + 10)
                # cv2.putText(img, fps_text , fps_text_loc, fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(10, 50, 255),thickness=2, lineType=cv2.LINE_AA)
            
            cv2.imshow('detection', bgr_image)
            video_writer.write(bgr_image)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

                        
else: #do inference for image
    output = model_object(source, show=False, conf=0.75)
    img = cv2.imread(source)
    #extract bounding box and class names
    for i in output:
        bbox = i.boxes
        for box in bbox:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img,(x1, y1), (x2, y2), (255, 0, 0), 1)
            confs = math.ceil((box.conf[0]*100))/100
            cls_names = int(box.cls[0])
            if cls_names == 1:
                cv2.putText(img, f'{confs}', (max(40, x2 + 5), max(40, y2 + 5)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 20, 255),thickness=1, lineType=cv2.LINE_AA)
            elif cls_names == 0:
                cv2.putText(img, f'{confs}', (max(40, x1), max(40, y1)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.6, color=(0, 20, 255),thickness=1, lineType=cv2.LINE_AA)

            #check plate to recognize characters with yolov8n model
            if cls_names == 1:
                char_display = []
                #crop plate from frame
                plate_img = img[y1:y2, x1:x2]
                #detect characters of plate with yolov8n model
                plate_output = model_char(plate_img, conf=0.4)
                
                #extract bounding box and class names
                bbox = plate_output[0].boxes.xyxy
                cls = plate_output[0].boxes.cls
                #make a dict and sort it from left to right to show the correct characters of plate
                keys = cls.numpy().astype(int)
                values =bbox[:, 0].numpy().astype(int)
                dictionary = list(zip(keys, values))
                sorted_list = sorted(dictionary, key=lambda x: x[1])
                #convert all characters to a string
                for i in sorted_list:
                    char_class = i[0]
                    #char_display.append(plate_output[0].names[char_class])
                    char_display.append(charclassnames[char_class])
                char_result = (''.join(char_display))
                char_result = char_result[3:6] + char_result[6:8] + char_result[2] + char_result[0:2]

                #just show the correct characters in output
                if len(char_display) == 8:
                    img = Image.open(source)
                    # Create a drawing context
                    draw = ImageDraw.Draw(img)
                    # Set the font
                    font = ImageFont.truetype("font/B-NAZANIN.TTF", size=32, encoding='unic')
                    # Reshape and reorder the text
                    reshaped_text = arabic_reshaper.reshape(char_result)
                    display_text = get_display(reshaped_text)
                    # Get the text size
                    text_size = draw.textsize(display_text, font=font)
                    # Draw the text
                    draw.text((max(40, x1 - 15), max(40, y1 - 5)), display_text, fill='red', font=font)

                    # Convert the PIL image to a NumPy array
                    numpy_array = np.array(img)

                    # Convert the color space from RGB to BGR (OpenCV uses BGR by default)
                    bgr_image = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)

                    # cv2.line(bgr_image, (max(40, x1 - 25 ), max(40, y1 - 10)), (x2 + 25 ,y1 - 10), (0, 0, 0), 20,lineType=cv2.LINE_AA)
                    # cv2.putText(img, char_result , (max(40, x1 - 15), max(40, y1 - 5)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(10, 50, 255),thickness=1, lineType=cv2.LINE_AA)

                else:
                    img = Image.open(source)
                    # Create a drawing context
                    draw = ImageDraw.Draw(img)
                    # Set the font
                    font = ImageFont.truetype("font/B-NAZANIN.TTF", size=32, encoding='unic')
                    # Reshape and reorder the text
                    char_result = char_result[:3] + "گ" + char_result[6:]
                    reshaped_text = arabic_reshaper.reshape(char_result)
                    display_text = get_display(reshaped_text)
                    # Get the text size
                    text_size = draw.textsize(display_text, font=font)
                    # Draw the text
                    draw.text((max(40, x1 - 15), max(40, y1 - 5)), display_text, fill='red', font=font)

                    # Convert the PIL image to a NumPy array
                    numpy_array = np.array(img)

                    # Convert the color space from RGB to BGR (OpenCV uses BGR by default)
                    bgr_image = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)


    cv2.imshow('detection', bgr_image)
    # img.show()
    # cv2.imwrite('output/' + output_imagename, img)
    # exit if 'q' is pressed
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
