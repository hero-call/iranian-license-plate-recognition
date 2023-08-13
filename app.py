from pathlib import Path
import streamlit as st

import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam

# setting page layout
st.set_page_config(
    page_title="Persian Plate Recognition",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# main page heading
st.title("Persian Plate Recognition")

# sidebar
st.sidebar.header("DL Model Config")

# model options
task_type = st.sidebar.selectbox(
    "Select Task",
    ["Detection"]
)

# model_type = None
# if task_type == "Detection":
#     model_type = st.sidebar.selectbox(
#         "Select Model",
#         config.DETECTION_MODEL_LIST
#     )
# else:
#     st.error("Currently only 'Detection' function is implemented")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 30, 100, 50)) / 100


model_path_object = Path(config.DETECTION_MODEL_DIR, 'best.pt')
print(model_path_object)
model_path_char = Path(config.DETECTION_MODEL_DIR, 'yolov8n_char_new.pt')

 

# load pretrained DL model
try:

    
    model_object = load_model(model_path_object)
    model_char = load_model(model_path_char)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path_object}")

# image/video options
st.sidebar.header("Image/Video Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    config.SOURCES_LIST
)

source_img = None
if source_selectbox == config.SOURCES_LIST[0]: # Image
    infer_uploaded_image(confidence, model_object, model_char)
elif source_selectbox == config.SOURCES_LIST[1]: # Video
    infer_uploaded_video(confidence, model_object, model_char)
elif source_selectbox == config.SOURCES_LIST[2]: # Webcam
    infer_uploaded_webcam(confidence, model_object, model_char)
else:
    st.error("Currently only 'Image' and 'Video' source are implemented")