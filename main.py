import streamlit as st
import numpy as np
import cv2
import torch

st.snow()


def detect_objects(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # img = cv2.imread(file_path)
    results = model([img])

    # Print results
    results.print()

    # Render results on the image
    results_img = np.squeeze(results.render())

    # Convert from RGB (used by PyTorch) to BGR (used by OpenCV)
    results_img = cv2.cvtColor(results_img, cv2.COLOR_RGB2BGR)
    st.image(results_img)

    st.success("Detection completed.")


st.title("Advanced Detection Using Deep Learning")
# Define data for the table
data = [
    {"Name": "Abdulrahman Mohamed Mahmoud Burham", "ID": 4221264},
    {"Name": "Alaa Muhamed Abu Zaid Ibrahim", "ID": 4221257},
    {"Name": "Momen Tarek Abdel Moneim Gaber", "ID": 4221022},
    {"Name": "Salma Abdel Fattah Al-Sayed Agwa", "ID": 4221082}
]

# Display the table
st.table(data)

uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg'])

model_path = r"Model/best (100).pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.eval()

if uploaded_file is not None:
    detect_objects(uploaded_file)
    st.balloons()
