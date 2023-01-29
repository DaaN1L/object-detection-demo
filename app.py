import cv2
import numpy as np
import streamlit as st

from predictor import Predictor


def get_model(path_to_model):
    return Predictor(path_to_model)


def predict(model, img, conf, iou):
    return model.predict_and_annotate(img, conf=conf, iou=iou)


def build_sidebar():
    with st.sidebar:
        st.title("Setting up the model ⚙️")
        st.write("---")
        st.selectbox("Using model (may affect performance)", ["nano", "small", "medium", "large"], key="model")
        st.slider("Object confidence threshold for detection", 0.0, 1.0, 0.25, 0.01, key="conf")
        st.slider("Intersection over union threshold for NMS", 0.0, 1.0, 0.6, 0.01, key="iou")


def get_image_from_buffer(buf):
    bytes_data = buf.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    return img


def main():
    build_sidebar()
    model_path = "models/yolov8{}.pt".format(st.session_state.model[0])
    model = get_model(model_path)

    st.write('''
    # Object Detection Demo
    #### This is a demo of the deployment of the latest YOLOv8 models.
    ---
    ##### To get started, upload a photo or use your webcam. You can set model parameters via sidebar.''')

    uploaded_file_buffer = st.file_uploader("FILE LOADER", type=["png", "jpg", "jpeg"], label_visibility="hidden")
    uploaded_file_type = uploaded_file_buffer.type.split("/")[0] if uploaded_file_buffer is not None else None
    use_webcam = st.checkbox("Use webcam")

    fr = st.empty()
    if use_webcam:
        st.write("")
        cap = cv2.VideoCapture(-1)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            res = predict(model, frame, st.session_state.conf, st.session_state.iou)
            fr.image(res, channels="BGR")
        else:
            st.warning("No cameras available")
    elif uploaded_file_type == "image":
        img = get_image_from_buffer(uploaded_file_buffer)
        res = predict(model, img, st.session_state.conf, st.session_state.iou)
        st.image(res, channels="BGR")


if __name__ == '__main__':
    main()
