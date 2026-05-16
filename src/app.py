import streamlit as st
import cv2 as cv
import numpy as np
import easyocr
from streamlit_cropper import st_cropper
from PIL import Image
#from streamlit_image_zoom import image_zoom
def ocr(src_img):
    if src_img is not None:
    # To read file as bytes:
        bytes_data = src_img.getvalue()
        np_img = np.frombuffer(bytes_data, np.uint8)
        img= cv.imdecode(np_img, cv.IMREAD_COLOR)
        rgb_img=cv.cvtColor(img, cv.COLOR_BGR2RGB)

        #CROPPING
        realtime_update = st.checkbox(label="Update in Real Time", value=True)
        box_color = st.color_picker(label="Box Color", value='#0000FF')
        aspect_choice = st.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
        aspect_dict = {
            "1:1": (1, 1),
            "16:9": (16, 9),
            "4:3": (4, 3),
            "2:3": (2, 3),
            "Free": None
        }
        aspect_ratio = aspect_dict[aspect_choice]
        if rgb_img is not None:
            #PIL image object
            opened_rgb = Image.fromarray(rgb_img)
            if not realtime_update:
                st.write("Double click to save crop")
            # Get a cropped image from the frontend
            cropped_img = st_cropper(opened_rgb, realtime_update=realtime_update, box_color=box_color,
                                        aspect_ratio=aspect_ratio)
            
            # Manipulate cropped image at will
            st.write("Cropped")
            st.image(cropped_img)
            array_cropped_img = np.array(cropped_img)
            cropped_gray=cv.cvtColor(array_cropped_img, cv.COLOR_RGB2GRAY)
            # 1. Blur the image (smooths out fine details)
            kernel_size = st.slider(label="Kernel size for Gaussian Blur", min_value=1, max_value=31, value=5, step=2)

            blurred = cv.GaussianBlur(cropped_gray, (kernel_size, kernel_size), 1.0)
            # Detect second-derivative edges
            laplacian_edges = cv.Laplacian(blurred, cv.CV_64F)

            # Convert back to 8-bit unsigned integer
            laplacian_8u = cv.convertScaleAbs(laplacian_edges)

            # Add edges back to original image to sharpen
            sharpened = cv.add(blurred, laplacian_8u)

            #_,th3 = cv.threshold(sharpened,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)

            st.image(sharpened)
            reader = easyocr.Reader(['en'], gpu=False)
            result=reader.readtext(sharpened, detail=0)
            #single click to zoom in, click and drag to move the zoomed image, and double-click to zoom out.
            #image_zoom(th3, mode="dragmove", size=(800, 600), keep_aspect_ratio=False, zoom_factor=4.0, increment=0.2)

            # Extract only the text element (index 1 of each item)
            st.write(result)
    else:
        st.info("Please upload before you proceed!")

input_method = st.radio(
    "Choose how to provide the image:",
    ("Upload a File", "Use Camera")
)
if input_method == "Upload a File":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])
    ocr(uploaded_file)
elif input_method == "Use Camera":
    enable = st.checkbox("Enable camera")
    picture = st.camera_input("Take a picture", disabled=not enable)
    ocr(picture)
