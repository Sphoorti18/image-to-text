import streamlit as st
import cv2 as cv
import numpy as np
import easyocr
from streamlit_cropper import st_cropper
from PIL import Image
#from streamlit_image_zoom import image_zoom

@st.cache_resource
def load_reader(language):
    return easyocr.Reader([language], gpu=False)
def ocr(src_img):
    if src_img is not None:
    # To read file as bytes:
        bytes_data = src_img.getvalue()
        np_img = np.frombuffer(bytes_data, np.uint8)
        img= cv.imdecode(np_img, cv.IMREAD_COLOR)
        rgb_img=cv.cvtColor(img, cv.COLOR_BGR2RGB)

        #CROPPING
        with st.sidebar:
            st.subheader("Cropping tool")
            realtime_update = st.checkbox(label="Update in Real Time", value=True)
            box_color = st.color_picker(label="Box Color", value='#0000FF')
            aspect_choice = st.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])

            kernel_size = st.slider(label="Kernel size for Gaussian Blur", min_value=1, max_value=9, value=3, step=2)
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
            
            
            array_cropped_img = np.array(cropped_img)
            cropped_gray=cv.cvtColor(array_cropped_img, cv.COLOR_RGB2GRAY)
            # restoration / denoising
            denoised = cv.fastNlMeansDenoising(cropped_gray)
            # 1. Blur the image (smooths out fine details)

            blurred = cv.GaussianBlur(denoised, (kernel_size, kernel_size), 0)
            # Detect second-derivative edges
            #laplacian_edges = cv.Laplacian(blurred, cv.CV_64F)

            # Convert back to 8-bit unsigned integer
            #laplacian_8u = cv.convertScaleAbs(laplacian_edges)

            # Add edges back to original image to sharpen
            #sharpened = cv.add(blurred, laplacian_8u)

            _, otsu = cv.threshold(
                blurred,
                0,
                255,
                cv.THRESH_BINARY + cv.THRESH_OTSU
            )

            #_,th3 = cv.threshold(sharpened,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
            # Manipulate cropped image at will
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Cropped")
                st.image(cropped_img)
            with col2:
                st.subheader("Processed")
                st.image(otsu)
            lang_dict = {"English": "en", "Korean": "ko", "Japanese": "ja", "Hindi": "hi"}
            option = st.selectbox(
                    "Language",
                    options=list(lang_dict.keys()),
                    placeholder="Select a language",
                    accept_new_options=False,
                )
            if option is not None:
                selected_value = lang_dict[option]
            reader = load_reader(selected_value)
            result=reader.readtext(otsu, detail=0, paragraph=True)
            #single click to zoom in, click and drag to move the zoomed image, and double-click to zoom out.
            #image_zoom(th3, mode="dragmove", size=(800, 600), keep_aspect_ratio=False, zoom_factor=4.0, increment=0.2)

            # Extract only the text element (index 1 of each item)
            if result:
                st.write(result)
                full_text = "\n".join(result)
                st.download_button("Download Text", full_text, file_name="ocr.txt")
            else:
                st.warning("No text detected. Try adjusting the crop or kernel size.")
    else:
        st.info("Please upload before you proceed!")

st.set_page_config(
    page_title="Image to Text OCR",
    page_icon="🔍",
    layout="wide"
)
with st.popover("Instructions"):
    st.markdown("""
    ### How to Use This OCR Tool 🔍

    **Step 1 – Provide an Image**
    - Choose **Upload a File** to upload a JPG or PNG from your device.
    - Or choose **Use Camera** → enable the camera → snap a photo directly.

    **Step 2 – Crop the Region of Interest**
    In the sidebar:
    - Pick an **Aspect Ratio** (or choose *Free* for freeform).
    - Drag the crop box over the text you want to extract.
    - Toggle **Update in Real Time** to see the processed image update live (or double-click to confirm the crop when it's off).
    - Adjust the **Box Color** to make the crop handle easier to see.

    **Step 3 – Tune Pre-processing**
    - Use the **Kernel Size** slider (sidebar) to control Gaussian blur strength.  
      A smaller value (e.g. 1–3) works well for sharp, printed text; larger values help smooth noisy or handwritten text.
    - Compare the **Cropped** vs **Processed** previews to check quality before running OCR.

    **Step 4 – Select a Language**
    - Choose the language of the text in your image: **English, Korean, Japanese, or Hindi**.

    **Step 5 – Get Your Text**
    - The extracted text appears automatically below the previews.
    - Click **Download Text** to save it as a `.txt` file.

    ---
    💡 **Tips for best results:**
    - Crop as tightly as possible around the text.
    - Ensure good lighting and contrast in the original image.
    - If no text is detected, try a different kernel size or re-crop.
    """)
    
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
