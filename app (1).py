import streamlit as st
import torch
import cv2
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from model import ViTColorization

st.set_page_config(page_title="Image Colorization", layout="wide")

# ------------------------------------------------
# STYLE
# ------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color:#f0f8ff;
}

.stButton>button {
    background-color:#007BFF;
    color:white;
    border-radius:8px;
    height:3em;
    width:200px;
    font-weight:bold;
}

.stButton>button:hover {
    background-color:#0056b3;
}

/* Compare button fix */
.compare-button button {
    width:320px !important;
    white-space: nowrap !important;
    overflow: hidden;
    text-overflow: ellipsis;
    background-color:#007BFF;
    color:white;
    font-weight:bold;
    border-radius:8px;
    display:block;
    margin:auto;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# PAGE STATE
# ------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "main"

if "image" not in st.session_state:
    st.session_state.image = None

if "output" not in st.session_state:
    st.session_state.output = None

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

device = torch.device("cpu")

model = ViTColorization().to(device)
model.load_state_dict(torch.load("vit_best_model.pth", map_location=device))
model.eval()

# ------------------------------------------------
# COLORIZATION FUNCTION
# ------------------------------------------------

def colorize(img):

    img = cv2.resize(img,(224,224))
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    lab = rgb2lab(img_rgb)
    L = lab[:,:,0]

    L_input = torch.tensor(L/100.0).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        ab = model(L_input)[0]

    ab = ab.numpy().transpose(1,2,0) * 128 * 1.8

    lab_output = np.zeros((224,224,3))
    lab_output[:,:,0] = L
    lab_output[:,:,1:] = ab

    rgb_output = lab2rgb(lab_output)
    rgb_output = np.clip(rgb_output,0,1)

    return img_rgb, rgb_output


# ==========================================================
# MAIN PAGE
# ==========================================================

if st.session_state.page == "main":

    st.markdown(
    """
    <h1 style='text-align:center;'>🎨 Image Colorization using Deep Learning</h1>
    <p style='text-align:center; font-size:18px; color:#1a73e8;'>
    Upload grayscale image and click <b>Colorize Image</b>
    </p>
    """,
    unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

    if uploaded_file:
        st.session_state.image = uploaded_file

    col1,col2 = st.columns(2)

    colorize_btn = col1.button("🎨 Colorize Image")
    clear_btn = col2.button("🧹 Clear")

    if clear_btn:
        st.session_state.image = None
        st.session_state.output = None
        st.rerun()

    if colorize_btn and st.session_state.image:

        image = Image.open(st.session_state.image)
        image = np.array(image)

        if len(image.shape)==2:
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

        input_img, output_img = colorize(image)

        st.session_state.input_img = input_img
        st.session_state.output = output_img

    # ------------------------------------------------
    # SHOW RESULTS
    # ------------------------------------------------

    if st.session_state.output is not None:

        input_img = st.session_state.input_img
        output_img = st.session_state.output

        col1,col2 = st.columns(2)

        with col1:
            st.markdown("### Input Image")
            st.image(input_img,width=450)

        with col2:
            st.markdown("### Colorized Image")
            st.image(output_img,width=450)

        # ------------------------------------------------
        # DOWNLOAD BUTTON
        # ------------------------------------------------

        output_uint8 = (output_img*255).astype(np.uint8)

        st.download_button(
            "⬇ Download Colorized Image",
            data=cv2.imencode(".png",output_uint8)[1].tobytes(),
            file_name="colorized_output.png",
            mime="image/png"
        )

        # ------------------------------------------------
        # METRICS DROPDOWN
        # ------------------------------------------------

        img_norm = input_img.astype(np.float32)/255.0

        psnr_value = psnr(img_norm, output_img, data_range=1.0)
        ssim_value = ssim(img_norm, output_img, channel_axis=2, data_range=1.0)

        accuracy = ssim_value * 100

        with st.expander("📊SHOW EVALUATION METRICS"):

            c1,c2,c3 = st.columns(3)

            with c1:
                st.metric("PSNR",f"{psnr_value:.2f}")

            with c2:
                st.metric("SSIM",f"{ssim_value:.4f}")

            with c3:
                st.metric("Accuracy",f"{accuracy:.2f}%")

        st.divider()

        # ------------------------------------------------
        # COMPARE BUTTON
        # ------------------------------------------------

        st.markdown('<div class="compare-button">', unsafe_allow_html=True)

        if st.button("Compare with Original Image"):

            st.session_state.page = "compare"
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================================
# SECOND PAGE
# ==========================================================

elif st.session_state.page == "compare":

    st.title("Comparison with Original Image")

    input_img = st.session_state.input_img
    output_img = st.session_state.output

    gray = cv2.cvtColor(input_img,cv2.COLOR_RGB2GRAY)

    gt_file = st.file_uploader("Upload Original Colored Image")

    if gt_file:

        gt = Image.open(gt_file)
        gt = np.array(gt)
        gt = cv2.resize(gt,(224,224))

        col1,col2,col3 = st.columns(3)

        with col1:
            st.markdown("### Black & White Image")
            st.image(gray,width=450)

        with col2:
            st.markdown("### Model Output")
            st.image(output_img,width=450)

        with col3:
            st.markdown("### Original Image")
            st.image(gt,width=450)

    st.divider()

    if st.button("⬅ Back to Main Page"):

        st.session_state.page = "main"
        st.rerun()