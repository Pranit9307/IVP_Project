import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pywt
from scipy.fftpack import dct, idct
import sys

st.set_page_config(layout="wide")
st.title("🖼️ Cotton Leaf Image Processing App")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select Section", ["Image Processing", "Compression"])

# Upload or example image
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)  # Keep RGB format
else:
    st.sidebar.info("No image uploaded. Using default sample.")
    image = Image.open("sample.png")  # Keep RGB format

# Display original RGB image with increased size
st.image(image, caption="Original Image (RGB)", width=500)

# Convert to grayscale for processing
img_gray = np.array(image.convert("L"))
img_rgb = np.array(image)

# Function to get size in bytes
def get_size(obj):
    return sys.getsizeof(obj)

if page == "Image Processing":
    # Image Processing Filters
    st.sidebar.header("Image Processing Filters")
    processing_filters = {
        "Point Processing": {
            "Image Negation": False,
            "Thresholding": False,
            "Bit Plane Slicing": False
        },
        "Enhancement": {
            "Histogram Equalization": False,
            "Smoothing (Gaussian Blur)": False,
            "Sharpening (Laplacian)": False,
            "High Boost Filter": False
        }
    }

    # Create checkboxes for each category
    for category, filters in processing_filters.items():
        st.sidebar.subheader(category)
        for filter_name in filters:
            processing_filters[category][filter_name] = st.sidebar.checkbox(filter_name)

    # --- Point Processing ---
    if any(processing_filters["Point Processing"].values()):
        st.header("🧩 Point Processing")
        
        if processing_filters["Point Processing"]["Image Negation"]:
            st.subheader("Image Negation")
            neg_img = 255 - img_gray
            st.image(neg_img, width=500)
            st.info(f"Original size: {get_size(img_gray)} bytes, Processed size: {get_size(neg_img)} bytes")
        
        if processing_filters["Point Processing"]["Thresholding"]:
            st.subheader("Thresholding")
            _, threshed = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
            st.image(threshed, width=500)
            st.info(f"Original size: {get_size(img_gray)} bytes, Processed size: {get_size(threshed)} bytes")
        
        if processing_filters["Point Processing"]["Bit Plane Slicing"]:
            bit = st.slider("Select Bit Plane (0-7)", 0, 7, 7)
            bit_img = ((img_gray >> bit) & 1) * 255
            st.subheader("Bit Plane Slicing")
            st.image(bit_img, width=500)
            st.info(f"Original size: {get_size(img_gray)} bytes, Processed size: {get_size(bit_img)} bytes")

    # --- Enhancement ---
    if any(processing_filters["Enhancement"].values()):
        st.header("✨ Image Enhancement")
        
        if processing_filters["Enhancement"]["Histogram Equalization"]:
            st.subheader("Histogram Equalization")
            equalized = cv2.equalizeHist(img_gray)
            st.image(equalized, width=500)
            st.info(f"Original size: {get_size(img_gray)} bytes, Processed size: {get_size(equalized)} bytes")
        
        if processing_filters["Enhancement"]["Smoothing (Gaussian Blur)"]:
            st.subheader("Smoothing (Gaussian Blur)")
            blurred = cv2.GaussianBlur(img_gray, (5,5), 0)
            st.image(blurred, width=500)
            st.info(f"Original size: {get_size(img_gray)} bytes, Processed size: {get_size(blurred)} bytes")
        
        if processing_filters["Enhancement"]["Sharpening (Laplacian)"]:
            st.subheader("Sharpening (Laplacian)")
            kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
            sharp = cv2.filter2D(img_gray, -1, kernel)
            st.image(sharp, width=500)
            st.info(f"Original size: {get_size(img_gray)} bytes, Processed size: {get_size(sharp)} bytes")
        
        if processing_filters["Enhancement"]["High Boost Filter"]:
            st.subheader("High Boost Filter")
            k = st.slider("Select Boost Factor (k)", 1.0, 3.0, 1.5, 0.1)
            blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
            mask = img_gray.astype(float) - blurred.astype(float)
            high_boost = img_gray.astype(float) + k * mask
            high_boost = np.clip(high_boost, 0, 255).astype(np.uint8)
            st.image(high_boost, width=500)
            st.info(f"Original size: {get_size(img_gray)} bytes, Processed size: {get_size(high_boost)} bytes")

else:  # Compression page
    st.sidebar.header("Compression Techniques")
    compression_filters = {
        "Run-Length Encoding": False,
        "DCT Compression": False,
        "Haar Transform": False
    }

    # Create checkboxes for compression techniques
    for filter_name in compression_filters:
        compression_filters[filter_name] = st.sidebar.checkbox(filter_name)

    if any(compression_filters.values()):
        st.header("🗜️ Compression Techniques")
        
        if compression_filters["Run-Length Encoding"]:
            st.subheader("Run-Length Encoding (RLE)")
            def rle_encode(im):
                flat = im.flatten()
                values, counts = [], []
                count = 1
                for i in range(1, len(flat)):
                    if flat[i] == flat[i-1]:
                        count += 1
                    else:
                        values.append(flat[i-1])
                        counts.append(count)
                        count = 1
                values.append(flat[-1])
                counts.append(count)
                return list(zip(values, counts))

            resized_img = cv2.resize(img_gray, (64,64))
            rle = rle_encode(resized_img)
            st.text(f"Sample RLE (First 20): {rle[:20]}")
            st.info(f"Original size: {get_size(resized_img)} bytes, Compressed size: {get_size(rle)} bytes")
            st.info(f"Compression ratio: {get_size(resized_img)/get_size(rle):.2f}x")
        
        if compression_filters["DCT Compression"]:
            st.subheader("DCT Compression (8x8 block)")
            # Normalize the image to [0, 1] range
            block = cv2.resize(img_gray, (256,256))[:8,:8].astype(float) / 255.0
            
            # Apply DCT
            dct_trans = dct(dct(block.T, norm='ortho').T, norm='ortho')
            
            # Apply inverse DCT
            idct_img = idct(idct(dct_trans.T, norm='ortho').T, norm='ortho')
            
            # Convert back to [0, 255] range for display
            dct_display = np.abs(dct_trans) * 255.0
            idct_display = np.clip(idct_img * 255.0, 0, 255).astype(np.uint8)

            col7, col8 = st.columns(2)
            with col7:
                st.image(dct_display, caption="DCT Transformed", width=400)
            with col8:
                st.image(idct_display, caption="Inverse DCT", width=400)
            
            st.info(f"Original block size: {get_size(block)} bytes")
            st.info(f"DCT coefficients size: {get_size(dct_trans)} bytes")
            st.info(f"Compression ratio: {get_size(block)/get_size(dct_trans):.2f}x")
        
        if compression_filters["Haar Transform"]:
            st.subheader("Haar Transform")
            # Resize image to power of 2 for Haar transform
            size = 2**int(np.log2(min(img_gray.shape)))
            img_resized = cv2.resize(img_gray, (size, size))
            
            # Normalize the image to [0, 1] range
            img_normalized = img_resized.astype(float) / 255.0
            
            try:
                # Apply Haar transform with proper error handling
                coeffs = pywt.dwt2(img_normalized, 'haar')
                cA, (cH, cV, cD) = coeffs
                
                # Ensure coefficients are within valid range
                cA = np.clip(cA, 0, 1)
                cH = np.clip(cH, -1, 1)
                cV = np.clip(cV, -1, 1)
                cD = np.clip(cD, -1, 1)
                
                # Convert coefficients back to [0, 255] range for display
                cA_display = (cA * 255.0).astype(np.uint8)
                cH_display = ((np.abs(cH) + 1) * 127.5).astype(np.uint8)
                
                col9, col10 = st.columns(2)
                with col9:
                    st.image(cA_display, caption="Approximation (cA)", width=400)
                with col10:
                    st.image(cH_display, caption="Horizontal Detail (cH)", width=400)
                
                st.info(f"Original image size: {get_size(img_normalized)} bytes")
                st.info(f"Approximation coefficients size: {get_size(cA)} bytes")
                st.info(f"Compression ratio: {get_size(img_normalized)/get_size(cA):.2f}x")
                
            except Exception as e:
                st.error(f"Error in Haar transform: {str(e)}")
                st.info("Try using a different image size or format")

if any(processing_filters.values()) if page == "Image Processing" else any(compression_filters.values()):
    st.success("✅ Selected filters applied successfully!")
