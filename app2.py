import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
from PIL import Image
import pywt
from scipy.fftpack import dct, idct
import sys
import time
import base64
from PIL import ImageDraw
import random  # Added for random sampling
import zipfile
import tempfile
import os
import concurrent.futures
from functools import partial


# Page configuration
st.set_page_config(
    page_title="Cotton Leaf Analysis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2e7d32;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #388e3c;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .filter-section {
        background-color: #f1f8e9;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stAlert {
        border-radius: 8px;
    }
    .info-text {
        font-size: 0.9rem;
        color: #555;
    }
    .result-section {
        background-color: #f9f9f9;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Custom header
st.markdown("<h1 class='main-header'>🌿 Cotton Leaf Image Analysis Tool</h1>", unsafe_allow_html=True)

# Function to get size in bytes
def get_size(obj):
    return sys.getsizeof(obj)

# Function to create download button
def get_download_link(img, filename="processed_image.png", text="Download Processed Image"):
    buffered = io.BytesIO()
    Image.fromarray(img).save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Function to analyze image
def analyze_image(img_gray):
    results = {}
    
    # Basic statistics
    results["Mean"] = np.mean(img_gray)
    results["Std Dev"] = np.std(img_gray)
    results["Min"] = np.min(img_gray)
    results["Max"] = np.max(img_gray)
    
    # Calculate histogram
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    
    # Detect edges (count of edge pixels)
    edges = cv2.Canny(img_gray, 100, 200)
    results["Edge Pixels"] = np.count_nonzero(edges)
    
    # Texture analysis (GLCM contrast)
    gradient_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    results["Texture Complexity"] = np.mean(gradient_magnitude)
    
    return results, hist, edges

# Sidebar navigation with better organization
with st.sidebar:
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/5/5e/Cotton_plant_boll_1_%28Cropped%29.jpg", width=100)
    st.sidebar.title("Control Panel")
    
    # Navigation
    st.sidebar.subheader("📋 Navigation")
    page = st.sidebar.radio("Select Section", ["Image Processing", "Compression", "Analysis", "Background Removal", "Color Enhancement"])
    
    # Image upload
    st.sidebar.subheader("🖼️ Image Input")
    uploaded_file = st.sidebar.file_uploader("Upload Cotton Leaf Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.sidebar.success("✅ Image uploaded successfully!")
    else:
        st.sidebar.info("ℹ️ No image uploaded. Using default sample.")
        # Create a dummy image
        image = Image.new('RGB', (512, 512), color=(240, 240, 240))
        draw = ImageDraw.Draw(image)
        draw.text((200, 250), "No Image Loaded", fill=(0, 0, 0))
        st.sidebar.warning("⚠️ Please upload an image for best results")
    
    # Image info
    st.sidebar.subheader("📊 Image Info")
    if uploaded_file:
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.1f} KB",
            "Width": image.width,
            "Height": image.height
        }
        for key, value in file_details.items():
            st.sidebar.text(f"{key}: {value}")

# Show original image in main area
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("<h3>Original Image</h3>", unsafe_allow_html=True)
    st.image(image, caption="Original Image", use_column_width=True)

# Convert to grayscale for processing
img_gray = np.array(image.convert("L"))
img_rgb = np.array(image)

# Display histogram
with col2:
    st.markdown("<h3>Image Histogram</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(img_gray.ravel(), bins=256, range=[0, 256], color='#4CAF50', alpha=0.7)
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Add image statistics
    results, hist, edges = analyze_image(img_gray)
    st.markdown("<h3>Image Statistics</h3>", unsafe_allow_html=True)
    stats_df = pd.DataFrame({
        'Metric': list(results.keys()),
        'Value': list(results.values())
    })
    st.dataframe(stats_df, hide_index=True)

# Image Processing Page
if page == "Image Processing":
    st.markdown("<h2 class='sub-header'>🧩 Image Processing Techniques</h2>", unsafe_allow_html=True)
    
    # Create tabs for different categories
    tab1, tab2, tab3 = st.tabs(["Basic Processing", "Enhancement", "Color Processing"])
    
    with tab1:
        st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
        st.subheader("Point Processing")
        
        processing_option = st.selectbox(
            "Select Point Processing Technique",
            ["None", "Image Negation", "Thresholding", "Bit Plane Slicing", "Brightness Adjustment", "Contrast Adjustment"]
        )
        
        if processing_option == "Image Negation":
            neg_img = 255 - img_gray
            st.image(neg_img, caption="Image Negation", use_column_width=True)
            st.markdown(get_download_link(neg_img), unsafe_allow_html=True)
            
        elif processing_option == "Thresholding":
            threshold_value = st.slider("Threshold Value", 0, 255, 127)
            threshold_type = st.selectbox("Threshold Type", ["Binary", "Binary Inverted", "Truncate", "To Zero", "To Zero Inverted"])
            
            threshold_types = {
                "Binary": cv2.THRESH_BINARY,
                "Binary Inverted": cv2.THRESH_BINARY_INV,
                "Truncate": cv2.THRESH_TRUNC,
                "To Zero": cv2.THRESH_TOZERO,
                "To Zero Inverted": cv2.THRESH_TOZERO_INV
            }
            
            _, threshed = cv2.threshold(img_gray, threshold_value, 255, threshold_types[threshold_type])
            st.image(threshed, caption=f"Thresholding ({threshold_type})", use_column_width=True)
            st.markdown(get_download_link(threshed), unsafe_allow_html=True)
            
        elif processing_option == "Bit Plane Slicing":
            bit = st.slider("Select Bit Plane (0-7)", 0, 7, 7)
            bit_img = ((img_gray >> bit) & 1) * 255
            st.image(bit_img, caption=f"Bit Plane {bit}", use_column_width=True)
            st.markdown(get_download_link(bit_img), unsafe_allow_html=True)
            
            # Show all bit planes as an additional feature
            if st.checkbox("Show All Bit Planes"):
                cols = st.columns(4)
                for i, col in enumerate(cols):
                    bit_img = ((img_gray >> i) & 1) * 255
                    col.image(bit_img, caption=f"Bit {i}", use_column_width=True)
                    
                cols = st.columns(4)
                for i, col in enumerate(cols):
                    bit_img = ((img_gray >> (i+4)) & 1) * 255
                    col.image(bit_img, caption=f"Bit {i+4}", use_column_width=True)
                    
        elif processing_option == "Brightness Adjustment":
            beta = st.slider("Brightness (-100 to 100)", -100, 100, 0)
            bright_img = np.clip(img_gray.astype(np.int16) + beta, 0, 255).astype(np.uint8)
            st.image(bright_img, caption=f"Brightness Adjustment ({beta})", use_column_width=True)
            st.markdown(get_download_link(bright_img), unsafe_allow_html=True)
            
        elif processing_option == "Contrast Adjustment":
            alpha = st.slider("Contrast Factor (0.1 to 3.0)", 0.1, 3.0, 1.0, 0.1)
            contrast_img = np.clip(img_gray.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
            st.image(contrast_img, caption=f"Contrast Adjustment ({alpha})", use_column_width=True)
            st.markdown(get_download_link(contrast_img), unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
        st.subheader("Enhancement Techniques")
        
        enhancement_option = st.selectbox(
            "Select Enhancement Technique",
            ["None", "Histogram Equalization", "Adaptive Histogram Equalization", "Gaussian Blur", 
             "Sharpening", "High Boost Filter", "Median Filter", "Bilateral Filter"]
        )
        
        if enhancement_option == "Histogram Equalization":
            equalized = cv2.equalizeHist(img_gray)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(equalized, caption="Histogram Equalized", use_column_width=True)
                st.markdown(get_download_link(equalized), unsafe_allow_html=True)
            
            with col2:  
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.hist(equalized.ravel(), bins=256, range=[0, 256], color='#2196F3', alpha=0.7)
                ax.set_xlabel('Pixel Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
        elif enhancement_option == "Adaptive Histogram Equalization":
            clahe = cv2.createCLAHE(clipLimit=st.slider("Clip Limit", 1.0, 5.0, 2.0, 0.1), 
                                   tileGridSize=(st.slider("Tile Size", 2, 16, 8, 2), 
                                                st.slider("Tile Size", 2, 16, 8, 2)))
            adaptive_eq = clahe.apply(img_gray)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(adaptive_eq, caption="Adaptive Histogram Equalization", use_column_width=True)
                st.markdown(get_download_link(adaptive_eq), unsafe_allow_html=True)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.hist(adaptive_eq.ravel(), bins=256, range=[0, 256], color='#9C27B0', alpha=0.7)
                ax.set_xlabel('Pixel Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
        elif enhancement_option == "Gaussian Blur":
            kernel_size = st.slider("Kernel Size", 1, 31, 5, 2)
            sigma = st.slider("Sigma", 0.1, 10.0, 1.0, 0.1)
            
            if kernel_size % 2 == 0:  # Ensure kernel size is odd
                kernel_size += 1
                
            blurred = cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), sigma)
            st.image(blurred, caption=f"Gaussian Blur (k={kernel_size}, σ={sigma})", use_column_width=True)
            st.markdown(get_download_link(blurred), unsafe_allow_html=True)
            
        elif enhancement_option == "Sharpening":
            sharpen_type = st.selectbox("Sharpening Method", ["Laplacian", "Unsharp Mask"])
            
            if sharpen_type == "Laplacian":
                kernel_size = st.slider("Kernel Size", 3, 7, 3, 2)
                kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
                if kernel_size == 5:
                    kernel = np.array([[-1,-1,-1,-1,-1], [-1,-1,-1,-1,-1], [-1,-1,25,-1,-1], [-1,-1,-1,-1,-1], [-1,-1,-1,-1,-1]])
                elif kernel_size == 7:
                    kernel = np.array([[-1,-1,-1,-1,-1,-1,-1], [-1,-1,-1,-1,-1,-1,-1], [-1,-1,-1,-1,-1,-1,-1], 
                                       [-1,-1,-1,49,-1,-1,-1], [-1,-1,-1,-1,-1,-1,-1], [-1,-1,-1,-1,-1,-1,-1], [-1,-1,-1,-1,-1,-1,-1]])
                sharp = cv2.filter2D(img_gray, -1, kernel)
                
            else:  # Unsharp Mask
                blur_amount = st.slider("Blur Amount", 0.1, 5.0, 1.0, 0.1)
                strength = st.slider("Strength", 0.1, 5.0, 1.5, 0.1)
                
                blurred = cv2.GaussianBlur(img_gray, (5, 5), blur_amount)
                sharp = cv2.addWeighted(img_gray, 1.0 + strength, blurred, -strength, 0)
                
            st.image(sharp, caption=f"Sharpening ({sharpen_type})", use_column_width=True)
            st.markdown(get_download_link(sharp), unsafe_allow_html=True)
            
        elif enhancement_option == "High Boost Filter":
            k = st.slider("Boost Factor", 1.0, 5.0, 1.5, 0.1)
            blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
            mask = img_gray.astype(float) - blurred.astype(float)
            high_boost = img_gray.astype(float) + k * mask
            high_boost = np.clip(high_boost, 0, 255).astype(np.uint8)
            
            st.image(high_boost, caption=f"High Boost Filter (k={k})", use_column_width=True)
            st.markdown(get_download_link(high_boost), unsafe_allow_html=True)
            
        elif enhancement_option == "Median Filter":
            kernel_size = st.slider("Kernel Size", 1, 31, 5, 2)
            if kernel_size % 2 == 0:  # Ensure kernel size is odd
                kernel_size += 1
                
            median = cv2.medianBlur(img_gray, kernel_size)
            st.image(median, caption=f"Median Filter (k={kernel_size})", use_column_width=True)
            st.markdown(get_download_link(median), unsafe_allow_html=True)
            
        elif enhancement_option == "Bilateral Filter":
            d = st.slider("Diameter", 5, 15, 9, 2)
            sigma_color = st.slider("Sigma Color", 10, 150, 75, 5)
            sigma_space = st.slider("Sigma Space", 10, 150, 75, 5)
            
            bilateral = cv2.bilateralFilter(img_gray, d, sigma_color, sigma_space)
            st.image(bilateral, caption="Bilateral Filter", use_column_width=True)
            st.markdown(get_download_link(bilateral), unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    with tab3:
        if len(img_rgb.shape) == 3:  # Only for color images
            st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
            st.subheader("Color Processing")
            
            color_option = st.selectbox(
                "Select Color Processing Technique",
                ["None", "Color Channels", "Color Spaces", "Color Quantization", "Color Balance"]
            )
            
            if color_option == "Color Channels":
                # Show RGB channels
                b, g, r = cv2.split(img_rgb)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(r, caption="Red Channel", use_column_width=True)
                with col2:
                    st.image(g, caption="Green Channel", use_column_width=True)
                with col3:
                    st.image(b, caption="Blue Channel", use_column_width=True)
                    
                # Allow channel mixing
                st.subheader("Channel Mixer")
                r_weight = st.slider("Red Weight", 0.0, 2.0, 1.0, 0.1)
                g_weight = st.slider("Green Weight", 0.0, 2.0, 1.0, 0.1)
                b_weight = st.slider("Blue Weight", 0.0, 2.0, 1.0, 0.1)
                
                mixed = cv2.merge([
                    np.clip(b * b_weight, 0, 255).astype(np.uint8),
                    np.clip(g * g_weight, 0, 255).astype(np.uint8),
                    np.clip(r * r_weight, 0, 255).astype(np.uint8)
                ])
                
                st.image(mixed, caption="Mixed Channels", use_column_width=True)
                
            elif color_option == "Color Spaces":
                space = st.selectbox("Select Color Space", ["RGB", "HSV", "LAB", "YCrCb"])
                
                if space == "HSV":
                    converted = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
                    h, s, v = cv2.split(converted)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(h, caption="Hue", use_column_width=True)
                    with col2:
                        st.image(s, caption="Saturation", use_column_width=True)
                    with col3:
                        st.image(v, caption="Value", use_column_width=True)
                        
                elif space == "LAB":
                    converted = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
                    l, a, b_ch = cv2.split(converted)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(l, caption="Lightness", use_column_width=True)
                    with col2:
                        a_normalized = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
                        st.image(a_normalized, caption="A Channel", use_column_width=True)
                    with col3:
                        b_normalized = cv2.normalize(b_ch, None, 0, 255, cv2.NORM_MINMAX)
                        st.image(b_normalized, caption="B Channel", use_column_width=True)
                        
                elif space == "YCrCb":
                    converted = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
                    y, cr, cb = cv2.split(converted)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(y, caption="Luminance", use_column_width=True)
                    with col2:
                        st.image(cr, caption="Cr (R-Y)", use_column_width=True)
                    with col3:
                        st.image(cb, caption="Cb (B-Y)", use_column_width=True)
                        
            elif color_option == "Color Quantization":
                k = st.slider("Number of Colors", 2, 64, 8)
                
                # Reshape the image
                pixels = img_rgb.reshape((-1, 3))
                pixels = np.float32(pixels)
                
                # Define criteria and apply kmeans
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                
                # Convert back to uint8
                centers = np.uint8(centers)
                segmented_data = centers[labels.flatten()]
                quantized = segmented_data.reshape(img_rgb.shape)
                
                # Create a color palette
                palette = centers.reshape((1, k, 3))
                palette = np.repeat(palette, 50, axis=0)
                
                st.image(quantized, caption=f"Color Quantization (k={k})", use_column_width=True)
                st.image(palette, caption="Color Palette", use_column_width=True)
                st.markdown(get_download_link(quantized), unsafe_allow_html=True)
                
            elif color_option == "Color Balance":
                st.subheader("Color Balance Adjustment")
                
                r_adj = st.slider("Red Adjustment", -50, 50, 0)
                g_adj = st.slider("Green Adjustment", -50, 50, 0)
                b_adj = st.slider("Blue Adjustment", -50, 50, 0)
                
                # Split the image into channels
                b, g, r = cv2.split(img_rgb)
                
                # Adjust each channel
                r = np.clip(r.astype(np.int16) + r_adj, 0, 255).astype(np.uint8)
                g = np.clip(g.astype(np.int16) + g_adj, 0, 255).astype(np.uint8)
                b = np.clip(b.astype(np.int16) + b_adj, 0, 255).astype(np.uint8)
                
                # Merge back
                balanced = cv2.merge([b, g, r])
                
                st.image(balanced, caption="Color Balanced", use_column_width=True)
                st.markdown(get_download_link(balanced), unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Color processing requires an RGB image. The current image is grayscale.")
        
# Compression Page
elif page == "Compression":
    st.markdown("<h2 class='sub-header'>🗜️ Image Compression</h2>", unsafe_allow_html=True)
    
    # Create tabs for different compression techniques
    tab1, tab2, tab3, tab4 = st.tabs(["Basic Compression", "Transform Compression", "Advanced Techniques", "Batch Processing"])
    
    with tab1:
        st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
        st.subheader("Basic Compression")
        
        basic_option = st.selectbox(
            "Select Basic Compression Method",
            ["None", "Run-Length Encoding", "Huffman Coding", "JPEG Quality Control"]
        )
        
        if basic_option == "Run-Length Encoding":
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

            # Allow image resize to see effect on compression
            resize_factor = st.slider("Resize Factor", 0.1, 1.0, 0.25, 0.05)
            resized = cv2.resize(img_gray, (0, 0), fx=resize_factor, fy=resize_factor)
            
            # Show resized image
            st.image(resized, caption=f"Resized Image ({resize_factor:.2f}x)", use_column_width=True)
            
            # Perform RLE
            with st.spinner("Encoding..."):
                rle = rle_encode(resized)
            
            # Show results
            original_size = get_size(resized)
            compressed_size = get_size(rle)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Size", f"{original_size/1024:.2f} KB")
            with col2:
                st.metric("Compressed Size", f"{compressed_size/1024:.2f} KB")
            with col3:
                st.metric("Compression Ratio", f"{compression_ratio:.2f}x")
            
            # Display first few RLE values
            st.subheader("RLE Encoded Values (Sample)")
            rle_df = pd.DataFrame(rle[:50], columns=["Value", "Count"])
            st.dataframe(rle_df)
            
            # Show distribution of run lengths
            fig, ax = plt.subplots(figsize=(8, 3))
            counts = [count for _, count in rle]
            ax.hist(counts, bins=30, color='#FF5722', alpha=0.7)
            ax.set_xlabel('Run Length')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        elif basic_option == "Huffman Coding":
            # Simple Huffman coding implementation
            from collections import Counter
            import heapq
            
            def build_huffman_tree(data):
                # Count frequency of each byte
                freq = Counter(data)
                
                # Create a priority queue (min heap)
                heap = [[weight, [byte, ""]] for byte, weight in freq.items()]
                heapq.heapify(heap)
                
                # Build Huffman tree
                while len(heap) > 1:
                    lo = heapq.heappop(heap)
                    hi = heapq.heappop(heap)
                    
                    for pair in lo[1:]:
                        pair[1] = '0' + pair[1]
                    for pair in hi[1:]:
                        pair[1] = '1' + pair[1]
                    
                    heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
                
                # Extract the codes
                huffman_codes = sorted(heapq.heappop(heap)[1:], key=lambda p: len(p[1]))
                return {byte: code for byte, code in huffman_codes}
            
            # Resize image for faster processing
            resize_factor = st.slider("Resize Factor", 0.1, 1.0, 0.25, 0.05)
            resized = cv2.resize(img_gray, (0, 0), fx=resize_factor, fy=resize_factor)
            
            # Calculate Huffman codes
            with st.spinner("Building Huffman Codes..."):
                huffman_codes = build_huffman_tree(resized.flatten())
            
            # Calculate encoded size (in bits)
            encoded_size_bits = sum(len(huffman_codes[pixel]) for pixel in resized.flatten())
            encoded_size_bytes = encoded_size_bits / 8
            
            # Show compression results
            original_size = get_size(resized)
            compression_ratio = original_size / encoded_size_bytes if encoded_size_bytes > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Size", f"{original_size/1024:.2f} KB")
            with col2:
                st.metric("Encoded Size", f"{encoded_size_bytes/1024:.2f} KB")
            with col3:
                st.metric("Compression Ratio", f"{compression_ratio:.2f}x")
            
            # Display code table
            # Display code table
            st.subheader("Huffman Codes")
            codes_df = pd.DataFrame({
                "Pixel Value": list(huffman_codes.keys()),
                "Code": list(huffman_codes.values()),
                "Code Length": [len(code) for code in huffman_codes.values()]
            })
            st.dataframe(codes_df.sort_values("Code Length"))
            
            # Show code length distribution
            fig, ax = plt.subplots(figsize=(8, 3))
            code_lengths = [len(code) for code in huffman_codes.values()]
            ax.hist(code_lengths, bins=range(min(code_lengths), max(code_lengths) + 2), color='#4CAF50', alpha=0.7)
            ax.set_xlabel('Code Length (bits)')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Show the image
            st.image(resized, caption=f"Resized Image ({resize_factor:.2f}x)", use_column_width=True)
            
        elif basic_option == "JPEG Quality Control":
            quality = st.slider("JPEG Quality", 1, 100, 75)
            
            # Convert to RGB if needed for PIL
            if len(img_rgb.shape) == 2:
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
                
            # Use PIL for JPEG compression
            with st.spinner("Compressing..."):
                # Convert to PIL Image
                pil_img = Image.fromarray(img_rgb)
                
                # Save with specified quality
                buffer = io.BytesIO()
                pil_img.save(buffer, format="JPEG", quality=quality)
                buffer.seek(0)
                
                # Get compressed size
                compressed_size = len(buffer.getvalue())
                
                # Load the compressed image back
                compressed_img = Image.open(buffer)
                compressed_img_array = np.array(compressed_img)
            
            # Show original and compressed images
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_rgb, caption="Original Image", use_column_width=True)
            with col2:
                st.image(compressed_img_array, caption=f"JPEG (Quality: {quality})", use_column_width=True)
            
            # Show compression metrics
            original_size = get_size(img_rgb)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Size", f"{original_size/1024:.2f} KB")
            with col2:
                st.metric("Compressed Size", f"{compressed_size/1024:.2f} KB")
            with col3:
                st.metric("Compression Ratio", f"{compression_ratio:.2f}x")
            
            # Calculate and show PSNR (Peak Signal-to-Noise Ratio)
            if st.checkbox("Show Image Quality Metrics"):
                with st.spinner("Computing image quality metrics..."):
                    # Convert both to same format for comparison
                    original = img_rgb.astype(float)
                    compressed = compressed_img_array.astype(float)
                    
                    # Calculate MSE
                    mse = np.mean((original - compressed) ** 2)
                    if mse == 0:  # Same images
                        psnr = float('inf')
                    else:
                        psnr = 10 * np.log10((255 ** 2) / mse)
                    
                    st.metric("PSNR (dB)", f"{psnr:.2f}")
                    st.info("Higher PSNR indicates better quality. Values above 30dB generally indicate good quality.")
    
    with tab2:
        st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
        st.subheader("Transform Compression")
        
        transform_option = st.selectbox(
            "Select Transform Method",
            ["None", "Discrete Cosine Transform (DCT)", "Discrete Wavelet Transform (DWT)"]
        )
        
        if transform_option == "Discrete Cosine Transform (DCT)":
            # DCT implementation
            st.info("DCT is the basis for JPEG compression. We'll visualize how it works.")
            
            # Get block size for DCT
            block_size = st.select_slider("Block Size", options=[8, 16, 32, 64], value=8)
            
            # Get quality factor (similar to JPEG quality)
            quality_factor = st.slider("Quality Factor", 1, 100, 50)
            
            # Function to apply DCT to an image block
            def apply_dct(block):
                return cv2.dct(np.float32(block))
            
            # Function to apply inverse DCT
            def apply_idct(dct_block):
                return cv2.idct(dct_block)
            
            # Function to quantize DCT coefficients
            def quantize_dct(dct_block, quality):
                # Standard JPEG quantization matrix for Y (luminance)
                q_matrix = np.array([
                    [16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]
                ])
                
                # Scale the matrix based on quality
                scale = 5000 / quality if quality < 50 else 200 - 2 * quality
                q = np.ones(q_matrix.shape) * scale
                q = (q_matrix * q + 50) / 100
                
                # If block size is different, resize the quantization matrix
                if block_size != 8:
                    q = cv2.resize(q, (block_size, block_size))
                
                # Apply quantization
                return np.round(dct_block / q) * q
            
            # Process the image
            with st.spinner("Applying DCT transformation..."):
                # Use grayscale for simplicity
                img_show = cv2.resize(img_gray, (256, 256))
                
                # Prepare result containers
                dct_img = np.zeros_like(img_show, dtype=np.float32)
                reconstructed_img = np.zeros_like(img_show)
                
                # Process each block
                for y in range(0, img_show.shape[0], block_size):
                    for x in range(0, img_show.shape[1], block_size):
                        # Extract block
                        block = img_show[y:y+block_size, x:x+block_size]
                        
                        # Handle incomplete blocks at edges
                        if block.shape[0] != block_size or block.shape[1] != block_size:
                            continue
                        
                        # Apply DCT
                        dct_block = apply_dct(block)
                        
                        # Quantize
                        quantized_dct = quantize_dct(dct_block, quality_factor)
                        
                        # Store DCT result
                        dct_img[y:y+block_size, x:x+block_size] = np.abs(quantized_dct)
                        
                        # Apply inverse DCT
                        reconstructed_block = apply_idct(quantized_dct)
                        reconstructed_img[y:y+block_size, x:x+block_size] = np.clip(reconstructed_block, 0, 255)
            
            # Show original and reconstructed images
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_show, caption="Original Image", use_column_width=True)
            with col2:
                st.image(reconstructed_img.astype(np.uint8), caption=f"Reconstructed (Quality: {quality_factor})", use_column_width=True)
            
            # Show DCT coefficients
            st.subheader("DCT Coefficients (log scale)")
            # Apply log scale for better visualization and normalize to [0, 1]
            log_dct = np.log(np.abs(dct_img) + 1)  # Add 1 to avoid log(0)
            log_dct_normalized = (log_dct - np.min(log_dct)) / (np.max(log_dct) - np.min(log_dct))
            st.image(log_dct_normalized, caption="DCT Coefficients", use_column_width=True)
            
            # Show a sample block
            st.subheader("Sample Block Analysis")
            block_y = st.slider("Block Y Position", 0, img_show.shape[0] - block_size, block_size)
            block_x = st.slider("Block X Position", 0, img_show.shape[1] - block_size, block_size)
            
            # Extract the sample block
            sample_block = img_show[block_y:block_y+block_size, block_x:block_x+block_size]
            sample_dct = apply_dct(sample_block)
            sample_quantized = quantize_dct(sample_dct, quality_factor)
            sample_reconstructed = apply_idct(sample_quantized)
            
            # Calculate how many coefficients are zeroed out
            zero_count = np.sum(sample_quantized == 0)
            zero_percentage = zero_count / (block_size * block_size) * 100
            
            st.metric("Zero Coefficients", f"{zero_count} out of {block_size*block_size} ({zero_percentage:.1f}%)")
            
            # Show block comparison
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(sample_block, caption="Original Block", use_column_width=True)
            with col2:
                st.image(np.log(np.abs(sample_quantized) + 1) / np.log(np.max(np.abs(sample_quantized)) + 1) * 255, 
                         caption="DCT Coefficients", use_column_width=True)
            with col3:
                st.image(np.clip(sample_reconstructed, 0, 255).astype(np.uint8), 
                         caption="Reconstructed Block", use_column_width=True)
        
        elif transform_option == "Discrete Wavelet Transform (DWT)":
            st.info("Wavelet transform is used in JPEG2000 and provides better quality at high compression ratios.")
            
            # Check if PyWavelets is available
            try:
                import pywt
                has_pywt = True
            except ImportError:
                has_pywt = False
                st.error("PyWavelets library is not installed. Please install it to use DWT.")
                st.code("pip install PyWavelets")
            
            if has_pywt:
                # Select wavelet type
                wavelet = st.selectbox("Wavelet Type", ["haar", "db2", "sym2", "coif1", "bior1.3"])
                
                # Select decomposition level
                level = st.slider("Decomposition Level", 1, 3, 1)
                
                # Select threshold percentage for compression
                threshold_pct = st.slider("Threshold (% of max coefficient)", 0, 99, 50)
                
                with st.spinner("Applying wavelet transform..."):
                    # Resize for faster processing
                    img_resize = cv2.resize(img_gray, (256, 256))
                    
                    # Apply wavelet transform
                    coeffs = pywt.wavedec2(img_resize, wavelet, level=level)
                    
                    # Create a copy for thresholding
                    coeffs_thresholded = [coeffs[0].copy()]
                    
                    # Find maximum coefficient value (excluding approximation)
                    max_coeff = max([np.max(np.abs(detail)) for details in coeffs[1:] for detail in details])
                    threshold = max_coeff * threshold_pct / 100
                    
                    # Apply thresholding to detail coefficients
                    for i in range(1, len(coeffs)):
                        coeffs_thresholded.append(tuple(pywt.threshold(detail, threshold) for detail in coeffs[i]))
                    
                    # Count zeroed coefficients
                    total_coeffs = sum(c.size for details in coeffs[1:] for c in details)
                    zero_coeffs = sum(np.sum(c == 0) for details in coeffs_thresholded[1:] for c in details)
                    zero_pct = zero_coeffs / total_coeffs * 100
                    
                    # Reconstruct image
                    reconstructed = pywt.waverec2(coeffs_thresholded, wavelet)
                    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
                
                # Show compression metrics
                st.metric("Coefficients Set to Zero", f"{zero_pct:.1f}%")
                
                # Show original and reconstructed
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img_resize, caption="Original Image", use_column_width=True)
                with col2:
                    st.image(reconstructed, caption="Reconstructed Image", use_column_width=True)
                
                # Show wavelet decomposition
                st.subheader("Wavelet Decomposition")
                
                # Create visual representation of wavelet decomposition
                arr = coeffs_thresholded[0].copy()
                for i in range(level):
                    h, w = arr.shape
                    
                    # Normalize detail coefficients for visualization
                    h_arr, v_arr, d_arr = [np.abs(detail) for detail in coeffs_thresholded[i+1]]
                    h_max, v_max, d_max = np.max(h_arr), np.max(v_arr), np.max(d_arr)
                    
                    h_arr = h_arr / (h_max if h_max > 0 else 1) * 255
                    v_arr = v_arr / (v_max if v_max > 0 else 1) * 255
                    d_arr = d_arr / (d_max if d_max > 0 else 1) * 255
                    
                    # Stack details next to approximation
                    arr = np.vstack((np.hstack((arr, h_arr)),
                                    np.hstack((v_arr, d_arr))))
                
                # Normalize for display
                arr = arr / np.max(arr) * 255 if np.max(arr) > 0 else arr
                st.image(arr.astype(np.uint8), caption="Wavelet Decomposition", use_column_width=True)
    
    with tab3:
        st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
        st.subheader("Advanced Compression Techniques")
        
        advanced_option = st.selectbox(
            "Select Advanced Technique",
            ["None", "Vector Quantization", "Fractal Compression"]
        )
        
        if advanced_option == "Vector Quantization":
            st.info("Vector Quantization compresses images by mapping blocks to a codebook of representative vectors.")
            
            # Parameters
            block_size = st.select_slider("Block Size", options=[2, 4, 8], value=4)
            codebook_size = st.select_slider("Codebook Size", options=[16, 32, 64, 128, 256], value=64)
            
            # Resize image for faster processing
            resize_factor = st.slider("Resize Factor", 0.1, 1.0, 0.25, 0.05)
            resized = cv2.resize(img_gray, (0, 0), fx=resize_factor, fy=resize_factor)
            
            with st.spinner("Performing Vector Quantization..."):
                # Extract image blocks
                h, w = resized.shape
                blocks = []
                for y in range(0, h - block_size + 1, block_size):
                    for x in range(0, w - block_size + 1, block_size):
                        block = resized[y:y+block_size, x:x+block_size].flatten()
                        blocks.append(block)
                
                # Convert to numpy array
                blocks = np.array(blocks)
                
                # Perform K-means clustering to create codebook
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=codebook_size, random_state=0).fit(blocks)
                
                # Get cluster centers (codebook) and labels (indices)
                codebook = kmeans.cluster_centers_
                indices = kmeans.labels_
                
                # Reconstruct image
                reconstructed = np.zeros((h, w))
                idx = 0
                for y in range(0, h - block_size + 1, block_size):
                    for x in range(0, w - block_size + 1, block_size):
                        reconstructed[y:y+block_size, x:x+block_size] = codebook[indices[idx]].reshape(block_size, block_size)
                        idx += 1
            
            # Calculate compression metrics
            original_size = resized.size * resized.itemsize
            compressed_size = (codebook.size * codebook.itemsize) + (indices.size * indices.itemsize)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
            
            # Show compression metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Size", f"{original_size/1024:.2f} KB")
            with col2:
                st.metric("Compressed Size", f"{compressed_size/1024:.2f} KB")
            with col3:
                st.metric("Compression Ratio", f"{compression_ratio:.2f}x")
            
            # Show original and reconstructed images
            col1, col2 = st.columns(2)
            with col1:
                st.image(resized, caption="Original Image", use_column_width=True)
            with col2:
                st.image(reconstructed.astype(np.uint8), caption="Reconstructed Image", use_column_width=True)
            
            # Show codebook
            st.subheader("Codebook (First 16 entries)")
            
            # Calculate grid dimensions
            grid_size = min(16, codebook_size)
            grid_cols = int(np.sqrt(grid_size))
            grid_rows = (grid_size + grid_cols - 1) // grid_cols
            
            # Create grid of codebook entries
            codebook_grid = np.zeros((grid_rows * block_size, grid_cols * block_size))
            for i in range(min(grid_size, codebook_size)):
                r, c = i // grid_cols, i % grid_cols
                codebook_grid[r*block_size:(r+1)*block_size, c*block_size:(c+1)*block_size] = codebook[i].reshape(block_size, block_size)
            
            st.image(codebook_grid.astype(np.uint8), caption="Codebook Entries", use_column_width=True)
            
            # Show histogram of codebook usage
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.hist(indices, bins=range(codebook_size + 1), color='#9C27B0', alpha=0.7)
            ax.set_xlabel('Codebook Index')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        elif advanced_option == "Fractal Compression":
            st.info("Fractal compression exploits self-similarity in images. This is a simple demonstration of the concept.")
            st.warning("Note: Full fractal compression is computationally intensive. This is a simplified version.")
            
            # Parameters
            domain_size = st.select_slider("Domain Block Size", options=[8, 16, 32], value=16)
            range_size = domain_size // 2
            
            # Resize image for faster processing
            resize_factor = st.slider("Resize Factor", 0.1, 0.5, 0.25, 0.05)
            resized = cv2.resize(img_gray, (0, 0), fx=resize_factor, fy=resize_factor)
            
            # Ensure dimensions are multiples of domain_size
            h, w = resized.shape
            h_new = (h // domain_size) * domain_size
            w_new = (w // domain_size) * domain_size
            resized = resized[:h_new, :w_new]
            
            with st.spinner("Finding self-similarities..."):
                # Create domain blocks (larger blocks)
                domain_blocks = []
                domain_positions = []
                for y in range(0, h_new, domain_size):
                    for x in range(0, w_new, domain_size):
                        block = resized[y:y+domain_size, x:x+domain_size]
                        # Downsample to range size
                        downsampled = cv2.resize(block, (range_size, range_size))
                        domain_blocks.append(downsampled)
                        domain_positions.append((y, x))
                
                # Create range blocks (smaller blocks)
                range_blocks = []
                range_positions = []
                for y in range(0, h_new, range_size):
                    for x in range(0, w_new, range_size):
                        block = resized[y:y+range_size, x:x+range_size]
                        range_blocks.append(block)
                        range_positions.append((y, x))
                
                # Simple matching (find closest domain block for each range block)
                mappings = []
                for i, range_block in enumerate(range_blocks):
                    best_match = 0
                    min_error = float('inf')
                    
                    # Find best matching domain block
                    for j, domain_block in enumerate(domain_blocks):
                        error = np.sum((range_block - domain_block) ** 2)
                        if error < min_error:
                            min_error = error
                            best_match = j
                    
                    mappings.append((best_match, min_error))
                
                # Calculate compression
                original_size = resized.size * resized.itemsize
                # Each mapping stores: domain index, brightness offset, contrast scale
                mapping_size = len(mappings) * (4 + 1 + 1)  # 4 bytes for index, 1 byte each for offset and scale
                compression_ratio = original_size / mapping_size if mapping_size > 0 else 0
            
            # Show metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Size", f"{original_size/1024:.2f} KB")
            with col2:
                st.metric("Mapping Size", f"{mapping_size/1024:.2f} KB")
            with col3:
                st.metric("Compression Ratio", f"{compression_ratio:.2f}x")
            
            # Show heat map of mapping errors
            errors = np.array([m[1] for m in mappings])
            error_img = np.zeros((h_new, w_new))
            
            for i, pos in enumerate(range_positions):
                y, x = pos
                error_img[y:y+range_size, x:x+range_size] = errors[i]
            
            # Normalize error for visualization
            error_img = error_img / np.max(error_img) if np.max(error_img) > 0 else error_img
            
            # Show error heat map
            st.subheader("Mapping Error Heat Map")
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(error_img, cmap='hot')
            plt.colorbar(im, ax=ax, label="Normalized Error")
            ax.set_title("Self-similarity Mapping Errors")
            st.pyplot(fig)
            
            # Show original image
            st.image(resized, caption="Original Image", use_column_width=True)
            
            # Show some domain-range mappings
            st.subheader("Sample Domain-Range Mappings")
            
            # Select a few random mappings to display
            num_samples = min(5, len(mappings))
            sample_indices = random.sample(range(len(mappings)), num_samples)
            
            for idx in sample_indices:
                col1, col2 = st.columns(2)
                domain_idx, error = mappings[idx]
                range_y, range_x = range_positions[idx]
                domain_y, domain_x = domain_positions[domain_idx]
                
                with col1:
                    st.image(resized[domain_y:domain_y+domain_size, domain_x:domain_x+domain_size], 
                             caption=f"Domain Block at ({domain_y}, {domain_x})", use_column_width=True)
                
                with col2:
                    st.image(resized[range_y:range_y+range_size, range_x:range_x+range_size],
                             caption=f"Range Block at ({range_y}, {range_x})", use_column_width=True)
                
                st.text(f"Mapping Error: {error:.2f}")
                st.divider()

    with tab4:
        st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
        st.subheader("Batch Processing")
        
        # Allow multiple file uploads
        uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
        
        if uploaded_files:
            # Select compression method
            compression_method = st.selectbox(
                "Select Compression Method",
                ["JPEG Quality Control", "DCT Compression", "Wavelet Transform"]
            )
            
            # Add resize option for faster processing
            resize_factor = st.slider("Resize Factor (for faster processing)", 0.1, 1.0, 0.5, 0.1)
            
            if compression_method == "JPEG Quality Control":
                quality = st.slider("JPEG Quality", 1, 100, 75)
            elif compression_method == "DCT Compression":
                block_size = st.select_slider("Block Size", options=[8, 16, 32, 64], value=8)
                quality_factor = st.slider("Quality Factor", 1, 100, 50)
            elif compression_method == "Wavelet Transform":
                wavelet = st.selectbox("Wavelet Type", ["haar", "db2", "sym2", "coif1", "bior1.3"])
                level = st.slider("Decomposition Level", 1, 3, 1)
                threshold_pct = st.slider("Threshold (% of max coefficient)", 0, 99, 50)
            
            def process_single_image(uploaded_file, method, resize_factor, quality=None, block_size=None, 
                                   quality_factor=None, wavelet=None, level=None, threshold_pct=None):
                try:
                    # Load and resize image
                    image = Image.open(uploaded_file)
                    if resize_factor < 1.0:
                        new_size = (int(image.width * resize_factor), int(image.height * resize_factor))
                        image = image.resize(new_size, Image.Resampling.LANCZOS)
                    
                    # Keep original color format
                    img_array = np.array(image)
                    original_size = get_size(img_array)
                    
                    # Apply selected compression
                    if method == "JPEG Quality Control":
                        # Save directly with original color format
                        buffer = io.BytesIO()
                        image.save(buffer, format="JPEG", quality=quality, optimize=True)
                        compressed_size = len(buffer.getvalue())
                        compressed_data = buffer.getvalue()
                        
                    elif method == "DCT Compression":
                        # Convert to YCrCb for DCT processing
                        if len(img_array.shape) == 3:
                            img_ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
                        else:
                            img_ycrcb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                            img_ycrcb = cv2.cvtColor(img_ycrcb, cv2.COLOR_BGR2YCrCb)
                        
                        # Resize to power of 2 for DCT
                        size = 2**int(np.log2(min(img_ycrcb.shape[:2])))
                        img_resized = cv2.resize(img_ycrcb, (size, size))
                        
                        # Process each channel
                        compressed_size = 0
                        for channel in range(3):
                            for y in range(0, img_resized.shape[0], block_size):
                                for x in range(0, img_resized.shape[1], block_size):
                                    block = img_resized[y:y+block_size, x:x+block_size, channel]
                                    if block.shape[0] == block_size and block.shape[1] == block_size:
                                        dct_block = cv2.dct(np.float32(block))
                                        compressed_size += get_size(dct_block)
                        
                        # Save compressed version
                        buffer = io.BytesIO()
                        Image.fromarray(img_resized).save(buffer, format="JPEG", quality=quality_factor, optimize=True)
                        compressed_data = buffer.getvalue()
                        compressed_size = len(compressed_data)
                    
                    elif method == "Wavelet Transform":
                        # Process each channel if color image
                        if len(img_array.shape) == 3:
                            compressed_size = 0
                            for channel in range(3):
                                coeffs = pywt.wavedec2(img_array[:,:,channel], wavelet, level=level)
                                compressed_size += get_size(coeffs[0])
                        else:
                            coeffs = pywt.wavedec2(img_array, wavelet, level=level)
                            compressed_size = get_size(coeffs[0])
                        
                        # Save compressed version
                        buffer = io.BytesIO()
                        image.save(buffer, format="JPEG", quality=90, optimize=True)
                        compressed_data = buffer.getvalue()
                        compressed_size = len(compressed_data)
                    
                    return {
                        "filename": uploaded_file.name,
                        "original_size": original_size,
                        "compressed_size": compressed_size,
                        "compression_ratio": original_size / compressed_size,
                        "compressed_data": compressed_data
                    }
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    return None
            
            if st.button("Process Batch"):
                # Initialize statistics
                total_original_size = 0
                total_compressed_size = 0
                compression_ratios = []
                processed_images = []
                compressed_files = []
                
                # Create progress container
                progress_container = st.empty()
                progress_text = progress_container.text("Processing images...")
                
                # Process images in parallel
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Create partial function with fixed parameters
                    process_func = partial(
                        process_single_image,
                        method=compression_method,
                        resize_factor=resize_factor,
                        quality=quality if compression_method == "JPEG Quality Control" else None,
                        block_size=block_size if compression_method == "DCT Compression" else None,
                        quality_factor=quality_factor if compression_method == "DCT Compression" else None,
                        wavelet=wavelet if compression_method == "Wavelet Transform" else None,
                        level=level if compression_method == "Wavelet Transform" else None,
                        threshold_pct=threshold_pct if compression_method == "Wavelet Transform" else None
                    )
                    
                    # Submit all tasks
                    futures = [executor.submit(process_func, file) for file in uploaded_files]
                    
                    # Process results as they complete
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        result = future.result()
                        if result:
                            processed_images.append({
                                "filename": result["filename"],
                                "original_size": result["original_size"],
                                "compressed_size": result["compressed_size"],
                                "compression_ratio": result["compression_ratio"]
                            })
                            compressed_files.append((result["filename"], result["compressed_data"]))
                            
                            total_original_size += result["original_size"]
                            total_compressed_size += result["compressed_size"]
                            compression_ratios.append(result["compression_ratio"])
                            
                            # Update progress
                            progress = (i + 1) / len(uploaded_files)
                            progress_text.text(f"Processed {i + 1}/{len(uploaded_files)} images")
                
                # Create ZIP file in memory
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for filename, data in compressed_files:
                        zipf.writestr(f"compressed_{filename}", data)
                
                # Reset buffer position
                zip_buffer.seek(0)
                
                # Display batch processing results
                st.subheader("Batch Processing Results")
                
                # Overall statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Original Size", f"{total_original_size/1024/1024:.2f} MB")
                with col2:
                    st.metric("Total Compressed Size", f"{total_compressed_size/1024/1024:.2f} MB")
                with col3:
                    avg_ratio = total_original_size / total_compressed_size
                    st.metric("Average Compression Ratio", f"{avg_ratio:.2f}x")
                
                # Detailed results table
                st.subheader("Individual Image Results")
                results_df = pd.DataFrame(processed_images)
                st.dataframe(results_df)
                
                # Compression ratio distribution
                st.subheader("Compression Ratio Distribution")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(compression_ratios, bins=20, color='#4CAF50', alpha=0.7)
                ax.set_xlabel('Compression Ratio')
                ax.set_ylabel('Number of Images')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    # Download results as CSV
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="compression_results.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Download compressed images as ZIP
                    st.download_button(
                        label="Download Compressed Images (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name="compressed_images.zip",
                        mime="application/zip"
                    )
                
                # Clear the buffer
                zip_buffer.close()
        
        st.markdown("</div>", unsafe_allow_html=True)

# Add new section for Background Removal
elif page == "Background Removal":
    st.markdown("<h2 class='sub-header'>🎨 Background Removal</h2>", unsafe_allow_html=True)
    
    if uploaded_file:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Create tabs for different background removal methods
        tab1, tab2, tab3 = st.tabs(["Color-Based", "Edge-Based", "AI-Based"])
        
        with tab1:
            st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
            st.subheader("Color-Based Background Removal")
            
            # Color thresholding method
            method = st.selectbox(
                "Select Color Space",
                ["RGB", "HSV", "LAB"]
            )
            
            if method == "RGB":
                # RGB color range selection
                col1, col2 = st.columns(2)
                with col1:
                    lower_r = st.slider("Lower Red", 0, 255, 0, key="lower_r")
                    lower_g = st.slider("Lower Green", 0, 255, 0, key="lower_g")
                    lower_b = st.slider("Lower Blue", 0, 255, 0, key="lower_b")
                with col2:
                    upper_r = st.slider("Upper Red", 0, 255, 255, key="upper_r")
                    upper_g = st.slider("Upper Green", 0, 255, 255, key="upper_g")
                    upper_b = st.slider("Upper Blue", 0, 255, 255, key="upper_b")
                
                lower = np.array([lower_b, lower_g, lower_r])
                upper = np.array([upper_b, upper_g, upper_r])
                mask = cv2.inRange(img_array, lower, upper)
                
            elif method == "HSV":
                # Convert to HSV
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                col1, col2 = st.columns(2)
                with col1:
                    lower_h = st.slider("Lower Hue", 0, 179, 0, key="lower_h")
                    lower_s = st.slider("Lower Saturation", 0, 255, 0, key="lower_s")
                    lower_v = st.slider("Lower Value", 0, 255, 0, key="lower_v")
                with col2:
                    upper_h = st.slider("Upper Hue", 0, 179, 179, key="upper_h")
                    upper_s = st.slider("Upper Saturation", 0, 255, 255, key="upper_s")
                    upper_v = st.slider("Upper Value", 0, 255, 255, key="upper_v")
                
                lower = np.array([lower_h, lower_s, lower_v])
                upper = np.array([upper_h, upper_s, upper_v])
                mask = cv2.inRange(hsv, lower, upper)
                
            elif method == "LAB":
                # Convert to LAB
                lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                col1, col2 = st.columns(2)
                with col1:
                    lower_l = st.slider("Lower L", 0, 255, 0, key="lower_l")
                    lower_a = st.slider("Lower A", 0, 255, 0, key="lower_a")
                    lower_b = st.slider("Lower B", 0, 255, 0, key="lower_b")
                with col2:
                    upper_l = st.slider("Upper L", 0, 255, 255, key="upper_l")
                    upper_a = st.slider("Upper A", 0, 255, 255, key="upper_a")
                    upper_b = st.slider("Upper B", 0, 255, 255, key="upper_b")
                
                lower = np.array([lower_l, lower_a, lower_b])
                upper = np.array([upper_l, upper_a, upper_b])
                mask = cv2.inRange(lab, lower, upper)
            
            # Apply morphological operations to clean up the mask
            kernel_size = st.slider("Cleanup Kernel Size", 1, 21, 5, 2, key="color_kernel_size")
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            # Apply morphological operations
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Create transparent background
            result = img_array.copy()
            result = cv2.cvtColor(result, cv2.COLOR_RGB2RGBA)
            result[:, :, 3] = mask
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_array, caption="Original Image", use_column_width=True)
            with col2:
                st.image(result, caption="Background Removed", use_column_width=True)
            
            # Download options
            st.download_button(
                label="Download Result (PNG)",
                data=cv2.imencode('.png', result)[1].tobytes(),
                file_name="background_removed.png",
                mime="image/png",
                key="color_download"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab2:
            st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
            st.subheader("Edge-Based Background Removal")
            
            # Edge detection parameters
            threshold1 = st.slider("Lower Threshold", 0, 255, 100, key="edge_threshold1")
            threshold2 = st.slider("Upper Threshold", 0, 255, 200, key="edge_threshold2")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, threshold1, threshold2)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create mask from largest contour
            mask = np.zeros_like(gray)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            
            # Apply morphological operations
            kernel_size = st.slider("Cleanup Kernel Size", 1, 21, 5, 2, key="edge_kernel_size")
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Create transparent background
            result = img_array.copy()
            result = cv2.cvtColor(result, cv2.COLOR_RGB2RGBA)
            result[:, :, 3] = mask
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_array, caption="Original Image", use_column_width=True)
            with col2:
                st.image(result, caption="Background Removed", use_column_width=True)
            
            # Download options
            st.download_button(
                label="Download Result (PNG)",
                data=cv2.imencode('.png', result)[1].tobytes(),
                file_name="background_removed.png",
                mime="image/png",
                key="edge_download"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab3:
            st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
            st.subheader("AI-Based Background Removal")
            
            st.info("This feature requires an AI model for background removal. Please upload an image to process.")
            
            # Placeholder for AI-based background removal
            # This would typically use a pre-trained model like U2Net or DeepLab
            # For now, we'll show a message about the feature
            
            st.markdown("""
            ### Coming Soon!
            This feature will use advanced AI models to automatically remove backgrounds with high accuracy.
            
            Features to be included:
            - Automatic background detection
            - High-quality edge preservation
            - Support for complex backgrounds
            - Real-time processing
            """)
            
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Please upload an image to use background removal features.")

# Add new section for Color Enhancement
elif page == "Color Enhancement":
    st.markdown("<h2 class='sub-header'>🎨 Color Enhancement</h2>", unsafe_allow_html=True)
    
    if uploaded_file:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Create tabs for different color enhancement methods
        tab1, tab2, tab3, tab4 = st.tabs(["Selective Color Enhancement", "Color Balance", "Color Space Enhancement", "Deficiency Enhancement"])
        
        with tab1:
            st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
            st.subheader("Selective Color Enhancement")
            
            # Color selection
            color_space = st.selectbox(
                "Select Color Space",
                ["RGB", "HSV", "LAB"],
                key="enhance_color_space"
            )
            
            if color_space == "RGB":
                # RGB color enhancement
                col1, col2, col3 = st.columns(3)
                with col1:
                    red_enhance = st.slider("Red Enhancement", 0.0, 2.0, 1.0, 0.1, key="red_enhance")
                with col2:
                    green_enhance = st.slider("Green Enhancement", 0.0, 2.0, 1.0, 0.1, key="green_enhance")
                with col3:
                    blue_enhance = st.slider("Blue Enhancement", 0.0, 2.0, 1.0, 0.1, key="blue_enhance")
                
                # Apply enhancement
                enhanced = img_array.copy()
                enhanced[:,:,0] = np.clip(enhanced[:,:,0] * blue_enhance, 0, 255)
                enhanced[:,:,1] = np.clip(enhanced[:,:,1] * green_enhance, 0, 255)
                enhanced[:,:,2] = np.clip(enhanced[:,:,2] * red_enhance, 0, 255)
                
            elif color_space == "HSV":
                # Convert to HSV
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    hue_shift = st.slider("Hue Shift", -180, 180, 0, key="hue_shift")
                with col2:
                    saturation_enhance = st.slider("Saturation Enhancement", 0.0, 2.0, 1.0, 0.1, key="saturation_enhance")
                with col3:
                    value_enhance = st.slider("Value Enhancement", 0.0, 2.0, 1.0, 0.1, key="value_enhance")
                
                # Apply enhancement
                hsv_enhanced = hsv.copy()
                hsv_enhanced[:,:,0] = (hsv_enhanced[:,:,0] + hue_shift) % 180
                hsv_enhanced[:,:,1] = np.clip(hsv_enhanced[:,:,1] * saturation_enhance, 0, 255)
                hsv_enhanced[:,:,2] = np.clip(hsv_enhanced[:,:,2] * value_enhance, 0, 255)
                
                # Convert back to RGB
                enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)
                
            elif color_space == "LAB":
                # Convert to LAB
                lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    lightness_enhance = st.slider("Lightness Enhancement", 0.0, 2.0, 1.0, 0.1, key="lightness_enhance")
                with col2:
                    a_enhance = st.slider("A Channel Enhancement", 0.0, 2.0, 1.0, 0.1, key="a_enhance")
                with col3:
                    b_enhance = st.slider("B Channel Enhancement", 0.0, 2.0, 1.0, 0.1, key="b_enhance")
                
                # Apply enhancement
                lab_enhanced = lab.copy()
                lab_enhanced[:,:,0] = np.clip(lab_enhanced[:,:,0] * lightness_enhance, 0, 255)
                lab_enhanced[:,:,1:] = np.clip(lab_enhanced[:,:,1:] * a_enhance, 0, 255)
                lab_enhanced[:,:,2] = np.clip(lab_enhanced[:,:,2] * b_enhance, 0, 255)
                
                # Convert back to RGB
                enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_array, caption="Original Image", use_column_width=True)
            with col2:
                st.image(enhanced, caption="Enhanced Image", use_column_width=True)
            
            # Download options
            st.download_button(
                label="Download Enhanced Image (PNG)",
                data=cv2.imencode('.png', enhanced)[1].tobytes(),
                file_name="color_enhanced.png",
                mime="image/png",
                key="selective_download"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab2:
            st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
            st.subheader("Color Balance")
            
            # Color balance adjustments
            col1, col2, col3 = st.columns(3)
            with col1:
                red_balance = st.slider("Red Balance", -50, 50, 0, key="red_balance")
            with col2:
                green_balance = st.slider("Green Balance", -50, 50, 0, key="green_balance")
            with col3:
                blue_balance = st.slider("Blue Balance", -50, 50, 0, key="blue_balance")
            
            # Apply color balance
            balanced = img_array.copy()
            balanced[:,:,0] = np.clip(balanced[:,:,0] + blue_balance, 0, 255)
            balanced[:,:,1] = np.clip(balanced[:,:,1] + green_balance, 0, 255)
            balanced[:,:,2] = np.clip(balanced[:,:,2] + red_balance, 0, 255)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_array, caption="Original Image", use_column_width=True)
            with col2:
                st.image(balanced, caption="Color Balanced Image", use_column_width=True)
            
            # Download options
            st.download_button(
                label="Download Balanced Image (PNG)",
                data=cv2.imencode('.png', balanced)[1].tobytes(),
                file_name="color_balanced.png",
                mime="image/png",
                key="balance_download"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab3:
            st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
            st.subheader("Color Space Enhancement")
            
            # Color space selection
            space = st.selectbox(
                "Select Color Space for Enhancement",
                ["RGB", "HSV", "LAB", "YCrCb"],
                key="space_enhance"
            )
            
            if space == "RGB":
                # RGB enhancement
                contrast = st.slider("Contrast", 0.0, 2.0, 1.0, 0.1, key="rgb_contrast")
                brightness = st.slider("Brightness", -50, 50, 0, key="rgb_brightness")
                
                enhanced = cv2.convertScaleAbs(img_array, alpha=contrast, beta=brightness)
                
            elif space == "HSV":
                # HSV enhancement
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                
                saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.1, key="hsv_saturation")
                value = st.slider("Value", 0.0, 2.0, 1.0, 0.1, key="hsv_value")
                
                hsv_enhanced = hsv.copy()
                hsv_enhanced[:,:,1] = np.clip(hsv_enhanced[:,:,1] * saturation, 0, 255)
                hsv_enhanced[:,:,2] = np.clip(hsv_enhanced[:,:,2] * value, 0, 255)
                
                enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)
                
            elif space == "LAB":
                # LAB enhancement
                lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                
                lightness = st.slider("Lightness", 0.0, 2.0, 1.0, 0.1, key="lab_lightness")
                colorfulness = st.slider("Colorfulness", 0.0, 2.0, 1.0, 0.1, key="lab_colorfulness")
                
                lab_enhanced = lab.copy()
                lab_enhanced[:,:,0] = np.clip(lab_enhanced[:,:,0] * lightness, 0, 255)
                lab_enhanced[:,:,1:] = np.clip(lab_enhanced[:,:,1:] * colorfulness, 0, 255)
                
                enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
                
            elif space == "YCrCb":
                # YCrCb enhancement
                ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
                
                luminance = st.slider("Luminance", 0.0, 2.0, 1.0, 0.1, key="ycrcb_luminance")
                chroma = st.slider("Chroma", 0.0, 2.0, 1.0, 0.1, key="ycrcb_chroma")
                
                ycrcb_enhanced = ycrcb.copy()
                ycrcb_enhanced[:,:,0] = np.clip(ycrcb_enhanced[:,:,0] * luminance, 0, 255)
                ycrcb_enhanced[:,:,1:] = np.clip(ycrcb_enhanced[:,:,1:] * chroma, 0, 255)
                
                enhanced = cv2.cvtColor(ycrcb_enhanced, cv2.COLOR_YCrCb2RGB)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_array, caption="Original Image", use_column_width=True)
            with col2:
                st.image(enhanced, caption="Enhanced Image", use_column_width=True)
            
            # Download options
            st.download_button(
                label="Download Enhanced Image (PNG)",
                data=cv2.imencode('.png', enhanced)[1].tobytes(),
                file_name="space_enhanced.png",
                mime="image/png",
                key="space_download"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab4:
            st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
            st.subheader("Deficiency Enhancement")
            
            st.info("This feature enhances specific colors related to nutrient deficiencies in leaves.")
            
            # Define HSV color ranges for symptoms
            color_ranges = {
                "N": [(25, 40, 40), (35, 255, 255)],  # Yellowish-Green (N Deficiency)
                "P": [(125, 40, 40), (160, 255, 255)],  # Purplish (P Deficiency)
                "K": [(5, 40, 40), (20, 255, 255)]  # Reddish-Brown (K Deficiency)
            }
            
            # Enhancement strength (fixed at 4 as per the reference code)
            enhancement_strength = 4
            
            # Batch processing option
            batch_mode = st.checkbox("Enable Batch Processing", key="batch_mode")
            
            if batch_mode:
                # Allow multiple file uploads
                uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "png", "jpeg"], accept_multiple_files=True, key="batch_uploader")
                
                if uploaded_files:
                    # Create a progress container
                    progress_container = st.empty()
                    progress_text = progress_container.text("Processing images...")
                    
                    # Initialize results
                    enhanced_images = []
                    original_sizes = []
                    enhanced_sizes = []
                    
                    # Process each image
                    for i, uploaded_file in enumerate(uploaded_files):
                        # Update progress
                        progress = (i + 1) / len(uploaded_files)
                        progress_text.text(f"Processing {i + 1}/{len(uploaded_files)} images")
                        
                        try:
                            # Load and process image
                            image = Image.open(uploaded_file)
                            img_array = np.array(image)
                            
                            # Convert BGR to RGB if needed
                            if len(img_array.shape) == 3:
                                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                            
                            # Convert to HSV for color isolation
                            hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
                            
                            # Enhance each color region
                            for key, (lower, upper) in color_ranges.items():
                                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                                hsv[:, :, 1] = cv2.add(hsv[:, :, 1], mask // enhancement_strength)  # Boost saturation
                                hsv[:, :, 2] = cv2.add(hsv[:, :, 2], mask // enhancement_strength)  # Boost brightness
                            
                            # Convert back to BGR
                            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                            
                            # Convert back to RGB for display
                            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                            
                            # Store results
                            enhanced_images.append((uploaded_file.name, enhanced))
                            original_sizes.append(get_size(img_array))
                            enhanced_sizes.append(get_size(enhanced))
                            
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                            continue
                    
                    if enhanced_images:  # Only proceed if we have successfully processed images
                        # Create ZIP file in memory
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                            for filename, enhanced in enhanced_images:
                                # Convert to PNG and add to ZIP
                                _, buffer = cv2.imencode('.png', cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
                                zipf.writestr(f"enhanced_{filename}", buffer.tobytes())
                        
                        # Reset buffer position
                        zip_buffer.seek(0)
                        
                        # Display batch processing results
                        st.subheader("Batch Processing Results")
                        
                        # Overall statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Original Size", f"{sum(original_sizes)/1024/1024:.2f} MB")
                        with col2:
                            st.metric("Total Enhanced Size", f"{sum(enhanced_sizes)/1024/1024:.2f} MB")
                        with col3:
                            avg_ratio = sum(original_sizes) / sum(enhanced_sizes)
                            st.metric("Average Size Ratio", f"{avg_ratio:.2f}x")
                        
                        # Detailed results table
                        st.subheader("Individual Image Results")
                        results_df = pd.DataFrame({
                            'Filename': [f[0] for f in enhanced_images],
                            'Original Size (KB)': [s/1024 for s in original_sizes],
                            'Enhanced Size (KB)': [s/1024 for s in enhanced_sizes],
                            'Size Ratio': [o/e for o, e in zip(original_sizes, enhanced_sizes)]
                        })
                        st.dataframe(results_df)
                        
                        # Download options
                        col1, col2 = st.columns(2)
                        with col1:
                            # Download results as CSV
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name="deficiency_enhancement_results.csv",
                                mime="text/csv",
                                key="deficiency_csv"
                            )
                        
                        with col2:
                            # Download enhanced images as ZIP
                            st.download_button(
                                label="Download Enhanced Images (ZIP)",
                                data=zip_buffer.getvalue(),
                                file_name="deficiency_enhanced_images.zip",
                                mime="application/zip",
                                key="deficiency_zip"
                            )
                        
                        # Clear the buffer
                        zip_buffer.close()
            else:
                # Single image processing
                if uploaded_file:
                    # Convert PIL Image to numpy array
                    img_array = np.array(image)
                    
                    # Convert BGR to RGB if needed
                    if len(img_array.shape) == 3:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    # Convert to HSV for color isolation
                    hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
                    
                    # Enhance each color region
                    for key, (lower, upper) in color_ranges.items():
                        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                        hsv[:, :, 1] = cv2.add(hsv[:, :, 1], mask // enhancement_strength)  # Boost saturation
                        hsv[:, :, 2] = cv2.add(hsv[:, :, 2], mask // enhancement_strength)  # Boost brightness
                    
                    # Convert back to BGR
                    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    
                    # Convert back to RGB for display
                    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                    
                    # Display color ranges information
                    st.markdown("### Color Ranges for Nutrient Deficiencies")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("**Nitrogen (N) Deficiency**")
                        st.markdown("- Yellowish-Green")
                        st.markdown(f"- HSV Range: {color_ranges['N']}")
                    with col2:
                        st.markdown("**Phosphorus (P) Deficiency**")
                        st.markdown("- Purplish")
                        st.markdown(f"- HSV Range: {color_ranges['P']}")
                    with col3:
                        st.markdown("**Potassium (K) Deficiency**")
                        st.markdown("- Reddish-Brown")
                        st.markdown(f"- HSV Range: {color_ranges['K']}")
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        # Convert back to RGB for display
                        original_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                        st.image(original_rgb, caption="Original Image", use_column_width=True)
                    with col2:
                        st.image(enhanced, caption="Enhanced Image", use_column_width=True)
                    
                    # Download options
                    st.download_button(
                        label="Download Enhanced Image (PNG)",
                        data=cv2.imencode('.png', cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))[1].tobytes(),
                        file_name="deficiency_enhanced.png",
                        mime="image/png",
                        key="deficiency_download"
                    )
                else:
                    st.warning("Please upload an image to use deficiency enhancement features.")
            
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Please upload an image to use color enhancement features.")