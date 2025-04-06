# 🌿 Cotton Leaf Image Analysis Tool

A comprehensive image processing and analysis tool specifically designed for cotton leaf images, featuring various image processing techniques, compression methods, background removal, and color enhancement capabilities.

## 🚀 Features

### 1. Image Processing
- **Basic Processing**
  - Image Negation
  - Thresholding
  - Bit Plane Slicing
  - Brightness Adjustment
  - Contrast Adjustment

- **Enhancement Techniques**
  - Histogram Equalization
  - Adaptive Histogram Equalization
  - Gaussian Blur
  - Sharpening
  - High Boost Filter
  - Median Filter
  - Bilateral Filter

- **Color Processing**
  - Color Channels
  - Color Spaces (RGB, HSV, LAB, YCrCb)
  - Color Quantization
  - Color Balance

### 2. Image Compression
- **Basic Compression**
  - Run-Length Encoding
  - Huffman Coding
  - JPEG Quality Control

- **Transform Compression**
  - Discrete Cosine Transform (DCT)
  - Discrete Wavelet Transform (DWT)

- **Advanced Techniques**
  - Vector Quantization
  - Fractal Compression

- **Batch Processing**
  - Process multiple images simultaneously
  - Support for various compression methods
  - Download results as CSV
  - Download compressed images as ZIP

### 3. Background Removal
- **Color-Based Removal**
  - RGB color space selection
  - HSV color space selection
  - LAB color space selection
  - Adjustable color thresholds
  - Morphological operations for cleanup

- **Edge-Based Removal**
  - Canny edge detection
  - Contour finding
  - Adjustable edge thresholds
  - Morphological operations for cleanup

- **AI-Based Removal** (Coming Soon)
  - Automatic background detection
  - High-quality edge preservation
  - Support for complex backgrounds

### 4. Color Enhancement
- **Selective Color Enhancement**
  - RGB color enhancement
  - HSV color enhancement
  - LAB color enhancement
  - Adjustable enhancement parameters

- **Color Balance**
  - Red channel adjustment
  - Green channel adjustment
  - Blue channel adjustment

- **Color Space Enhancement**
  - RGB contrast and brightness
  - HSV saturation and value
  - LAB lightness and colorfulness
  - YCrCb luminance and chroma

- **Deficiency Enhancement**
  - Nitrogen (N) deficiency enhancement
  - Phosphorus (P) deficiency enhancement
  - Potassium (K) deficiency enhancement
  - Batch processing support
  - Download enhanced images as ZIP

## 🛠️ Technical Details

### Color Ranges for Nutrient Deficiencies
- **Nitrogen (N) Deficiency**
  - Color: Yellowish-Green
  - HSV Range: [(25, 40, 40), (35, 255, 255)]

- **Phosphorus (P) Deficiency**
  - Color: Purplish
  - HSV Range: [(125, 40, 40), (160, 255, 255)]

- **Potassium (K) Deficiency**
  - Color: Reddish-Brown
  - HSV Range: [(5, 40, 40), (20, 255, 255)]

### Image Processing Pipeline
1. Image Upload
2. Color Space Conversion
3. Enhancement/Processing
4. Result Display
5. Download Options

## 📦 Installation

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app2.py
```

## 📋 Requirements
- Python 3.7+
- OpenCV
- NumPy
- Streamlit
- Pandas
- Matplotlib
- PyWavelets
- Pillow

## 🎯 Usage

1. Launch the application
2. Upload an image through the sidebar
3. Select the desired section from the navigation menu
4. Choose the specific processing technique
5. Adjust parameters as needed
6. View results and download processed images

## 📝 Notes
- The application supports JPG, PNG, and JPEG image formats
- Batch processing is available for compression and deficiency enhancement
- All processed images can be downloaded in PNG format
- CSV reports are available for batch processing results

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
- OpenCV for image processing capabilities
- Streamlit for the web interface
- All contributors and users of this tool
