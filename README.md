# 🖼️ Premium Image Toolkit

A collection of **Streamlit web apps** for professional‑grade image processing — built entirely with Python libraries (no external APIs).  
From background removal to format conversion, resizing, sketch effects, OCR, and creative generators, this toolkit is designed to be **modular, offline, and premium‑quality**.



## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bs-img-toolkit.streamlit.app/)

Check out the deployed app here:  
👉 [bs-img-toolkit.streamlit.app](https://bs-img-toolkit.streamlit.app/)

---

## ✨ Included Apps

### 🪄 Background Remover
- Remove image backgrounds using [rembg](https://github.com/danielgatis/rembg) (U²‑Net).  
- Replace with transparent, solid color, custom image, or blurred background.  
- Styling: soft shadow, subject outline, circle crop, rounded corners.  
- Batch processing + ZIP download.

---

### 🔄 Format Changer
- Convert between PNG, JPG, WEBP, TIFF, BMP.  
- Side‑by‑side preview of **original vs converted**.  
- Shows format + file size for both.  
- One‑click download.

---

### 📏 Size / Resizer
- Resize by width/height, percentage, or max dimension.  
- Maintains aspect ratio.  
- Side‑by‑side preview with dimensions + file size.  

---

### ✏️ Sketch Converter
- Convert photos into pencil sketch or cartoon‑style images.  
- Adjustable intensity.  
- Side‑by‑side preview + download.

---

### 🔤 Text Reader (OCR)
- Extract text from images using [pytesseract](https://github.com/madmaze/pytesseract).  
- Supports multiple languages.  
- Copy text or export as `.txt`.

---

### 🌐 Image → URL Converter
- Upload an image → instantly generate a shareable URL.  
- Useful for embedding in docs, websites, or APIs.  

---

### 🪟 Icon Generator
- Generate multiple icon sizes (16×16, 32×32, 64×64, 128×128, 256×256, 512×512).  
- Preview all sizes side‑by‑side.  
- Download all icons as a ZIP.  

---

### 🔠 ASCII Art Converter
- Convert images into ASCII text art.  
- Adjustable resolution (detail level).  
- Export as `.txt` or styled HTML.  

---

### 🕶️ Blur / Pixelate Tool
- Apply blur or pixelation for privacy.  
- Adjustable intensity.  
- (Planned upgrade: brush mode for selective blur).  

---

### ✍️ Premium Text Adder
- Add multiple text overlays (captions, watermarks, memes).  
- Control font, size, color, and position (X/Y).  
- Save multiple text layers.  
- Side‑by‑side preview + final export.  
- (Planned upgrade: drag‑and‑drop positioning).  

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/bhanu2006-24/image-toolkit
cd image-toolkit

# Create a virtual environment (recommended)
conda create -n imagetools python=3.11
conda activate imagetools

# Install requirements
pip install -r requirements.txt
