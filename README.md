# 🖼️ Premium Image Toolkit

A collection of **7 Streamlit web apps** for professional‑grade image processing — built entirely with Python libraries (no external APIs).  
This toolkit is designed to be **modular, offline, and premium‑quality**, covering everything from background removal to OCR and icon generation.

👉 **Live Demo:** [bs-img-toolkit.streamlit.app](https://bs-img-toolkit.streamlit.app/)

---

## ✨ Included Apps

### 🌿 Background Remover
- Remove image backgrounds using [rembg](https://github.com/danielgatis/rembg) (U²‑Net).  
- Replace with transparent, solid color, custom image, or blurred background.  
- Styling options: soft shadow, subject outline, circle crop, rounded corners.  
- Batch processing + ZIP download.  

### 🔄 Format Converter
- Convert between PNG, JPG, WEBP, TIFF, BMP.  
- Side‑by‑side preview of **original vs converted**.  
- Shows format + file size for both.  
- One‑click download.  

### 📏 Resizer
- Resize by width/height, percentage, or max dimension.  
- Maintains aspect ratio.  
- Side‑by‑side preview with dimensions + file size.  

### ✏️ Sketch Converter
- Convert photos into pencil sketch or cartoon‑style images.  
- Adjustable intensity.  
- Side‑by‑side preview + download.  

### 🔥 OCR Text Reader
- Extract text from images using [pytesseract](https://github.com/madmaze/pytesseract).  
- Supports multiple languages.  
- Copy text or export as `.txt`.  

### 🌐 Image → URL Converter
- Upload an image → instantly generate a shareable URL.  
- Useful for embedding in docs, websites, or APIs.  

### 🪙 Icon Generator
- Generate multiple icon sizes (16×16, 32×32, 64×64, 128×128, 256×256, 512×512).  
- Preview all sizes side‑by‑side.  
- Download all icons as a ZIP.  

---

## 📸 Screenshots

*(Replace these placeholders with actual screenshots saved in `assets/`)*

- Background Remover → 
- ![Background Remover](assets/background1.png)  
- ![Background Remover](assets/background2.png) 
- Format Converter → ![Format Converter](assets/format.png)  
- Resizer → 
- ![Resizer](assets/size1.png)  
- ![Resizer](assets/size2.png) 
- Sketch Converter → ![Sketch Converter](assets/sketch.png)  
- OCR → ![OCR](assets/ocr.png)  
- Image → URL Converter → 
- ![Image to URL](assets/url1.png) 
- ![Image to URL](assets/url2.png)   
- Icon Generator → 
- ![Icon Generator](assets/icon1.png)  
- ![Icon Generator](assets/icon2.png)  

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/bhanu2006-24/img-toolkit.git
cd img-toolkit

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

# Install requirements
pip install -r requirements.txt

# Run the app
streamlit run app.py
