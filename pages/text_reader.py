# super_ocr_reader.py
import io
import os
import zipfile
from typing import List, Tuple, Dict

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import cv2
import pytesseract

# Optional: point pytesseract to tesseract executable on Windows
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="Super OCR Reader", page_icon="📖", layout="wide")
st.title("📖 Super OCR Reader")
st.caption("Multi-language OCR with preprocessing, region highlights, batch mode, and searchable PDF export.")

# ---------- Sidebar controls ----------
with st.sidebar:
    st.subheader("OCR settings")
    langs = st.multiselect(
        "Languages (install traineddata in Tesseract)",
        options=["eng", "hin", "fra", "spa", "deu"],
        default=[]   # nothing selected by default
    )
    oem = st.selectbox("OCR Engine Mode (OEM)", [0, 1, 2, 3], index=3)
    psm = st.selectbox(
        "Page Segmentation Mode (PSM)",
        options=[3, 4, 6, 11, 12, 13],
        index=2
    )
    dpi = st.number_input("DPI hint", min_value=72, max_value=600, value=300)

    st.subheader("Preprocessing")
    use_grayscale = st.checkbox("Grayscale", False)
    bin_thresh = st.checkbox("Binarize (adaptive threshold)", False)
    denoise = st.checkbox("Denoise (bilateral)", False)
    sharpen = st.checkbox("Sharpen", False)
    deskew = st.checkbox("Deskew", False)

    st.subheader("Visualization")
    show_boxes = st.checkbox("Show text regions", False)

    st.subheader("Batch & Export")
    batch_mode = st.checkbox("Batch mode (multiple images)", False)
    export_pdf = st.checkbox("Export searchable PDF", False)

uploaded = st.file_uploader(
    "Upload image(s)",
    type=["png", "jpg", "jpeg", "webp", "tif", "tiff"],
    accept_multiple_files=batch_mode
)

# ---------- Utilities ----------
def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def cv_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def auto_deskew(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return img_bgr
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess(img_bgr: np.ndarray, cfg: Dict) -> np.ndarray:
    out = img_bgr.copy()
    if cfg["deskew"]:
        out = auto_deskew(out)
    if cfg["grayscale"]:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    if cfg["binarize"]:
        if len(out.shape) == 2:
            out = cv2.adaptiveThreshold(out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 35, 11)
        else:
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            out = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 35, 11)
    if cfg["denoise"]:
        if len(out.shape) == 2:
            out = cv2.fastNlMeansDenoising(out, None, 12, 7, 21)
        else:
            out = cv2.bilateralFilter(out, d=9, sigmaColor=75, sigmaSpace=75)
    if cfg["sharpen"]:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        if len(out.shape) == 2:
            out = cv2.filter2D(out, -1, kernel)
        else:
            out = cv2.filter2D(out, -1, kernel)
    # Ensure 3-channel BGR for visualization downstream
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return out

def ocr_config(oem: int, psm: int, dpi: int) -> str:
    return f"--oem {oem} --psm {psm} -c dpi={dpi}"

def run_ocr(img_pil: Image.Image, lang_list: List[str], cfg: str) -> str:
    lang = "+".join(lang_list) if lang_list else "eng"
    return pytesseract.image_to_string(img_pil, lang=lang, config=cfg)

def detect_boxes(img_pil: Image.Image, lang_list: List[str], cfg: str) -> List[Dict]:
    lang = "+".join(lang_list) if lang_list else "eng"
    data = pytesseract.image_to_data(img_pil, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)
    boxes = []
    n = len(data["level"])
    for i in range(n):
        text = data["text"][i]
        conf = data["conf"][i]
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        if conf != "-1" and text.strip():
            boxes.append({"text": text, "conf": float(conf), "bbox": (x, y, w, h)})
    return boxes

def draw_boxes(img_bgr: np.ndarray, boxes: List[Dict]) -> np.ndarray:
    out = img_bgr.copy()
    for b in boxes:
        x, y, w, h = b["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 165, 255), 2)  # orange
    return out

def make_searchable_pdf(pages: List[Image.Image], lang_list: List[str], cfg: str) -> bytes:
    # Use pytesseract to convert each image into a single PDF page with text overlay
    pdf_bytes = io.BytesIO()
    # Accumulate PDFs
    pdf_pages = []
    lang = "+".join(lang_list) if lang_list else "eng"
    for pg in pages:
        pdf = pytesseract.image_to_pdf_or_hocr(pg, lang=lang, config=cfg, extension='pdf')
        pdf_pages.append(pdf)
    # Merge PDFs (simple concat for single-page outputs works for OCR PDFs)
    # For robust merging, use PyPDF2, but many viewers accept concatenation.
    for p in pdf_pages:
        pdf_bytes.write(p)
    return pdf_bytes.getvalue()

# ---------- Processing ----------
def process_single(file) -> Tuple[str, Image.Image, Image.Image, str, List[Dict]]:
    pil = Image.open(file).convert("RGB")
    bgr = pil_to_cv(pil)
    cfg = {
        "grayscale": use_grayscale,
        "binarize": bin_thresh,
        "denoise": denoise,
        "sharpen": sharpen,
        "deskew": deskew,
    }
    pre_bgr = preprocess(bgr, cfg)
    pre_pil = cv_to_pil(pre_bgr)

    cfg_str = ocr_config(oem, psm, dpi)
    text = run_ocr(pre_pil, langs, cfg_str)
    boxes = detect_boxes(pre_pil, langs, cfg_str) if show_boxes else []

    vis_bgr = draw_boxes(pre_bgr, boxes) if show_boxes else pre_bgr
    vis_pil = cv_to_pil(vis_bgr)

    return file.name, pil, vis_pil, text, boxes

# ---------- UI ----------
if uploaded:
    if batch_mode:
        results = []
        for f in uploaded:
            results.append(process_single(f))

        st.subheader("Batch preview")
        for name, orig_pil, vis_pil, text, boxes in results[:3]:
            col1, col2 = st.columns(2)
            with col1:
                st.image(orig_pil, use_container_width=True, caption=f"Original: {name}")
            with col2:
                st.image(vis_pil, use_container_width=True, caption="Preprocessed + regions")
            st.text_area(f"Extracted text: {name}", text, height=200)

        # ZIP download (texts)
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, _, _, text, _ in results:
                base = os.path.splitext(name)[0]
                zf.writestr(f"{base}.txt", text.encode("utf-8"))
        st.download_button(
            "⬇️ Download all texts (ZIP)",
            data=zip_buf.getvalue(),
            file_name="ocr_texts.zip",
            mime="application/zip"
        )

        # PDF export (searchable)
        if export_pdf:
            pages = [r[2] for r in results]  # use preprocessed visualization image as page
            pdf_bytes = make_searchable_pdf(pages, langs, ocr_config(oem, psm, dpi))
            st.download_button(
                "⬇️ Download searchable PDF (merged)",
                data=pdf_bytes,
                file_name="ocr_searchable.pdf",
                mime="application/pdf"
            )

    else:
        name, orig_pil, vis_pil, text, boxes = process_single(uploaded)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(orig_pil, use_container_width=True)
        with col2:
            st.subheader("Preprocessed" + (" + regions" if show_boxes else ""))
            st.image(vis_pil, use_container_width=True)

        st.subheader("Extracted text")
        st.text_area("Text output", text, height=300)

        st.download_button(
            "⬇️ Download as TXT",
            data=text,
            file_name=f"{os.path.splitext(name)[0]}.txt",
            mime="text/plain"
        )

        if export_pdf:
            pdf_bytes = make_searchable_pdf([vis_pil], langs, ocr_config(oem, psm, dpi))
            st.download_button(
                "⬇️ Download searchable PDF",
                data=pdf_bytes,
                file_name=f"{os.path.splitext(name)[0]}.pdf",
                mime="application/pdf"
            )
else:
    st.info("Upload one or more images to begin OCR.")
