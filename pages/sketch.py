# app.py
import io
import zipfile
from typing import Tuple, Dict

import numpy as np
import streamlit as st
from PIL import Image
import cv2


# ---------- Utils ----------
def pil_to_cv(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(img: np.ndarray) -> Image.Image:
    if len(img.shape) == 2:
        return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def clamp_kernel(k: int) -> int:
    # Ensure odd and >= 3
    return max(3, k + (1 - k % 2))


# ---------- Sketch engines ----------
def classic_pencil_sketch(img_bgr: np.ndarray, blur_kernel: int = 21, blend_scale: float = 1.0) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    k = clamp_kernel(blur_kernel)
    blur = cv2.GaussianBlur(inv, (k, k), 0)
    dodge = cv2.divide(gray, 255 - blur, scale=256 * blend_scale)
    return dodge

def soft_pencil_sketch(img_bgr: np.ndarray, blur_kernel: int = 11, tone: float = 0.9, contrast: float = 1.1) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    k = clamp_kernel(blur_kernel)
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    detail = cv2.addWeighted(gray, contrast, blur, - (contrast - 1), 0)
    inv = 255 - detail
    blur2 = cv2.GaussianBlur(inv, (k, k), 0)
    dodge = cv2.divide(detail, 255 - blur2, scale=256 * tone)
    return dodge

def edge_sketch(img_bgr: np.ndarray, thresh1: int = 60, thresh2: int = 120, invert: bool = True) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, thresh1, thresh2)
    if invert:
        edges = 255 - edges
    return edges

def color_pencil_sketch(img_bgr: np.ndarray, sigma_s: int = 50, sigma_r: float = 0.1, shade_factor: float = 0.03) -> np.ndarray:
    # OpenCV pencilSketch requires grayscale and color outputs; stylization is a premium color pencil feel
    # Try native pencilSketch if available; otherwise fallback
    try:
        gray, color = cv2.pencilSketch(img_bgr, sigma_s=sigma_s, sigma_r=sigma_r, shade_factor=shade_factor)
        return color
    except Exception:
        stylized = cv2.stylization(img_bgr, sigma_s=sigma_s, sigma_r=sigma_r)
        return stylized


# ---------- Post-processing ----------
def adjust_levels(img: np.ndarray, brightness: int = 0, contrast: int = 0) -> np.ndarray:
    # Works for both grayscale and BGR; keep in uint8 range
    out = cv2.convertScaleAbs(img, alpha=1 + contrast / 100.0, beta=brightness)
    return out

def reduce_noise(img: np.ndarray, strength: int = 0) -> np.ndarray:
    if strength <= 0:
        return img
    if len(img.shape) == 2:
        return cv2.GaussianBlur(img, (clamp_kernel(strength), clamp_kernel(strength)), 0)
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Image → Sketch", page_icon="✏️", layout="wide")

st.title("Image → Sketch ✏️")
st.caption("Premium, controllable pencil and edge sketches. Upload, tweak, download.")

with st.sidebar:
    st.subheader("Controls")
    mode = st.selectbox("Sketch mode", ["Classic pencil", "Soft pencil", "Color pencil", "Edge sketch"])

    # Common controls
    brightness = st.slider("Brightness", -50, 50, 0)
    contrast = st.slider("Contrast", -50, 50, 10)
    denoise = st.slider("Noise reduction", 0, 15, 4)

    # Mode-specific controls
    if mode in ["Classic pencil", "Soft pencil"]:
        blur_kernel = st.slider("Blur kernel size", 3, 51, 21, step=2)
    if mode == "Classic pencil":
        blend_scale = st.slider("Blend intensity", 0.5, 2.0, 1.0)
    if mode == "Soft pencil":
        tone = st.slider("Tone (softness)", 0.5, 1.5, 0.9)
        soft_contrast = st.slider("Detail contrast", 0.8, 1.8, 1.1)
    if mode == "Edge sketch":
        t1 = st.slider("Edge threshold 1", 0, 255, 60)
        t2 = st.slider("Edge threshold 2", 0, 255, 120)
        invert_edges = st.checkbox("Invert (white paper)", True)
    if mode == "Color pencil":
        sigma_s = st.slider("Spatial sigma", 10, 200, 50)
        sigma_r = st.slider("Range sigma", 0.05, 0.8, 0.1)
        shade_factor = st.slider("Shade factor", 0.01, 0.1, 0.03)

    st.divider()
    auto_compare = st.checkbox("Generate all variants for comparison", False)

uploaded = st.file_uploader("Upload an image (PNG/JPG/WebP)", type=["png", "jpg", "jpeg", "webp"])

col_left, col_right = st.columns([1, 1])

def process_single(img_bgr: np.ndarray, mode: str) -> Tuple[np.ndarray, str]:
    if mode == "Classic pencil":
        sk = classic_pencil_sketch(img_bgr, blur_kernel=blur_kernel, blend_scale=blend_scale)
    elif mode == "Soft pencil":
        sk = soft_pencil_sketch(img_bgr, blur_kernel=blur_kernel, tone=tone, contrast=soft_contrast)
    elif mode == "Edge sketch":
        sk = edge_sketch(img_bgr, thresh1=t1, thresh2=t2, invert=invert_edges)
    elif mode == "Color pencil":
        sk = color_pencil_sketch(img_bgr, sigma_s=sigma_s, sigma_r=sigma_r, shade_factor=shade_factor)
    else:
        sk = classic_pencil_sketch(img_bgr)
    sk = reduce_noise(sk, denoise)
    sk = adjust_levels(sk, brightness=brightness, contrast=contrast)
    label = mode
    return sk, label

def process_all(img_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    results = {}
    results["Classic pencil"] = adjust_levels(
        reduce_noise(classic_pencil_sketch(img_bgr, blur_kernel=21, blend_scale=1.0), denoise),
        brightness=brightness, contrast=contrast
    )
    results["Soft pencil"] = adjust_levels(
        reduce_noise(soft_pencil_sketch(img_bgr, blur_kernel=11, tone=0.9, contrast=1.1), denoise),
        brightness=brightness, contrast=contrast
    )
    results["Edge sketch"] = adjust_levels(
        reduce_noise(edge_sketch(img_bgr, thresh1=60, thresh2=120, invert=True), denoise),
        brightness=brightness, contrast=contrast
    )
    results["Color pencil"] = adjust_levels(
        reduce_noise(color_pencil_sketch(img_bgr, sigma_s=50, sigma_r=0.1, shade_factor=0.03), denoise),
        brightness=brightness, contrast=contrast
    )
    return results

if uploaded is not None:
    pil_img = Image.open(uploaded)
    img_bgr = pil_to_cv(pil_img)

    with col_left:
        st.subheader("Original")
        st.image(pil_img, use_container_width=True)

    with col_right:
        st.subheader("Sketch")
        if not auto_compare:
            sketch, label = process_single(img_bgr, mode)
            st.image(cv_to_pil(sketch), use_container_width=True, caption=label)

            # Download single
            buffer = io.BytesIO()
            cv_to_pil(sketch).save(buffer, format="PNG")
            st.download_button(
                "Download sketch (PNG)",
                data=buffer.getvalue(),
                file_name=f"sketch_{label.lower().replace(' ', '_')}.png",
                mime="image/png"
            )
        else:
            results = process_all(img_bgr)
            for label, arr in results.items():
                st.image(cv_to_pil(arr), use_container_width=True, caption=label)

            # Zip download
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for label, arr in results.items():
                    b = io.BytesIO()
                    cv_to_pil(arr).save(b, format="PNG")
                    zf.writestr(f"{label.lower().replace(' ', '_')}.png", b.getvalue())
            st.download_button(
                "Download all variants (ZIP)",
                data=zip_buf.getvalue(),
                file_name="sketch_variants.zip",
                mime="application/zip"
            )
else:
    st.info("Upload an image to begin. Try a portrait or product shot for best results.")
