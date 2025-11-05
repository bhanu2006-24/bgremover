# img_optimizer_pro.py
import io
import zipfile
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Image Optimizer Pro", page_icon="🗜️", layout="wide")

st.title("🗜️ Image Optimizer Pro")
st.caption("Compress, resize, convert formats, strip metadata, and target file size.")

# Sidebar controls
with st.sidebar:
    st.subheader("Compression Settings")
    fmt = st.selectbox("Output format", ["JPEG", "PNG", "WEBP"])
    mode = st.radio("Compression mode", ["Quality slider", "Target size (KB)"])

    if mode == "Quality slider":
        quality = st.slider("Quality (lower = smaller size)", 10, 95, 70)
    else:
        target_kb = st.number_input("Target size (KB)", min_value=10, value=200)

    optimize = st.checkbox("Optimize", True)

    st.subheader("Resize")
    resize = st.checkbox("Resize image(s)", False)
    if resize:
        new_width = st.number_input("Width (px)", value=800, min_value=50)
        new_height = st.number_input("Height (px)", value=600, min_value=50)

    st.subheader("Other")
    strip_metadata = st.checkbox("Strip metadata (EXIF)", True)
    batch_mode = st.checkbox("Batch mode (multiple images)", False)

# File uploader
uploaded = st.file_uploader(
    "Upload image(s)", 
    type=["png", "jpg", "jpeg", "webp"], 
    accept_multiple_files=batch_mode
)

def compress_to_target(img: Image.Image, fmt: str, target_kb: int, optimize: bool) -> bytes:
    """Binary search quality until file size <= target_kb."""
    lo, hi = 10, 95
    best_bytes = None
    while lo <= hi:
        mid = (lo + hi) // 2
        buffer = io.BytesIO()
        save_params = {}
        if fmt == "JPEG":
            save_params = {"format": "JPEG", "quality": mid, "optimize": optimize}
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
        elif fmt == "PNG":
            save_params = {"format": "PNG", "optimize": optimize}
        elif fmt == "WEBP":
            save_params = {"format": "WEBP", "quality": mid, "method": 6}
        img.save(buffer, **save_params)
        data = buffer.getvalue()
        size_kb = len(data) / 1024
        if size_kb <= target_kb:
            best_bytes = data
            hi = mid - 1
        else:
            lo = mid + 1
    return best_bytes if best_bytes else data

def process_image(img: Image.Image) -> bytes:
    # Resize
    if resize:
        img = img.resize((int(new_width), int(new_height)), Image.LANCZOS)

    # Strip metadata
    if strip_metadata:
        data = list(img.getdata())
        img_no_meta = Image.new(img.mode, img.size)
        img_no_meta.putdata(data)
        img = img_no_meta

    # Save compressed
    if mode == "Quality slider":
        buffer = io.BytesIO()
        save_params = {}
        if fmt == "JPEG":
            save_params = {"format": "JPEG", "quality": quality, "optimize": optimize}
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
        elif fmt == "PNG":
            save_params = {"format": "PNG", "optimize": optimize}
        elif fmt == "WEBP":
            save_params = {"format": "WEBP", "quality": quality, "method": 6}
        img.save(buffer, **save_params)
        return buffer.getvalue()
    else:
        return compress_to_target(img, fmt, target_kb, optimize)

if uploaded:
    if batch_mode:
        # Multiple images
        results = []
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in uploaded:
                img = Image.open(file)
                compressed_bytes = process_image(img)
                fname = f"{file.name.rsplit('.',1)[0]}.{fmt.lower()}"
                zf.writestr(fname, compressed_bytes)
                results.append((file.name, img, compressed_bytes))

        st.subheader("Preview (first image)")
        if results:
            orig_name, orig_img, comp_bytes = results[0]
            col1, col2 = st.columns(2)
            with col1:
                st.image(orig_img, use_container_width=True, caption=f"Original: {orig_name}")
                st.write(f"Size: {file.size/1024:.2f} KB")
            with col2:
                st.image(Image.open(io.BytesIO(comp_bytes)), use_container_width=True, caption="Compressed")
                st.write(f"Size: {len(comp_bytes)/1024:.2f} KB")

        st.download_button(
            "⬇️ Download All (ZIP)",
            data=zip_buf.getvalue(),
            file_name="compressed_images.zip",
            mime="application/zip"
        )

    else:
        # Single image
        img = Image.open(uploaded)
        compressed_bytes = process_image(img)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(img, use_container_width=True)
            st.write(f"Size: {uploaded.size/1024:.2f} KB")

        with col2:
            st.subheader("Compressed")
            st.image(Image.open(io.BytesIO(compressed_bytes)), use_container_width=True)
            st.write(f"Size: {len(compressed_bytes)/1024:.2f} KB")

        st.download_button(
            "⬇️ Download Compressed Image",
            data=compressed_bytes,
            file_name=f"compressed.{fmt.lower()}",
            mime=f"image/{fmt.lower()}"
        )
else:
    st.info("Upload one or more images to begin.")
