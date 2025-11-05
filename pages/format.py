# simple_format_changer.py
import io
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Simple Format Changer", page_icon="🔄", layout="wide")

st.title("🔄 Simple Image Format Changer")
st.caption("Upload an image, choose a format, and preview both original and converted with size info.")

# Sidebar: choose output format
target_fmt = st.sidebar.selectbox("Output format", ["JPEG", "PNG", "WEBP", "TIFF", "BMP"])

uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg","webp","tif","bmp"])

def convert_image(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    save_params = {"format": fmt}
    if fmt == "JPEG" and img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(buf, **save_params)
    return buf.getvalue()

if uploaded:
    # Original
    orig_bytes = uploaded.getbuffer().nbytes
    orig_img = Image.open(uploaded)

    # Convert
    converted_bytes = convert_image(orig_img, target_fmt)
    conv_img = Image.open(io.BytesIO(converted_bytes))

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(orig_img, use_container_width=True)
        st.write(f"**Format:** {orig_img.format or uploaded.type.split('/')[-1].upper()}")
        st.write(f"**Size:** {orig_bytes/1024:.2f} KB")

    with col2:
        st.subheader(f"Converted → {target_fmt}")
        st.image(conv_img, use_container_width=True)
        st.write(f"**Format:** {target_fmt}")
        st.write(f"**Size:** {len(converted_bytes)/1024:.2f} KB")

        st.download_button(
            "⬇️ Download Converted Image",
            data=converted_bytes,
            file_name=f"converted.{target_fmt.lower()}",
            mime=f"image/{target_fmt.lower()}"
        )
else:
    st.info("Upload an image to start converting.")
