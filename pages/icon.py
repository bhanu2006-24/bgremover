# icon_generator_app.py
import io
import zipfile
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Icon Generator", page_icon="🪟", layout="wide")

st.title("🪟 Icon Generator")
st.caption("Upload an image and generate multiple icon sizes for apps, websites, or favicons.")

# Standard icon sizes
ICON_SIZES = [16, 32, 64, 128, 256, 512]

uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg","webp"])

if uploaded:
    orig_img = Image.open(uploaded).convert("RGBA")

    st.subheader("Original")
    st.image(orig_img, use_container_width=True)
    st.write(f"**Format:** {orig_img.format or uploaded.type.split('/')[-1].upper()}")
    st.write(f"**Size:** {uploaded.size/1024:.2f} KB")
    st.write(f"**Dimensions:** {orig_img.width} × {orig_img.height}")

    # Generate icons
    previews = []
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for size in ICON_SIZES:
            resized = orig_img.resize((size, size), Image.LANCZOS)
            buf = io.BytesIO()
            resized.save(buf, format="PNG")
            fname = f"icon_{size}x{size}.png"
            zf.writestr(fname, buf.getvalue())
            previews.append((size, resized, buf.getvalue()))

    st.subheader("Generated Icons")
    cols = st.columns(len(previews))
    for col, (size, img, data) in zip(cols, previews):
        with col:
            st.image(img, caption=f"{size}×{size}", use_container_width=True)
            st.write(f"{len(data)/1024:.1f} KB")

    st.download_button(
        "⬇️ Download All Icons (ZIP)",
        data=zip_buf.getvalue(),
        file_name="icons.zip",
        mime="application/zip"
    )
else:
    st.info("Upload an image to generate icons.")
