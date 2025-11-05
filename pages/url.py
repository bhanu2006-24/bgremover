# img_to_url_app.py
import requests
import base64
import streamlit as st

API_KEY = "a719fb010c12a4415623cf6b57103485"
UPLOAD_URL = "https://api.imgbb.com/1/upload"

st.set_page_config(page_title="Image → URL (ImgBB)", page_icon="🌐", layout="centered")

st.title("🌐 Image → URL Creator")
st.caption("Upload an image and get a shareable URL via ImgBB API.")

# --- Safety / Tips Section ---
st.warning(
    "⚠️ **Important Tips & Warnings**\n\n"
    "- Uploaded images are stored on **ImgBB servers**, not locally.\n"
    "- Anyone with the generated link can view your image.\n"
    "- Avoid uploading **sensitive, private, or confidential images**.\n"
    "- For best results, keep images under **5 MB**.\n"
    "- Use JPG/PNG/WebP for maximum compatibility."
)

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])

if uploaded:
    # Read file and encode to base64
    img_bytes = uploaded.read()
    encoded = base64.b64encode(img_bytes)

    with st.spinner("Uploading to ImgBB..."):
        response = requests.post(
            UPLOAD_URL,
            params={"key": API_KEY},
            data={"image": encoded}
        )

    if response.status_code == 200:
        data = response.json()
        url = data["data"]["url"]
        display_url = data["data"]["display_url"]

        st.success("✅ Upload successful!")
        st.image(uploaded, caption="Uploaded Image", use_container_width=True)

        st.write("**Direct URL:**")
        st.code(url, language="text")

        st.write("**Display URL:**")
        st.code(display_url, language="text")

        st.markdown(f"[🔗 Open Image in Browser]({url})")
    else:
        st.error(f"❌ Upload failed: {response.text}")
else:
    st.info("Upload an image to generate a URL.")
