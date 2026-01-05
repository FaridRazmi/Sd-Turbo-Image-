import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image
import os
import time
from PIL import Image

# cara run : streamlit run sdturboupdate.py

# 1. Setting Folder Simpanan
folder_name = "folder_name"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# 2. Fungsi Load Model (Optimized untuk CPU & Intel)
@st.cache_resource
def load_turbo_model():
    model_id = "stabilityai/sd-turbo"
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        use_safetensors=True
    )

    def dummy_safety_checker(images, **kwargs):
        return images, [False] * len(images)

    pipe.safety_checker = dummy_safety_checker

    if pipe.feature_extractor is None:
        from transformers import CLIPFeatureExtractor
        pipe.feature_extractor = CLIPFeatureExtractor()

    pipe = pipe.to("cpu")
    pipe.enable_attention_slicing()
    return pipe

# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="REID'S AI Art Turbo", page_icon="⚡", layout="wide")

# Inject clean, responsive styles
st.markdown(
    """
    <style>
    :root{
        --bg:#0f1724; /* deep navy */
        --card:#0b1220;
        --muted:#94a3b8;
        --accent:#7c3aed; /* purple */
        --accent-2:#06b6d4; /* cyan */
        --glass: rgba(255,255,255,0.03);
    }
    html, body, .streamlit-expanderHeader {
        background: linear-gradient(180deg, var(--bg), #071022);
        color: #e6eef8;
        font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    .stApp { padding: 20px 24px; }
    .header {
        display:flex; align-items:center; gap:12px; margin-bottom:12px;
    }
    .brand {
        font-weight:700; font-size:20px; color:var(--accent);
    }
    .subtitle { color:var(--muted); margin-top:0; margin-bottom:6px; }
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border-radius:12px; padding:18px; box-shadow: 0 6px 20px rgba(2,6,23,0.6);
        border: 1px solid rgba(255,255,255,0.03);
    }
    .muted { color:var(--muted); font-size:13px; }
    .accent-btn {
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        color: white; padding:8px 14px; border-radius:10px; border: none;
    }
    .small { font-size:13px; color:var(--muted); }
    /* Make image centered with rounded card */
    .result-wrap { display:flex; justify-content:center; }
    .result-card { border-radius:12px; overflow:hidden; background:var(--card); padding:8px; }
    /* Responsive tweaks */
    @media (max-width:600px){
        .header { gap:8px; }
        .brand { font-size:16px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="header">
      <div style="font-size:28px">⚡</div>
      <div>
        <div class="brand">REID'S SD-Turbo Generator (Intel Edition)</div>
        <div class="subtitle">Optimized for CPU & Intel — fast, lightweight, mobile-friendly</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Layout: main + right settings column (stacks automatically on small screens)
main_col, side_col = st.columns([3, 1], gap="medium")

with side_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Settings")
    steps = st.slider("Quality Steps", 1, 4, 2, help="Higher = more detail (uses more RAM)")
    st.caption("High Steps may increase RAM usage And Processing Time")
    st.markdown("**Prompt Presets**")
    preset = st.selectbox("Choose a sample", ["— Select —", "Portrait, dramatic lighting, 35mm", "Landscape, misty, sunrise", "Cyberpunk city, neon signs"], index=0)
    if preset and preset != "— Select —":
        st.caption("Selected preset will prefill prompt area.")
    st.markdown("</div>", unsafe_allow_html=True)

with main_col:
    with st.form("generate_form"):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Describe Your Picture (Prompt)**")
        prompt = st.text_area("Use commas to separate concepts. Put Masterpieces.", value="" if preset == "— Select —" else preset, height=120)
        st.write("", "")
        cols = st.columns([1, 1, 1])
        with cols[0]:
            seed = st.number_input("Seed (0 = random)", min_value=0, max_value=2_000_000_000, value=0)
        with cols[1]:
            width_choice = st.selectbox("Width", ["512", "640", "768"], index=0)
        with cols[2]:
            height_choice = st.selectbox("Height", ["512", "640", "768"], index=0)
        submitted = st.form_submit_button("Generate With REID'S SD-Turbo ⚡")
        st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        if not prompt or prompt.strip() == "":
            st.warning("Please enter a prompt.")
        else:
            start_time = time.time()
            with st.spinner("Generating image — usually under a minute on laptop..."):
                try:
                    pipe = load_turbo_model()
                    result = pipe(
                        prompt=prompt,
                        num_inference_steps=steps,
                        guidance_scale=0.0
                    ).images[0]

                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"turbo_{timestamp}.png"
                    file_path = os.path.join(folder_name, filename)
                    result.save(file_path)

                    duration = round(time.time() - start_time, 2)

                    # Display result in a centered card
                    st.markdown("<div class='result-wrap'>", unsafe_allow_html=True)
                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    st.image(result, caption=f"Ready in {duration} s", use_column_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    st.success(f"Saved: {file_path}")
                    with open(file_path, "rb") as f:
                        st.download_button("💾 Download Picture", f, file_name=filename)

                except Exception as e:
                    st.error(f"Generation error: {e}")

st.markdown("---")

st.markdown('<div class="small muted">Developed by REID | Powered by SD-Turbo ⚡</div>', unsafe_allow_html=True)
