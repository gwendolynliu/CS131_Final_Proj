import streamlit as st
import json
import os
import re
import html
import tempfile

st.set_page_config(
    page_title="Seeing In Verses",
    layout="wide",
    initial_sidebar_state="expanded",
)

HEADER_H = "180px"

st.markdown(f"""
<style>
    /* hide Streamlit chrome */
    [data-testid="stHeader"] {{ display: none !important; }}

    /* permanently show sidebar — force it out of collapsed position */
    section[data-testid="stSidebar"] {{
        transform: translateX(0) !important;
        min-width: 244px !important;
        width: 244px !important;
    }}
    /* hide both the collapse button and the re-expand arrow */
    [data-testid="stSidebarCollapseButton"] {{ display: none !important; }}
    [data-testid="collapsedControl"] {{ display: none !important; }}
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}

    /* zero out Streamlit's default top padding — spacer div handles clearance instead */
    .main .block-container,
    [data-testid="stMainBlockContainer"],
    .block-container {{
        padding-top: 0 !important;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 1300px;
    }}
    section[data-testid="stSidebar"] > div:first-child,
    [data-testid="stSidebarContent"] {{
        padding-top: 210px !important;
    }}

    /* overall page background */
    .stApp {{ background: #faf9f7; }}

    /* sidebar */
    section[data-testid="stSidebar"] {{
        background: #f4f3f0;
        border-right: 1px solid #e0ddd8;
    }}

    /* serif body default */
    .stMarkdown, .stText {{ font-family: Georgia, "Times New Roman", serif; }}

    /* full-width fixed header */
    .site-header {{
        position: fixed;
        top: 0; left: 0; right: 0;
        height: {HEADER_H};
        z-index: 999999;
        background: #111111;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border-bottom: 3px solid #c8973a;
    }}
</style>

<div class="site-header">
    <div style="
        font-family: Georgia, serif;
        font-size: 2.6em;
        font-weight: 700;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        color: #ffffff;
        line-height: 1.1;
    ">Seeing in Verses</div>
    <div style="
        font-family: system-ui, -apple-system, sans-serif;
        font-size: 0.78em;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #888888;
        margin-top: 12px;
    ">Vision-Based Poetry Retrieval</div>
</div>
""", unsafe_allow_html=True)

TEST_IMAGES_DIR = "test_images"
PRECOMPUTED_FILE = "cache/precomputed_results.json"

with open(PRECOMPUTED_FILE) as f:
    precomputed = json.load(f)


@st.cache_resource
def load_retrieval():
    from integrate import parallel_retrieve, rerank_retrieve
    return parallel_retrieve, rerank_retrieve


@st.cache_data
def run_retrieval_custom(image_path, strategy):
    parallel_retrieve, rerank_retrieve = load_retrieval()
    if strategy == "rerank":
        return rerank_retrieve(image_path, top_k=1)
    return parallel_retrieve(image_path, mode=strategy, top_k=1)


def normalize_poem(text):
    # poems come in two formats from the CSV:
    #   double-newline style: \n\n = line break, \n\n\n\n = stanza break
    #   single-newline style: \n = line break, \n\n = stanza break
    # normalize both to: \n = line break, \n\n = stanza break
    if "\n\n" in text:
        text = re.sub(r"\n{4,}", "\x00", text)
        text = text.replace("\n\n", "\n")
        text = text.replace("\x00", "\n\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def poem_to_html(text):
    text = normalize_poem(text)
    stanzas = text.split("\n\n")
    p_style = 'style="margin: 0 0 1.4em 0; padding: 0; line-height: 1.7;"'
    parts = []
    for stanza in stanzas:
        lines_escaped = [html.escape(line) for line in stanza.split("\n")]
        parts.append(f"<p {p_style}>{'<br>'.join(lines_escaped)}</p>")
    return "\n".join(parts)


# ---- load metadata ----
with open("test_images_metadata.json") as f:
    metadata = json.load(f)

image_labels = [f"{m['filename']} — {m['description']}" for m in metadata]

# ---- sidebar ----
selected_idx = st.sidebar.selectbox(
    "Image",
    range(len(metadata)),
    format_func=lambda i: image_labels[i],
)
uploaded = st.sidebar.file_uploader("Upload your own", type=["jpg", "jpeg", "png"])

st.sidebar.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

strategy = st.sidebar.radio(
    "Strategy",
    ["Rerank", "Content", "Mood", "Balanced", "Creative"],
    index=0,
)

# ---- resolve image path ----
if uploaded is not None:
    tmp_path = os.path.join(tempfile.gettempdir(), f"cs131_upload_{uploaded.name}")
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getvalue())
    image_path = tmp_path
else:
    image_path = os.path.join(TEST_IMAGES_DIR, metadata[selected_idx]["filename"])

# ---- main content ----
st.markdown('<div style="height: 240px;"></div>', unsafe_allow_html=True)

col_img, col_poem = st.columns([1, 1], gap="large")

with col_img:
    st.image(image_path, use_container_width=True)

with col_poem:
    if uploaded is not None:
        with st.spinner("Finding your poem..."):
            results = run_retrieval_custom(image_path, strategy.lower())
        poem = results[0]
    else:
        filename = metadata[selected_idx]["filename"]
        poem = precomputed[filename][strategy.lower()]
    st.markdown(
        f'<div style="padding-left: 2.5rem; padding-top: 1rem;">'

        f'<div style="'
        f'font-family: Georgia, serif; font-size: 1.9em; font-weight: 700; '
        f'line-height: 1.2; margin: 0 0 8px 0; color: #111;">'
        f'{html.escape(poem["title"])}</div>'

        f'<div style="'
        f'font-family: system-ui, -apple-system, sans-serif; '
        f'font-size: 0.7em; letter-spacing: 0.15em; color: #999; '
        f'text-transform: uppercase; margin-bottom: 20px;">'
        f'By {html.escape(poem["author"])}</div>'

        f'<hr style="border: none; border-top: 1px solid #ddd; margin: 0 0 28px 0;">'

        f'<div style="font-family: Georgia, serif; font-size: 1.1em; color: #222;">'
        f'{poem_to_html(poem["text"])}</div>'

        f'</div>',
        unsafe_allow_html=True,
    )
