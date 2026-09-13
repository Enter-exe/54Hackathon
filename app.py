"""Single-screen Streamlit demo for the truck appraisal pipeline."""

import tempfile
from pathlib import Path

import streamlit as st

from pipeline.appraise import AppraisalEngine, artifacts_ready


SAFE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
LABELS = {
    "rust": "Rust",
    "body_damage": "Body damage",
    "tire_wear": "Tire wear",
    "interior_wear": "Interior wear",
    "too_dark": "Too dark",
    "too_blurry": "Too blurry",
    "not_truck": "Not a truck",
    "unreadable": "Unreadable file",
    "no_usable_truck_photos": "No usable truck detected",
}
WARNINGS = {
    "limited_photo_coverage": (
        "Limited photo coverage. Add at least three clear angles for a tighter range."
    ),
    "some_photos_rejected": (
        "Some photos were excluded. The estimate uses only clear truck photos."
    ),
}


st.set_page_config(
    page_title="Kamion Truck Appraisal",
    page_icon="K",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --kamion-navy: #10283e;
        --kamion-ink: #173247;
        --kamion-muted: #536775;
        --kamion-green: #1f704d;
        --kamion-green-dark: #15553a;
        --kamion-line: #d5dedb;
        --kamion-surface: #ffffff;
        --kamion-canvas: #f4f7f5;
    }

    .stApp {
        background: var(--kamion-canvas);
        color: var(--kamion-ink);
    }

    .block-container {
        max-width: 1120px;
        padding-top: 3rem;
        padding-bottom: 5rem;
    }

    h1, h2, h3, h4, p, label {
        color: var(--kamion-navy);
    }

    h1 {
        max-width: 16ch;
        font-size: clamp(2.25rem, 5vw, 4rem) !important;
        letter-spacing: -0.04em !important;
        line-height: 0.98 !important;
        margin-bottom: 0.75rem !important;
    }

    h2 {
        letter-spacing: -0.025em !important;
        margin-top: 2.75rem !important;
    }

    [data-testid="stCaptionContainer"] p {
        color: var(--kamion-muted);
        font-size: 0.95rem;
        line-height: 1.55;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--kamion-surface);
        border-color: var(--kamion-line) !important;
        border-radius: 0.35rem !important;
        box-shadow: 0 12px 30px rgba(16, 40, 62, 0.06);
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #f8faf9;
        border-color: #a9b8b3;
        border-radius: 0.25rem;
    }

    [data-testid="stFileUploader"] label p {
        font-weight: 650;
    }

    .stButton > button[kind="primary"] {
        min-height: 3rem;
        background: var(--kamion-green);
        border: 1px solid var(--kamion-green);
        border-radius: 0.25rem;
        color: #ffffff;
        font-weight: 700;
    }

    .stButton > button[kind="primary"] p {
        color: #ffffff;
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--kamion-green-dark);
        border-color: var(--kamion-green-dark);
    }

    button:focus-visible, a:focus-visible, input:focus-visible {
        outline: 3px solid #2b7fa8 !important;
        outline-offset: 3px !important;
    }

    [data-testid="stMetric"] {
        min-height: 8rem;
        padding: 1rem 1.125rem;
        background: var(--kamion-surface);
        border-top: 3px solid var(--kamion-navy);
    }

    [data-testid="stMetricLabel"] p {
        color: var(--kamion-muted);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.055em;
        text-transform: uppercase;
    }

    [data-testid="stMetricValue"] {
        color: var(--kamion-navy);
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.035em;
    }

    [data-testid="stAlert"] {
        border-radius: 0.25rem;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background: #ffffff;
        border-color: #82958f;
        color: var(--kamion-navy);
    }

    [data-testid="stFileUploaderDropzone"] button p {
        color: var(--kamion-navy);
    }

    [data-testid="stImage"] img {
        aspect-ratio: 16 / 10;
        object-fit: cover;
        border-radius: 0.2rem;
    }

    [data-testid="stLinkButton"] a {
        min-height: 2.75rem;
        border-radius: 0.25rem;
        border-color: #9baca7;
        color: var(--kamion-navy);
    }

    ::selection {
        color: #ffffff;
        background: var(--kamion-green);
    }

    @media (max-width: 640px) {
        .block-container {
            padding-top: 1.5rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        h1 {
            font-size: 2.45rem !important;
        }

        [data-testid="stMetric"] {
            min-height: 6.5rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_engine():
    """Load and retain one engine only after the appraisal action."""
    return AppraisalEngine.from_artifacts(Path.cwd())


def save_uploads(uploads, directory: Path) -> list[str]:
    """Save uploads with numeric filenames and approved image extensions."""
    paths = []
    for index, upload in enumerate(uploads, start=1):
        suffix = Path(upload.name).suffix.lower()
        if suffix not in SAFE_EXTENSIONS:
            suffix = ".jpg"
        path = Path(directory) / f"{index}{suffix}"
        path.write_bytes(upload.getbuffer())
        paths.append(str(path))
    return paths


def render_result(result: dict, photo_names: list[str] | None = None) -> None:
    """Render one complete accepted or rejected appraisal."""
    photo_names = photo_names or []
    if not result["accepted"]:
        st.error(
            "No usable truck photos were found. Add clear exterior truck photos "
            "in good light and try again."
        )
        rejected = result.get("rejected", [])
        if rejected:
            for photo in rejected:
                path = Path(photo["path"])
                try:
                    upload_index = int(path.stem) - 1
                except ValueError:
                    upload_index = -1
                name = (
                    photo_names[upload_index]
                    if 0 <= upload_index < len(photo_names)
                    else path.name
                )
                reasons = ", ".join(
                    LABELS.get(reason, reason.replace("_", " ").title())
                    for reason in photo.get("reasons", [])
                )
                st.text(f"{name} — {reasons}")
        else:
            reason = LABELS.get(
                result.get("reasons", ["no_usable_truck_photos"])[0],
                "No usable truck detected",
            )
            for name in photo_names:
                st.text(f"{name} — {reason}")
        return

    confidence = result["confidence"].title()
    st.success(f"{confidence} confidence")
    for warning in result.get("warnings", []):
        st.warning(WARNINGS.get(warning, warning.replace("_", " ").capitalize()))

    st.subheader("Estimated market range")
    price_columns = st.columns(3)
    for column, label, key in zip(
        price_columns,
        ("Low", "Market estimate", "High"),
        ("price_low", "price_median", "price_high"),
    ):
        column.metric(label, f"${result[key]:,.0f}")

    st.subheader("Condition signals")
    with st.container(border=True):
        for name, assessment in result.get("condition", {}).items():
            label, score, status = st.columns([2, 1, 1])
            label.markdown(f"**{LABELS.get(name, name.replace('_', ' ').title())}**")
            score.markdown(f"{assessment['probability']:.0%} signal")
            status.markdown("**Review**" if assessment.get("flag") else "No issue flagged")

    st.subheader("Comparable sales")
    comparable_columns = st.columns(3)
    for column, comparable in zip(comparable_columns, result.get("comparables", [])[:3]):
        with column:
            with st.container(border=True):
                st.image(
                    comparable["thumbnail_path"],
                    caption="Comparable truck",
                    width="stretch",
                )
                st.text(
                    f"{comparable['year']} {comparable['make_name']} "
                    f"{comparable['model_name']}"
                )
                st.metric("Sale price", f"${comparable['price']:,.0f}")
                st.caption(f"{comparable['similarity']:.0%} visual similarity")
                st.link_button(
                    "View source",
                    comparable["source_url"],
                    width="stretch",
                )


st.title("Kamion Truck Appraisal")
st.caption(
    "Research prototype: USD estimates from 100 completed US auction sales—not "
    "a formal valuation or offer, and not calibrated for European or Turkish markets."
)

with st.container(border=True):
    st.markdown("### Upload truck photos")
    uploads = st.file_uploader(
        "Truck photos",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Use clear exterior, tire, and cab angles in good light.",
    )
    appraise_clicked = st.button(
        "Appraise truck", type="primary", width="stretch"
    )

if appraise_clicked:
    if not artifacts_ready(Path.cwd()):
        st.error("Model artifacts are missing. Run: python -m pipeline.build_artifacts")
    elif not uploads:
        st.warning("Upload at least one truck photo.")
    else:
        with tempfile.TemporaryDirectory() as temporary_directory:
            saved_paths = save_uploads(uploads, Path(temporary_directory))
            with st.spinner("Analyzing truck photos…"):
                appraisal = get_engine().appraise(saved_paths)
            render_result(appraisal, [upload.name for upload in uploads])
