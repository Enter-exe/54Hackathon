"""Demo UI: upload photos of a used box truck, get a price range, predicted
specs and condition assessment, and comparable listings as reasoning.

Run with: streamlit run ui/app.py

Requires the pipeline artifacts to already exist (run in order):
  python pipeline/clean_split.py
  python -m pipeline.extract_embeddings
  python -m pipeline.condition_assessment
  python -m modeling.train_price
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from pipeline.condition_assessment import ATTRIBUTES
from ui.appraisal import PRICE_MODEL_PATH, load_resources, run_appraisal

REJECTION_MESSAGES = {
    "too_dark": "This photo is too dark — try retaking it with better lighting.",
    "too_blurry": "This photo is too blurry — hold the camera steady and try again.",
    "unreadable": "This file couldn't be read as an image.",
    "not_truck": "This doesn't look like a truck.",
}

st.set_page_config(page_title="What's This Truck Worth?", page_icon="🚚", layout="centered")

cached_load_resources = st.cache_resource(load_resources)


def save_uploads(uploaded_files, tmp_dir: Path) -> list[str]:
    paths = []
    for i, f in enumerate(uploaded_files):
        p = tmp_dir / f"{i}_{f.name}"
        p.write_bytes(f.getbuffer())
        paths.append(str(p))
    return paths


def render_rejection(gate_result: dict) -> None:
    st.error("Can't assess this truck from these photos.")
    for item in gate_result["rejected"]:
        name = Path(item["path"]).name
        messages = [REJECTION_MESSAGES.get(r, r) for r in item["reasons"]]
        st.write(f"**{name}** — " + " ".join(messages))
    st.info("Add clearer, well-lit photos that show the whole truck, then try again.")


def render_price(price_result: dict) -> None:
    st.subheader("Estimated price")
    conf = price_result["confidence"]
    badge = {"high": "🟢 High confidence", "medium": "🟡 Medium confidence", "low": "🟠 Low confidence"}[conf]
    st.markdown(f"### ${price_result['price_low']:,.0f} – ${price_result['price_high']:,.0f}")
    st.caption(f"Median estimate: ${price_result['price_median']:,.0f} · {badge}")
    st.caption(
        "This is an 80% confidence range: on similar past listings, the real price fell inside "
        "a range like this about 80% of the time."
    )
    if price_result["warnings"]:
        for w in price_result["warnings"]:
            st.warning(w.replace("_", " ").capitalize())


def render_condition(condition_result: dict) -> None:
    st.subheader("Condition assessment")
    cols = st.columns(len(ATTRIBUTES))
    for col, attr in zip(cols, ATTRIBUTES):
        r = condition_result[attr["name"]]
        label = attr["name"].replace("_", " ").title()
        with col:
            if r["flag"]:
                st.error(f"⚠️ {label}")
            else:
                st.success(f"✓ {label}")
            st.caption(f"score {r['probability']:.2f}")
    st.caption("Flags mean this truck looks worse than about 80% of comparable listings on that attribute — not a guarantee of damage.")


def render_predicted_specs(comparables_result: list[dict] | None) -> None:
    st.subheader("Predicted specs")
    if not comparables_result:
        st.caption("No comparables index available — can't estimate specs.")
        return
    top = comparables_result[0]
    st.markdown(f"**{top['year']} {top['make_name']} {top['model_name']}**")
    st.caption(
        f"Closest visual match in the training data (similarity {top['similarity']:.2f}). "
        "This is a nearest-neighbor lookup, not a trained make/model/year classifier — "
        "use it to sanity-check that the visual match is at least in the right ballpark."
    )


def render_comparables(comparables_result: list[dict] | None) -> None:
    st.subheader("Why this price")
    if not comparables_result:
        st.info("🚧 Comparable listings aren't available (no comparables index found).")
        return
    st.caption("The most visually similar real listings behind this estimate:")
    cols = st.columns(len(comparables_result))
    for col, comp in zip(cols, comparables_result):
        with col:
            thumb_dir = Path(f"data/images/{comp['ad_id']}")
            thumb = next(thumb_dir.glob("*.webp"), None) if thumb_dir.exists() else None
            if thumb:
                st.image(str(thumb), use_container_width=True)
            st.write(f"{comp['year']} {comp['make_name']} {comp['model_name']}")
            st.write(f"${comp['price']:,.0f}")
            st.caption(f"similarity {comp['similarity']:.2f}")


def main():
    st.title("🚚 What's This Truck Worth?")
    st.caption("Upload photos of a used box truck for a price range and condition assessment — photos only, no typed details needed.")

    resources = cached_load_resources()
    if resources is None:
        st.error(
            f"Price model not found at `{PRICE_MODEL_PATH}`. Run the pipeline first:\n\n"
            "```\npython pipeline/clean_split.py\n"
            "python -m pipeline.extract_embeddings\n"
            "python -m pipeline.condition_assessment\n"
            "python -m modeling.train_price\n```"
        )
        return

    uploaded_files = st.file_uploader(
        "Upload truck photos", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True
    )
    if not uploaded_files:
        st.caption("Bad lighting, awkward angles, a missing view — that's fine, upload what you have.")
        return

    with st.spinner("Analyzing photos..."):
        tmp_dir = Path(tempfile.mkdtemp())
        paths = save_uploads(uploaded_files, tmp_dir)
        result = run_appraisal(paths, resources)

    st.image(list(uploaded_files), width=150)

    if not result["gate"]["accepted"]:
        render_rejection(result["gate"])
        return

    render_price(result["price"])
    render_predicted_specs(result["comparables"])
    render_condition(result["condition"])
    render_comparables(result["comparables"])


if __name__ == "__main__":
    main()
