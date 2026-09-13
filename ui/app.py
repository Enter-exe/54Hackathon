"""Demo UI: provide photos of a used commercial truck, get a price range, predicted
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
from pipeline.listing_images import (
    ListingExtractionError,
    extract_listing_images,
    render_html_with_playwright,
)
from ui.appraisal import PRICE_MODEL_PATH, load_resources, run_appraisal

REJECTION_MESSAGES = {
    "too_dark": "This photo is too dark — try retaking it with better lighting.",
    "too_blurry": "This photo is too blurry — hold the camera steady and try again.",
    "unreadable": "This file couldn't be read as an image.",
    "not_truck": "This doesn't look like a truck.",
}

LISTING_ERROR_MESSAGES = {
    "unsafe_url": "Enter a public HTTP or HTTPS listing URL.",
    "unavailable": "We couldn't reach that listing. Check the URL and try again.",
    "blocked": "That listing site blocked automatic photo extraction.",
    "redirect": "That listing redirected too many times.",
    "unsupported": "That URL did not return a supported listing page.",
    "too_large": "That listing is too large to process automatically.",
    "no_images": "We couldn't find usable truck photos on that listing.",
    "browser_unavailable": "Automatic browser extraction is unavailable.",
    "browser_failed": "We couldn't render that listing automatically.",
}
GENERIC_LISTING_ERROR_MESSAGE = (
    "We couldn't extract photos from that listing automatically."
)

st.set_page_config(page_title="What's This Truck Worth?", page_icon="🚚", layout="centered")

cached_load_resources = st.cache_resource(load_resources)


def appraise_listing_url(
    url,
    output_dir,
    resources,
    *,
    extractor=extract_listing_images,
    appraiser=run_appraisal,
    browser_renderer=render_html_with_playwright,
):
    extraction = extractor(url, output_dir, browser_renderer=browser_renderer)
    return extraction, appraiser(extraction["image_paths"], resources)


def listing_error_message(error: ListingExtractionError) -> str:
    return LISTING_ERROR_MESSAGES.get(error.code, GENERIC_LISTING_ERROR_MESSAGE)


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


def render_predicted_specs(predicted_class_result: list[dict] | None, predicted_make_result: list[dict] | None) -> None:
    st.subheader("Predicted specs")

    if predicted_class_result:
        top, runner_up = predicted_class_result[0], predicted_class_result[1] if len(predicted_class_result) > 1 else None
        label = top["class_name"].split("(")[0].strip().title()
        st.markdown(f"**Weight class: {label}** ({top['probability']:.0%} confidence)")
        if runner_up and runner_up["probability"] >= 0.25:
            runner_label = runner_up["class_name"].split("(")[0].strip().title()
            st.caption(f"Close call — could also be {runner_label} ({runner_up['probability']:.0%}).")
    else:
        st.caption("No trained weight-class classifier available.")

    if predicted_make_result:
        top, runner_up = predicted_make_result[0], predicted_make_result[1] if len(predicted_make_result) > 1 else None
        st.markdown(f"**Make: {top['make_name'].title()}** ({top['probability']:.0%} confidence)")
        if runner_up and runner_up["probability"] >= 0.25:
            st.caption(f"Close call — could also be {runner_up['make_name'].title()} ({runner_up['probability']:.0%}).")
    else:
        st.caption("No trained make classifier available.")

    st.caption(
        "Both are trained classifiers (logistic regression on the same image embeddings), not a lookup. "
        "Weight class tends to be more reliable — it tracks visible vehicle size, whereas a make's lineup can span "
        "very different-looking body styles (e.g. a van-based cutaway vs. a heavy conventional-cab truck), which "
        "make prediction can get confidently wrong when a brand is dominated in the training data by one shape."
    )
    st.caption(
        "Model/year aren't predicted here — there isn't enough training data per model to classify those "
        "reliably yet. The closest real match's model/year shows up in \"Why this price\" below as reference, not a claim."
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
    st.caption(
        "Provide photos of a used commercial truck for a price range and condition "
        "assessment — photos only, no typed details needed."
    )

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

    source_mode = st.radio(
        "How would you like to provide the truck photos?",
        ["Listing link", "Upload photos"],
        horizontal=True,
    )

    if source_mode == "Listing link":
        with st.form("listing-link-form"):
            listing_url = st.text_input(
                "Truck listing URL",
                placeholder="https://www.purplewave.com/auction/260917/item/FK3297",
            )
            submitted = st.form_submit_button(
                "Extract photos and appraise", type="primary"
            )
        if not submitted:
            st.caption(
                "The estimate uses listing photos only—not the asking price or current bid."
            )
            return
        try:
            with tempfile.TemporaryDirectory() as temporary:
                with st.spinner("Extracting and analyzing listing photos…"):
                    extraction, result = appraise_listing_url(
                        listing_url, Path(temporary), resources
                    )
                st.success(
                    f"Extracted {len(extraction['image_paths'])} photos from "
                    f"{extraction['source_host']}."
                )
                for warning in extraction["warnings"]:
                    st.warning(warning)
                st.image(
                    extraction["image_paths"],
                    caption=[
                        f"Listing photo {index}"
                        for index in range(1, len(extraction["image_paths"]) + 1)
                    ],
                    width=150,
                )
        except ListingExtractionError as exc:
            st.error(listing_error_message(exc))
            st.info("Switch to Upload photos to continue manually.")
            return
    else:
        uploaded_files = st.file_uploader(
            "Upload truck photos",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
        )
        if not uploaded_files:
            st.caption(
                "Bad lighting, awkward angles, a missing view—that's fine, upload what you have."
            )
            return
        with tempfile.TemporaryDirectory() as temporary:
            with st.spinner("Analyzing photos…"):
                paths = save_uploads(uploaded_files, Path(temporary))
                result = run_appraisal(paths, resources)
            st.image(list(uploaded_files), width=150)

    if not result["gate"]["accepted"]:
        render_rejection(result["gate"])
        return

    render_price(result["price"])
    render_predicted_specs(result["predicted_class"], result["predicted_make"])
    render_condition(result["condition"])
    render_comparables(result["comparables"])


if __name__ == "__main__":
    main()
