
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from PIL import Image, ImageEnhance
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torchvision import transforms
from transformers import AutoModel, CLIPModel, CLIPProcessor
from scipy.ndimage import gaussian_filter


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PathoTrust Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CONFIGURATION
# =========================================================

import os

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "phikon_probe.pt"
)
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

DATASET_NAME = "qwertyuiop456556/tcga-ut"
DATASET_CONFIG = "internal"

TEST_IMAGES_PER_CLASS = 40
TEST_SEED = 42

MIN_CONFIDENCE = 0.50
DEFAULT_PREVIEW_WIDTH = 480

ATTENTION_IMAGE_SIZE = 224

# HeatmapTrust settings
HEATMAP_GRID_SIZE = 14
HEATMAP_BATCH_SIZE = 16
HEATMAP_SMOOTHING = 1.2
HEATMAP_OVERLAY_ALPHA = 0.55

STAIN_VARIATIONS = {
    "Original": (1.0, 1.0, 1.0),
    "Slight brightness change": (1.15, 1.0, 1.0),
    "Reduced brightness": (0.85, 1.0, 1.0),
    "Increased contrast": (1.0, 1.2, 1.0),
    "Reduced contrast": (1.0, 0.8, 1.0),
    "Increased colour saturation": (1.0, 1.0, 1.25),
    "Reduced colour saturation": (1.0, 1.0, 0.75),
}

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

NAV_PAGES = [
    "🏠 Home",
    "🧬 TumorSpot",
    "💧 StainShift",
    "✨ HeatmapTrust",
    "🛡️ HonestPath",
    "ℹ️ About",
]


# =========================================================
# GENERAL HELPERS
# =========================================================

def html(markup):
    cleaned = "\n".join(
        line.strip()
        for line in markup.splitlines()
        if line.strip()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


def make_inputs(images):
    if not isinstance(images, list):
        images = [images]

    return {
        "pixel_values": torch.stack([
            PREPROCESS(img.convert("RGB"))
            for img in images
        ])
    }


def display_image_preview(image, caption):
    size = st.session_state.get(
        "preview_width",
        DEFAULT_PREVIEW_WIDTH,
    )

    _, centre, _ = st.columns([1, 3, 1])

    with centre:
        st.image(
            image,
            caption=caption,
            width=size,
        )


def checkpoint_ready():
    if os.path.exists(CHECKPOINT_PATH):
        return True

    st.error(
        f"Model file not found: {CHECKPOINT_PATH}. "
        "Run the Phikon probe training cell first."
    )
    return False


# =========================================================
# SHARED IMAGE STATE
# =========================================================

def get_active():
    return st.session_state.get("active_image")


def set_active(image, name, source, true_label=None):
    st.session_state["active_image"] = {
        "image": image,
        "name": name,
        "source": source,
        "true_label": true_label,
    }

    for key in [
        "last_result",
        "stain_results",
        "heatmap_results",
    ]:
        st.session_state.pop(key, None)


def clear_active():
    st.session_state["active_image"] = None
    st.session_state["upload_sig"] = None

    st.session_state["upload_counter"] = (
        st.session_state.get("upload_counter", 0) + 1
    )

    for key in [
        "last_result",
        "stain_results",
        "heatmap_results",
    ]:
        st.session_state.pop(key, None)


def go_home():
    st.session_state["nav"] = "🏠 Home"


def caption_for(active):
    if active["source"] == "sample":
        return "TCGA-UT test image"

    return f"Selected image: {active['name']}"


def require_active_image():
    active = get_active()

    if active is None:
        st.info(
            "No image selected yet. Choose or upload one on the Home "
            "page and it will be used across the modules."
        )

        st.button(
            "🏠 Go to Home",
            on_click=go_home,
            key="goto_home",
        )

        return None

    display_image_preview(
        active["image"],
        caption_for(active),
    )

    return active


# =========================================================
# CLIP SCREENING MODEL & BALANCED SCREENING LOGIC
# =========================================================

@st.cache_resource
def load_screening_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
    model.to(device)
    model.eval()
    return model, processor, device


TUMOR_HISTOLOGY_PROMPTS = [
    "An H&E stained cancerous tumor histopathology tissue image.",
    "A microscopic tissue section showing malignant tumor carcinoma.",
    "A digital pathology slide showing neoplastic cancer cells.",
    "A histological image of malignant human cancer biopsy.",
]

NORMAL_HISTOLOGY_PROMPTS = [
    "Normal non-cancerous healthy human tissue section.",
    "Healthy histological tissue with normal cellular architecture.",
    "Benign tissue slide with no malignant cancer present.",
    "Normal non-tumorous histology microscopy slide.",
]

NON_HISTOLOGY_PROMPTS = [
    "A photograph of a person.",
    "A photograph of an animal.",
    "A landscape photograph.",
    "A photograph of a building.",
    "A photograph of an everyday object.",
    "A screenshot of a computer screen.",
    "A scanned page of a document or book.",
    "A photograph of food.",
    "A cartoon or digital illustration.",
]


def screen_input_image(image):
    model, processor, device = load_screening_model()
    image_rgb = image.convert("RGB")

    all_prompts = (
        TUMOR_HISTOLOGY_PROMPTS +
        NORMAL_HISTOLOGY_PROMPTS +
        NON_HISTOLOGY_PROMPTS
    )

    inputs = processor(
        text=all_prompts, images=image_rgb, return_tensors="pt", padding=True
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        vision_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"], return_dict=True
        )
        image_features = model.visual_projection(vision_outputs.pooler_output)

        text_outputs = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )
        text_features = model.text_projection(text_outputs.pooler_output)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        similarities = (image_features @ text_features.T)[0]

    n_tumor = len(TUMOR_HISTOLOGY_PROMPTS)
    n_normal = len(NORMAL_HISTOLOGY_PROMPTS)

    tumor_scores = similarities[:n_tumor]
    normal_scores = similarities[n_tumor : n_tumor + n_normal]
    non_hist_scores = similarities[n_tumor + n_normal :]

    top_tumor = float(torch.topk(tumor_scores, k=min(2, n_tumor)).values.mean().item())
    top_normal = float(torch.topk(normal_scores, k=min(2, n_normal)).values.mean().item())
    top_non_hist = float(torch.topk(non_hist_scores, k=min(3, len(NON_HISTOLOGY_PROMPTS))).values.mean().item())

    # STAGE 1: Pathology tissue vs Non-pathology graphics (cars, photos, documents)
    histology_max = max(top_tumor, top_normal)
    if (histology_max - top_non_hist) < 0.008:
        return {
            "status": "unrelated",
            "tumor_score": top_tumor,
            "normal_score": top_normal,
            "non_hist_score": top_non_hist,
        }

    # STAGE 2: Only flag as 'normal_tissue' if CLIP is strongly confident it is normal
    normal_margin = top_normal - top_tumor

    if normal_margin > 0.020:
        status = "normal_tissue"
    else:
        status = "tumor_like"

    return {
        "status": status,
        "tumor_score": top_tumor,
        "normal_score": top_normal,
        "non_hist_score": top_non_hist,
        "margin": top_tumor - top_normal,
    }


def display_screening_result(screening):
    status = screening["status"]

    if status == "unrelated":
        st.error("🚫 Non-histopathology image detected.")
        st.write(
            "This image appears consistent with general photography, documents, or non-medical images."
        )
        st.info("Analysis halted. Please upload or select a valid histopathology slide.")
        return "reject"

    elif status == "normal_tissue":
        st.info("🟢 Normal / Non-Tumorous Tissue Detected")
        st.write(
            "The screening model indicates this sample is likely **healthy or normal tissue** rather than a malignant tumor."
        )
        st.caption(
            "Phikon is trained strictly on 31 malignant cancer categories (TCGA). Running cancer classification on healthy tissue may yield misleading category scores."
        )
        return "normal"

    else:
        return "allow"


# =========================================================
# PHIKON MODEL
# =========================================================

@st.cache_resource
def load_phikon():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    backbone = AutoModel.from_pretrained(
        checkpoint["model_id"]
    )

    backbone.to(device)
    backbone.eval()

    class_names = list(checkpoint["class_names"])

    mu = checkpoint["mu"].to(device)
    sd = checkpoint["sd"].to(device)

    probe = nn.Linear(
        mu.shape[0],
        len(class_names),
    )

    probe.load_state_dict(checkpoint["probe"])
    probe.to(device)
    probe.eval()

    return (
        backbone,
        probe,
        mu,
        sd,
        class_names,
        device,
    )


def extract_features(images):
    backbone, _, _, _, _, device = load_phikon()

    inputs = {
        key: value.to(device)
        for key, value in make_inputs(images).items()
    }

    with torch.inference_mode():
        outputs = backbone(**inputs)
        features = outputs.last_hidden_state[:, 0, :]

    return features


def predict_logits(images):
    _, probe, mu, sd, _, device = load_phikon()

    features = extract_features(images)

    with torch.inference_mode():
        z = (features - mu) / (sd + 1e-6)
        logits = probe(z)

    return logits


def analyse_image(image):
    backbone, probe, mu, sd, class_names, device = load_phikon()

    inputs = {
        key: value.to(device)
        for key, value in make_inputs(image).items()
    }

    with torch.inference_mode():
        features = backbone(
            **inputs
        ).last_hidden_state[:, 0, :]

        z = (features - mu) / (sd + 1e-6)

        probabilities = torch.softmax(
            probe(z),
            dim=1,
        )[0]

        values, ids = torch.topk(
            probabilities,
            k=min(3, len(class_names)),
        )

    topk = [
        (
            class_names[i].replace("_", " "),
            float(v),
        )
        for i, v in zip(ids.tolist(), values.tolist())
    ]

    status = (
        "low_conf"
        if topk[0][1] < MIN_CONFIDENCE
        else "ok"
    )

    return {
        "status": status,
        "topk": topk,
    }


@st.cache_resource(show_spinner=False)
def load_test_dataset():
    return load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="test",
    )


# =========================================================
# TUMORSPOT
# =========================================================

def render_analysis(result, true_label=None):
    status = result["status"]
    topk = result["topk"]

    best_name, best_conf = topk[0]

    if status == "low_conf":
        st.warning(
            "Low-confidence prediction. Review the top guesses below."
        )

    else:
        html(f"""
        <div class="result-card">
            <p style="color:#b7b8ff; font-size:14px;">
                TOP PREDICTION
            </p>
            <h2 style="color:#ffffff;">{best_name}</h2>
            <p style="color:#c1cfee;">Model score</p>
            <h2 style="color:#9d91ff;">
                {best_conf * 100:.1f}%
            </h2>
        </div>
        """)

    st.markdown("**Top 3 guesses**")

    for rank, (name, prob) in enumerate(topk, start=1):
        st.write(f"{rank}. {name} — {prob * 100:.1f}%")
        st.progress(min(int(prob * 100), 100))

    if true_label is not None:
        true_name = true_label.replace("_", " ")
        names = [name for name, _ in topk]

        if names[0] == true_name:
            st.success(
                f"True label: {true_name} — top prediction is correct."
            )
        elif true_name in names:
            st.info(
                f"True label: {true_name} — it was in the top 3."
            )
        else:
            st.error(
                f"True label: {true_name} — the model missed it."
            )

    st.caption(
        "The percentage is the model's output score, "
        "not a calibrated probability."
    )


def run_tumorspot(active):
    if not checkpoint_ready():
        return

    try:
        with st.spinner("Screening image type..."):
            screening = screen_input_image(active["image"])

        decision = display_screening_result(screening)

        if decision == "reject":
            return

        if decision == "normal":
            confirm_key = f"confirm_tumorspot_{active['name']}"
            confirmed = st.checkbox(
                "Proceed with Phikon cancer subtype classification anyway?",
                key=confirm_key
            )
            if not confirmed:
                st.info("Classification paused until confirmation.")
                return

        with st.spinner("Analysing image with Phikon..."):
            result = analyse_image(active["image"])

        st.session_state["last_result"] = {
            "status": result["status"],
            "topk": result["topk"],
            "name": active["name"],
        }

        render_analysis(
            result,
            active["true_label"],
        )

    except Exception as error:
        st.error(f"TumorSpot analysis failed: {error}")


def tumorspot_interface():
    st.title("🧬 TumorSpot")

    st.write(
        "Predict the cancer category of the current image using "
        "Phikon features and a trained linear classifier."
    )

    active = require_active_image()

    if active is None:
        return

    if st.button("🔬 ANALYSE IMAGE", key="tumorspot_run"):
        st.markdown("---")
        st.subheader("TumorSpot Analysis Results")
        run_tumorspot(active)


# =========================================================
# STAINSHIFT — IMAGE VARIATIONS AND BAR GRAPH
# =========================================================

def create_stain_variation(
    image,
    brightness=1.0,
    contrast=1.0,
    saturation=1.0,
):
    result = image.convert("RGB")

    if brightness != 1.0:
        result = ImageEnhance.Brightness(result).enhance(brightness)

    if contrast != 1.0:
        result = ImageEnhance.Contrast(result).enhance(contrast)

    if saturation != 1.0:
        result = ImageEnhance.Color(result).enhance(saturation)

    return result


def run_stainshift(image):
    names = []
    images = []

    for name, factors in STAIN_VARIATIONS.items():
        variation = create_stain_variation(
            image,
            brightness=factors[0],
            contrast=factors[1],
            saturation=factors[2],
        )

        names.append(name)
        images.append(variation)

    logits = predict_logits(images)

    probabilities = torch.softmax(
        logits,
        dim=1,
    ).cpu().numpy()

    class_names = load_phikon()[4]

    original_index = int(np.argmax(probabilities[0]))
    original_class = class_names[original_index]
    original_score = float(probabilities[0, original_index])

    results = []

    for name, probability in zip(names, probabilities):
        predicted_index = int(np.argmax(probability))
        predicted_class = class_names[predicted_index]

        confidence = float(probability[predicted_index])
        original_class_score = float(probability[original_index])

        results.append({
            "variation": name,
            "predicted_class": predicted_class.replace("_", " "),
            "confidence": confidence,
            "original_class_score": original_class_score,
            "same_class": predicted_index == original_index,
        })

    return {
        "original_class": original_class.replace("_", " "),
        "original_score": original_score,
        "results": results,
    }


def render_stainshift_results(results, original_image):
    st.markdown("---")
    st.subheader("StainShift Results")

    st.metric(
        "Original prediction",
        results["original_class"],
        f"{results['original_score'] * 100:.1f}% model score",
    )

    st.markdown("### Prediction score comparison")

    graph_results = results["results"]

    labels = [item["variation"] for item in graph_results]

    scores = [
        item["original_class_score"] * 100
        for item in graph_results
    ]

    colours = [
        "#8B83FF" if item["same_class"] else "#F28C45"
        for item in graph_results
    ]

    fig, ax = plt.subplots(figsize=(13, 5))

    bars = ax.bar(
        range(len(labels)),
        scores,
        color=colours,
        width=0.62,
    )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        labels,
        rotation=25,
        ha="right",
        fontsize=9,
    )

    ax.set_ylabel("Original predicted-class score (%)")
    ax.set_ylim(0, 105)

    ax.set_title(
        "Effect of Image Appearance on the Original Prediction",
        fontsize=12,
        pad=15,
    )

    ax.grid(axis="y", alpha=0.2)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{score:.1f}%",
            ha="center",
            fontsize=8,
        )

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        "The graph tracks the score of the original predicted category "
        "across all image variations. Bar colour indicates whether the "
        "top predicted category stayed the same."
    )

    changed = sum(
        not item["same_class"]
        for item in graph_results[1:]
    )

    total = len(graph_results) - 1

    left, right, third = st.columns(3)

    left.metric("Prediction changes", f"{changed} / {total}")

    right.metric(
        "Prediction consistency",
        f"{(total - changed) / total * 100:.1f}%",
    )

    third.metric(
        "Original model score",
        f"{results['original_score'] * 100:.1f}%",
    )

    st.markdown("---")
    st.subheader("Visual stain comparison")

    preview_items = [
        ("Original", original_image.convert("RGB")),
    ]

    for name, factors in list(STAIN_VARIATIONS.items())[1:]:
        preview_items.append((
            name,
            create_stain_variation(
                original_image,
                brightness=factors[0],
                contrast=factors[1],
                saturation=factors[2],
            ),
        ))

    columns = st.columns(4)

    for index, (name, variation) in enumerate(preview_items):
        with columns[index % 4]:
            st.image(
                variation,
                caption=name,
                use_container_width=True,
            )

    st.caption(
        "The displayed images use controlled brightness, contrast "
        "and saturation adjustments."
    )


def stainshift_interface():
    st.title("💧 StainShift")

    st.write(
        "Compare visible changes in tissue-image appearance and "
        "investigate how they affect TumorSpot predictions."
    )

    active = require_active_image()

    if active is None:
        return

    if st.button("💧 RUN STAINSHIFT", key="stainshift_run"):
        if not checkpoint_ready():
            return

        try:
            with st.spinner("Screening image type..."):
                screening = screen_input_image(active["image"])

            decision = display_screening_result(screening)

            if decision == "reject":
                return

            if decision == "normal":
                st.warning("Notice: Image is flagged as normal or healthy tissue. Running robustness tests for demonstration.")

            with st.spinner("Testing controlled image variations..."):
                results = run_stainshift(active["image"])

            st.session_state["stain_results"] = results

        except Exception as error:
            st.error(f"StainShift failed: {error}")
            return

    results = st.session_state.get("stain_results")

    if results is not None:
        render_stainshift_results(results, active["image"])
    else:
        st.info(
            "Run StainShift to compare the image variations "
            "and prediction scores."
        )


# =========================================================
# HEATMAPTRUST — PATCH OCCLUSION SENSITIVITY
# =========================================================

def get_prediction_scores(images, target_class):
    logits = predict_logits(images)
    scores = torch.softmax(
        logits,
        dim=1,
    )[:, target_class]

    return scores.detach().cpu().numpy()


def create_occlusion_images(image, grid_size):
    image = image.convert("RGB").resize(
        (ATTENTION_IMAGE_SIZE, ATTENTION_IMAGE_SIZE)
    )

    image_array = np.asarray(
        image,
        dtype=np.uint8,
    ).copy()

    height, width = image_array.shape[:2]

    patch_h = height / grid_size
    patch_w = width / grid_size

    replacement = np.mean(
        image_array.reshape(-1, 3),
        axis=0,
    ).astype(np.uint8)

    masked_images = []
    patch_positions = []

    for row in range(grid_size):
        for col in range(grid_size):
            y1 = int(round(row * patch_h))
            y2 = int(round((row + 1) * patch_h))

            x1 = int(round(col * patch_w))
            x2 = int(round((col + 1) * patch_w))

            masked = image_array.copy()
            masked[y1:y2, x1:x2] = replacement

            masked_images.append(
                Image.fromarray(masked)
            )

            patch_positions.append(
                (row, col)
            )

    return image, masked_images, patch_positions


def calculate_occlusion_heatmap(image):
    _, _, _, _, class_names, device = load_phikon()

    original, masked_images, patch_positions = create_occlusion_images(
        image,
        HEATMAP_GRID_SIZE,
    )

    original_logits = predict_logits(original)

    original_probabilities = torch.softmax(
        original_logits,
        dim=1,
    )[0]

    target_class = int(
        torch.argmax(original_probabilities).item()
    )

    baseline_score = float(
        original_probabilities[target_class].item()
    )

    target_name = class_names[target_class].replace("_", " ")

    masked_scores = []
    total = len(masked_images)

    progress = st.progress(0)
    status = st.empty()

    for start in range(0, total, HEATMAP_BATCH_SIZE):
        batch = masked_images[
            start:start + HEATMAP_BATCH_SIZE
        ]

        batch_scores = get_prediction_scores(
            batch,
            target_class,
        )

        masked_scores.extend(batch_scores.tolist())

        completed = min(
            start + len(batch),
            total,
        )

        progress.progress(completed / total)

        status.text(
            f"Testing image regions: {completed} / {total}"
        )

    progress.empty()
    status.empty()

    heatmap = np.zeros(
        (HEATMAP_GRID_SIZE, HEATMAP_GRID_SIZE),
        dtype=np.float32,
    )

    for position, masked_score in zip(
        patch_positions,
        masked_scores,
    ):
        row, col = position

        heatmap[row, col] = max(
            0.0,
            baseline_score - float(masked_score),
        )

    maximum = float(heatmap.max())

    if maximum > 1e-8:
        heatmap = heatmap / maximum

    heatmap = gaussian_filter(
        heatmap,
        sigma=HEATMAP_SMOOTHING,
    )

    if heatmap.max() > 1e-8:
        heatmap = heatmap / heatmap.max()

    heatmap = np.clip(
        heatmap,
        0,
        1,
    )

    heatmap_image = Image.fromarray(
        np.uint8(heatmap * 255)
    )

    heatmap_image = heatmap_image.resize(
        (ATTENTION_IMAGE_SIZE, ATTENTION_IMAGE_SIZE),
        Image.Resampling.BICUBIC,
    )

    heatmap = (
        np.asarray(
            heatmap_image,
            dtype=np.float32,
        ) / 255.0
    )

    heatmap = np.clip(
        heatmap,
        0,
        1,
    )

    return {
        "original_image": original,
        "heatmap": heatmap,
        "target_class": target_name,
        "baseline_score": baseline_score,
        "grid_size": HEATMAP_GRID_SIZE,
    }


def make_heatmap_overlay(original_image, heatmap):
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.imshow(original_image)

    colour_map = plt.get_cmap("turbo")

    overlay = ax.imshow(
        heatmap,
        cmap=colour_map,
        alpha=HEATMAP_OVERLAY_ALPHA,
        vmin=0,
        vmax=1,
        interpolation="bilinear",
    )

    ax.set_title(
        "HeatmapTrust — Model Sensitivity Overlay",
        fontsize=13,
        pad=12,
        color="white",
    )

    ax.axis("off")

    colourbar = fig.colorbar(
        overlay,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    colourbar.set_label(
        "Relative influence",
        fontsize=9,
    )

    fig.tight_layout()

    return fig


def make_heatmap_only(heatmap):
    fig, ax = plt.subplots(figsize=(7, 6))

    image_plot = ax.imshow(
        heatmap,
        cmap="turbo",
        vmin=0,
        vmax=1,
        interpolation="bilinear",
    )

    ax.set_title(
        "Regional sensitivity map",
        fontsize=12,
        pad=12,
    )

    ax.axis("off")

    fig.colorbar(
        image_plot,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="Relative influence",
    )

    fig.tight_layout()

    return fig


def render_heatmap_results(results):
    st.markdown("---")
    st.subheader("HeatmapTrust Results")

    st.metric(
        "Target prediction",
        results["target_class"],
        f"{results['baseline_score'] * 100:.1f}% model score",
    )

    st.write(
        f"Region analysis: "
        f"{results['grid_size']} × {results['grid_size']} patches."
    )

    original = results["original_image"]
    heatmap = results["heatmap"]

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            original,
            caption="Original tissue image",
            use_container_width=True,
        )

    with col2:
        fig = make_heatmap_overlay(
            original,
            heatmap,
        )

        st.pyplot(
            fig,
            use_container_width=True,
        )

        plt.close(fig)

    st.markdown("---")
    st.subheader("Heatmap distribution")

    _, centre, _ = st.columns([1, 2, 1])

    with centre:
        fig = make_heatmap_only(heatmap)

        st.pyplot(
            fig,
            use_container_width=True,
        )

        plt.close(fig)

    st.caption(
        "Warmer regions indicate areas where hiding the patch "
        "caused a larger reduction in the model's target-class score. "
        "This measures model sensitivity, not biological importance."
    )


def heatmaptrust_interface():
    st.title("✨ HeatmapTrust")

    st.write(
        "Investigate which tissue-image regions influence "
        "the model's predicted category."
    )

    active = require_active_image()

    if active is None:
        return

    if st.button(
        "✨ GENERATE HEATMAP",
        key="heatmap_run",
    ):
        if not checkpoint_ready():
            return

        try:
            with st.spinner("Screening image type..."):
                screening = screen_input_image(active["image"])

            decision = display_screening_result(screening)

            if decision == "reject":
                return

            if decision == "normal":
                st.warning("Notice: Generating occlusion sensitivity map on non-tumour or uncertain tissue.")

            with st.spinner(
                "Testing image regions and generating the heatmap..."
            ):
                results = calculate_occlusion_heatmap(
                    active["image"]
                )

            st.session_state["heatmap_results"] = results

        except Exception as error:
            st.error(f"HeatmapTrust failed: {error}")
            st.exception(error)
            return

    results = st.session_state.get("heatmap_results")

    if results is None:
        st.info(
            "Generate the heatmap to inspect which image regions "
            "influence the current prediction."
        )
        return

    render_heatmap_results(results)


# =========================================================
# HONESTPATH — EVALUATION
# =========================================================

def evaluate_honestpath():
    class_names = load_phikon()[4]
    dataset = load_test_dataset()

    labels = [
        item["label"]
        for item in dataset["json"]
    ]

    groups = {}

    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)

    rng = random.Random(TEST_SEED)
    chosen = []

    for label in sorted(groups):
        if label not in class_names:
            raise ValueError(
                f"Unknown tumour category: {label}"
            )

        rng.shuffle(groups[label])

        chosen.extend(
            groups[label][:TEST_IMAGES_PER_CLASS]
        )

    batch_size = 32

    total_batches = (
        len(chosen) + batch_size - 1
    ) // batch_size

    y_true = []
    y_pred = []
    in_top3 = []

    progress = st.progress(0)
    status = st.empty()

    for batch_number, start in enumerate(
        range(0, len(chosen), batch_size)
    ):
        rows = [
            dataset[i]
            for i in chosen[start:start + batch_size]
        ]

        images = [
            row["jpg"].convert("RGB")
            for row in rows
        ]

        batch_labels = [
            class_names.index(row["json"]["label"])
            for row in rows
        ]

        logits = predict_logits(images)

        top3 = torch.topk(
            logits,
            k=3,
            dim=1,
        ).indices.cpu().numpy()

        y_true.extend(batch_labels)
        y_pred.extend(top3[:, 0].tolist())

        in_top3.extend([
            label in row
            for label, row in zip(batch_labels, top3)
        ])

        progress.progress(
            (batch_number + 1) / total_batches
        )

        status.text(
            f"Evaluating batch {batch_number + 1} "
            f"of {total_batches}..."
        )

    progress.empty()
    status.empty()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    all_labels = list(range(len(class_names)))

    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "top3": float(np.mean(in_top3)),

        "macro_precision": precision_score(
            y_true,
            y_pred,
            labels=all_labels,
            average="macro",
            zero_division=0,
        ),

        "macro_recall": recall_score(
            y_true,
            y_pred,
            labels=all_labels,
            average="macro",
            zero_division=0,
        ),

        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=all_labels,
            average="macro",
            zero_division=0,
        ),

        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=all_labels,
        ),

        "class_names": class_names,
        "total_images": len(y_true),
    }


def honestpath_interface():
    st.title("🛡️ HonestPath")

    st.write(
        "Evaluate how consistently TumorSpot distinguishes "
        "the tumour categories."
    )

    st.info(
        "HonestPath evaluates the saved Phikon checkpoint on "
        "labelled test images. It does not retrain or modify the model."
    )

    if not checkpoint_ready():
        return

    class_names = load_phikon()[4]

    left, right = st.columns(2)

    with left:
        st.write("**Model:** Phikon + linear classification probe")
        st.write(f"**Tumour categories:** {len(class_names)}")

    with right:
        st.write(f"**Images per category:** {TEST_IMAGES_PER_CLASS}")
        st.write(f"**Selection seed:** {TEST_SEED}")

    if st.button(
        "▶ RUN HONESTPATH EVALUATION",
        key="honestpath_run",
    ):
        try:
            with st.spinner("Evaluating the test images..."):
                results = evaluate_honestpath()

            st.session_state["honestpath_results"] = results
            st.success("HonestPath evaluation completed.")

        except Exception as error:
            st.error(f"HonestPath evaluation failed: {error}")
            st.exception(error)

    results = st.session_state.get("honestpath_results")

    if results is None:
        st.info(
            "Run the evaluation to see metrics and the confusion matrix."
        )
        return

    st.markdown("---")
    st.subheader("📊 Overall Metrics")

    columns = st.columns(5)

    columns[0].metric(
        "Top-1 Accuracy",
        f"{results['accuracy'] * 100:.2f}%",
    )

    columns[1].metric(
        "Top-3 Accuracy",
        f"{results['top3'] * 100:.2f}%",
    )

    columns[2].metric(
        "Macro Precision",
        f"{results['macro_precision']:.4f}",
    )

    columns[3].metric(
        "Macro Recall",
        f"{results['macro_recall']:.4f}",
    )

    columns[4].metric(
        "Macro F1",
        f"{results['macro_f1']:.4f}",
    )

    st.caption(
        f"Evaluation sample: {results['total_images']:,} images "
        f"({TEST_IMAGES_PER_CLASS} per category)."
    )

    st.markdown("---")
    st.subheader("🔎 Confusion Matrix")

    st.caption(
        "Rows are actual categories; columns are predicted categories."
    )

    confusion = results["confusion_matrix"]
    names = results["class_names"]

    _, centre, _ = st.columns([1, 3, 1])

    with centre:
        fig, ax = plt.subplots(figsize=(9, 7))

        plot = ax.imshow(
            confusion,
            interpolation="nearest",
            aspect="auto",
            cmap="Blues",
        )

        ax.set_title(
            "Tumour Classification Confusion Matrix",
            fontsize=11,
            pad=12,
        )

        ax.set_xlabel("Predicted category", fontsize=9)
        ax.set_ylabel("Actual category", fontsize=9)

        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))

        ax.set_xticklabels(
            names,
            rotation=90,
            fontsize=6,
        )

        ax.set_yticklabels(
            names,
            fontsize=6,
        )

        fig.colorbar(
            plot,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )

        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    st.caption(
        "The diagonal shows correct classifications; values off "
        "the diagonal are misclassifications."
    )

    with st.expander("Evaluation limitations"):
        st.write(
            """
            - Results describe the selected test sample (40 images per class).
            - The Phikon backbone was pretrained on TCGA-derived material.
              Overlap with TCGA-UT test images has not been ruled out.
            - The model sees one small patch at a time, not a whole slide.
            - This prototype is not clinically validated.
            """
        )


# =========================================================
# PATHOBOT
# =========================================================

PATHOBOT_GREETING = (
    "Hi! I'm PathoBot. Ask me how to use the app, what a metric means, "
    "or about your latest result."
)


def _has(q, *words):
    return any(word in q for word in words)


def pathobot_reply(message):
    q = re.sub(
        r"[^a-z0-9%.\-\s]",
        " ",
        message.lower(),
    )

    q = re.sub(r"\s+", " ", q).strip()

    last = st.session_state.get("last_result")
    hp = st.session_state.get("honestpath_results")

    if not q:
        return "Please type a question about PathoTrust Engine."

    if re.fullmatch(
        r"(hi|hello|hey|good morning|good afternoon|good evening)"
        r"( pathobot)?",
        q,
    ):
        return PATHOBOT_GREETING

    if _has(
        q,
        "non-tumour",
        "non tumour",
        "normal tissue",
        "not tumour",
        "not a tumour",
        "non cancer",
        "non-cancer",
        "normal",
        "reject",
        "screening",
    ):
        return (
            "PathoTrust Engine uses a two-stage CLIP screening model before classification. "
            "1. Stage 1 rejects non-histopathology imagery (e.g. general photos, documents).\n"
            "2. Stage 2 flags normal/healthy tissue before Phikon multi-class classification. "
            "If normal tissue is detected, the app displays a clear warning and requests confirmation before assigning a TCGA cancer subtype score."
        )

    if _has(
        q,
        "my result",
        "my image",
        "my prediction",
        "last result",
        "what did it predict",
        "what did you predict",
        "predicted",
    ):
        if last is None:
            return (
                "There is no result yet. Pick an image on the Home "
                "page and run TumorSpot first."
            )

        lines = [
            f"{i}. {name} — {prob * 100:.1f}%"
            for i, (name, prob) in enumerate(last["topk"], 1)
        ]

        return (
            "Top guesses for the last image:\n\n"
            + "\n".join(lines)
            + "\n\nThese are model scores, not a diagnosis."
        )

    if _has(
        q,
        "how do i",
        "how to",
        "upload",
        "navigate",
        "use the app",
        "random",
        "sample",
        "change image",
        "another image",
    ):
        return (
            "1. On Home, upload an image or choose a random TCGA-UT "
            "test image.\n"
            "2. The selected image is shared across the modules.\n"
            "3. Open TumorSpot, StainShift or HeatmapTrust from "
            "the navigation menu.\n"
            "4. Use Clear image in the sidebar to start over."
        )

    if _has(
        q,
        "accuracy",
        "top-1",
        "top 1",
        "top-3",
        "top 3",
        "f1",
        "precision",
        "recall",
        "macro",
        "honestpath",
        "performance",
        "how well",
        "results",
    ):
        text = (
            "**Top-1 accuracy:** how often the highest-scoring class "
            "is correct. **Top-3:** how often the true class is among "
            "the three highest scores. **Precision:** of images "
            "predicted as a class, how many truly were. **Recall:** "
            "of true images of a class, how many were found. "
            "**Macro F1:** F1 averaged over classes with equal weight."
        )

        if hp is not None:
            text += (
                f"\n\nYour latest HonestPath run "
                f"({hp['total_images']:,} images): top-1 "
                f"{hp['accuracy'] * 100:.2f}%, top-3 "
                f"{hp['top3'] * 100:.2f}%, macro F1 "
                f"{hp['macro_f1']:.4f}."
            )
        else:
            text += (
                "\n\nRun HonestPath to get numbers for the current model."
            )

        return text

    if _has(
        q,
        "confusion",
        "confused",
        "misclassif",
        "mix up",
        "mistake",
    ):
        return (
            "The confusion matrix compares true classes (rows) with "
            "predicted classes (columns). The diagonal is correct "
            "predictions; cells off the diagonal show which cancer "
            "types get mixed up."
        )

    if _has(
        q,
        "stain",
        "colour",
        "color",
        "brightness",
        "robust",
    ):
        return (
            "StainShift displays the original image and six controlled "
            "brightness, contrast and colour variations. Its bar graph "
            "compares the original predicted category's score across "
            "the variations and indicates whether the top prediction "
            "changed."
        )

    if _has(
        q,
        "grad-cam",
        "gradcam",
        "heatmap",
        "heat map",
        "explain",
        "attention",
        "phikon attention",
    ):
        return (
            "HeatmapTrust uses patch-based occlusion sensitivity. "
            "It hides small image regions one at a time and measures "
            "how much the predicted-class score changes. The result "
            "is displayed as a dark-blue, yellow, orange and red "
            "overlay on the original tissue image."
        )

    if _has(
        q,
        "phikon",
        "probe",
        "backbone",
        "which model",
        "how does",
        "tumorspot",
        "model",
    ):
        return (
            "TumorSpot uses Phikon as a frozen feature extractor. "
            "A trained linear classifier on top predicts one of the "
            "tumour categories from TCGA-UT image patches."
        )

    if _has(
        q,
        "confidence",
        "calibrat",
        "certain",
        "probability",
        "score",
    ):
        return (
            "The percentage shown is the model's output score, not "
            "the true chance that the prediction is right. It has "
            "not been calibrated."
        )

    if _has(
        q,
        "diagnos",
        "patient",
        "clinical",
        "medical",
        "trust",
        "reliable",
        "hospital",
    ):
        return (
            "PathoTrust Engine is a research prototype and is not "
            "clinically validated. It should not be used to diagnose "
            "a patient or guide treatment."
        )

    if _has(
        q,
        "dataset",
        "tcga",
        "data",
        "classes",
        "cancer types",
    ):
        return (
            "The model uses H&E image patches from the TCGA-UT "
            "dataset, with tumour categories. Each prediction is "
            "for one small patch, not a whole slide."
        )

    if _has(
        q,
        "what is pathotrust",
        "modules",
        "what can you",
        "who are you",
        "help",
        "about",
    ):
        return (
            "PathoTrust Engine has four main modules: "
            "**TumorSpot** classifies image patches, "
            "**StainShift** tests sensitivity to image changes, "
            "**HeatmapTrust** investigates model sensitivity, "
            "and **HonestPath** evaluates classification performance."
        )

    return (
        "Try asking about TumorSpot, StainShift, HeatmapTrust, "
        "HonestPath, model scores or your latest result."
    )


def pathobot_panel():
    history = st.session_state.setdefault(
        "pathobot_history",
        [{
            "role": "assistant",
            "content": PATHOBOT_GREETING,
        }],
    )

    box = st.container(height=320)

    with st.form(
        "pathobot_form",
        clear_on_submit=True,
    ):
        question = st.text_input(
            "Ask PathoBot",
            placeholder="Ask about the app, metrics or your result...",
            label_visibility="collapsed",
        )

        sent = st.form_submit_button("Send")

    if sent and question.strip():
        history.append({
            "role": "user",
            "content": question.strip(),
        })

        history.append({
            "role": "assistant",
            "content": pathobot_reply(question),
        })

    with box:
        for item in history:
            st.chat_message(
                item["role"]
            ).write(item["content"])


def pathobot_floating():
    with st.container(key="pathobot_float"):
        with st.popover("💬 PathoBot"):
            pathobot_panel()


# =========================================================
# THEME AND CUSTOM CSS
# =========================================================

html("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #071329 0%,
        #0b1d3b 55%,
        #101638 100%
    );
    color: #f3f5ff;
}

header[data-testid="stHeader"] {
    background: #071329;
}

[data-testid="stAppViewContainer"] {
    background: #071329;
}

[data-testid="stSidebar"] {
    background: #06122a;
    border-right: 1px solid #253967;
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1500px;
}

h1, h2, h3, p, label {
    color: #f3f5ff;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #f3f5ff !important;
}

.hero {
    background: linear-gradient(
        110deg,
        #101d42 0%,
        #20245b 55%,
        #43266c 100%
    );
    border: 1px solid #514d9c;
    border-radius: 22px;
    padding: 32px;
    margin-bottom: 24px;
}

.hero-kicker {
    color: #b7b8ff;
    font-size: 13px;
    letter-spacing: 3px;
    font-weight: 700;
}

.hero h1 {
    font-size: 42px;
    line-height: 1.1;
    margin: 12px 0;
}

.hero p {
    color: #d0dcff;
    font-size: 17px;
}

.module-card {
    background: #101f3d;
    border: 1px solid #2b416e;
    border-radius: 16px;
    padding: 18px;
    min-height: 145px;
}

.module-card h3 {
    font-size: 18px;
    margin-bottom: 8px;
}

.module-card p {
    font-size: 13px;
    color: #c1cfee;
}

div[data-testid="stFileUploader"] {
    background: #101f3d;
    border: 1px dashed #7169ed;
    border-radius: 14px;
    padding: 12px;
}

div.stButton > button {
    background: linear-gradient(
        90deg,
        #6257ff,
        #bd4df0
    );
    color: white;
    border: none;
    border-radius: 12px;
    min-height: 48px;
    font-weight: 700;
    width: 100%;
}

div.stButton > button:hover {
    border: 1px solid #d6c5ff;
    color: white;
}

.result-card {
    background: #101f3d;
    border: 1px solid #514d9c;
    border-radius: 16px;
    padding: 24px;
    margin-top: 16px;
}

.st-key-pathobot_float {
    position: fixed;
    bottom: 24px;
    right: 28px;
    z-index: 1000;
    width: auto !important;
}

.st-key-pathobot_float button {
    border-radius: 999px;
    min-height: 54px;
    padding: 0 22px;
    background: linear-gradient(
        90deg,
        #6257ff,
        #bd4df0
    );
    color: white;
    border: none;
    font-weight: 700;
    box-shadow: 0 8px 24px rgba(98, 87, 255, 0.45);
}

div[data-testid="stPopoverBody"] {
    width: min(420px, 92vw);
    max-height: 75vh;
    background: #0c1b35;
    border: 1px solid #354c80;
    border-radius: 16px;
}

</style>
""")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 🔬 PathoTrust")
    st.markdown("### Engine")

    st.caption(
        "Better evaluation. More trustworthy pathology AI."
    )

    st.divider()

    page = st.selectbox(
        "NAVIGATION",
        NAV_PAGES,
        key="nav",
    )

    st.slider(
        "Image preview size",
        min_value=250,
        max_value=700,
        value=DEFAULT_PREVIEW_WIDTH,
        step=10,
        key="preview_width",
    )

    st.divider()

    st.markdown("**Current image**")

    active_sidebar = get_active()

    if active_sidebar is None:
        st.caption("No image selected yet.")

    else:
        st.image(
            active_sidebar["image"],
            width=180,
        )

        st.caption(
            "TCGA-UT test image"
            if active_sidebar["source"] == "sample"
            else active_sidebar["name"]
        )

        st.button(
            "✕ Clear image",
            on_click=clear_active,
            key="clear_image",
        )

    st.divider()

    st.caption("Research prototype only.")
    st.caption("Not for clinical diagnosis.")


pathobot_floating()


# =========================================================
# ABOUT PAGE
# =========================================================

def about_interface():
    st.title("About PathoTrust Engine")

    st.write(
        "A research prototype for evaluating pathology AI through "
        "classification, robustness checks, model explanations and "
        "honest performance reporting."
    )

    st.markdown("### Modules")

    modules = [
        (
            "🧬 TumorSpot",
            "Predicts a tumour category from a histopathology image patch.",
        ),
        (
            "💧 StainShift",
            "Displays controlled image variations and compares "
            "prediction scores using a bar graph.",
        ),
        (
            "✨ HeatmapTrust",
            "Uses patch-based occlusion sensitivity to visualise "
            "regions influencing the model's prediction.",
        ),
        (
            "🛡️ HonestPath",
            "Evaluates model performance on labelled test images.",
        ),
    ]

    for title, description in modules:
        st.markdown(f"**{title}**")
        st.write(description)

    st.caption(
        "This prototype is not clinically validated and must not "
        "be used to diagnose patients or guide treatment."
    )


# =========================================================
# HOME PAGE
# =========================================================

def home_interface():
    html("""
    <div class="hero">
        <div class="hero-kicker">WELCOME TO YOUR WORKSPACE</div>
        <h1>PathoTrust Engine</h1>
        <p>
            Detect the tumour. Challenge the evidence.
            Evaluate the model honestly.
        </p>
        <p>
            🔬 Classify &nbsp; · &nbsp; 💧 Test robustness
            &nbsp; · &nbsp; ✨ Explain &nbsp; · &nbsp; 🛡️ Evaluate
        </p>
    </div>
    """)

    st.markdown("## 📤 Upload Image & Run Analysis")

    st.write(
        "Choose an image once. It is used on every page, so you can "
        "switch modules without uploading again."
    )

    up_col, sample_col = st.columns(
        [2, 1],
        gap="large",
    )

    with up_col:
        uploaded_file = st.file_uploader(
            "Upload an H&E pathology image",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key=(
                f"home_upload_"
                f"{st.session_state.get('upload_counter', 0)}"
            ),
        )

        if uploaded_file is not None:
            signature = (
                uploaded_file.name,
                uploaded_file.size,
            )

            if signature != st.session_state.get("upload_sig"):
                try:
                    set_active(
                        Image.open(uploaded_file).convert("RGB"),
                        uploaded_file.name,
                        "upload",
                    )

                    st.session_state["upload_sig"] = signature

                except Exception:
                    st.error("Unable to read this image.")

    with sample_col:
        st.write("")
        st.write("")

        if st.button(
            "🎲 Use a random TCGA-UT test image",
            key="random_sample",
        ):
            try:
                with st.spinner("Loading test dataset..."):
                    dataset = load_test_dataset()
                    row = dataset[random.randrange(len(dataset))]

                set_active(
                    row["jpg"].convert("RGB"),
                    "TCGA-UT test image",
                    "sample",
                    row["json"]["label"],
                )

            except Exception as error:
                st.error(
                    f"Could not load a test image: {error}"
                )

        st.caption(
            "Loads a labelled patch from the held-out test set."
        )

    active = get_active()

    if active is not None:
        display_image_preview(
            active["image"],
            caption_for(active),
        )

        st.success(
            "Image ready. It will be used across the modules."
        )

    st.markdown("#### Select module(s)")

    c1, c2 = st.columns(2)

    with c1:
        run_tumor = st.checkbox(
            "🧬 TumorSpot — Classification",
            value=True,
        )

        run_stain = st.checkbox(
            "💧 StainShift — Robustness",
            value=False,
        )

    with c2:
        run_heatmap = st.checkbox(
            "✨ HeatmapTrust — Explanation",
            value=False,
        )

        run_honest = st.checkbox(
            "🛡️ HonestPath — Evaluation",
            value=False,
        )

    if st.button(
        "▶ RUN ANALYSIS",
        key="home_run_analysis",
    ):
        selected = [
            name
            for name, enabled in [
                ("TumorSpot", run_tumor),
                ("StainShift", run_stain),
                ("HeatmapTrust", run_heatmap),
                ("HonestPath", run_honest),
            ]
            if enabled
        ]

        if not selected:
            st.warning("Please select at least one module.")
            return

        if (
            "TumorSpot" in selected
            or "StainShift" in selected
            or "HeatmapTrust" in selected
        ) and active is None:
            st.warning(
                "Select an image before running image analysis."
            )
            return

        if not checkpoint_ready():
            return

        if "TumorSpot" in selected:
            st.markdown("---")
            st.subheader("🧬 TumorSpot Results")
            run_tumorspot(active)

        if "StainShift" in selected:
            st.markdown("---")
            st.subheader("💧 StainShift Results")

            try:
                with st.spinner("Screening image type..."):
                    screening = screen_input_image(active["image"])

                decision = display_screening_result(screening)

                if decision != "reject":
                    with st.spinner("Testing image variations..."):
                        results = run_stainshift(active["image"])

                    st.session_state["stain_results"] = results

                    render_stainshift_results(
                        results,
                        active["image"],
                    )

            except Exception as error:
                st.error(f"StainShift failed: {error}")

        if "HeatmapTrust" in selected:
            st.markdown("---")
            st.subheader("✨ HeatmapTrust Results")

            try:
                with st.spinner("Screening image type..."):
                    screening = screen_input_image(active["image"])

                decision = display_screening_result(screening)

                if decision != "reject":
                    with st.spinner(
                        "Testing image regions and generating the heatmap..."
                    ):
                        results = calculate_occlusion_heatmap(
                            active["image"]
                        )

                    st.session_state["heatmap_results"] = results
                    render_heatmap_results(results)

            except Exception as error:
                st.error(f"HeatmapTrust failed: {error}")
                st.exception(error)

        if "HonestPath" in selected:
            st.markdown("---")
            st.subheader("🛡️ HonestPath")

            try:
                with st.spinner(
                    "Running HonestPath evaluation..."
                ):
                    results = evaluate_honestpath()

                st.session_state["honestpath_results"] = results

                st.success(
                    "HonestPath evaluation completed. "
                    "Open HonestPath in the navigation menu "
                    "to inspect all metrics and the confusion matrix."
                )

            except Exception as error:
                st.error(f"HonestPath failed: {error}")

    st.caption(
        "PathoTrust Engine is a research prototype, not a clinical "
        "diagnostic tool. Predictions must not guide patient care."
    )


# =========================================================
# PAGE DISPATCH
# =========================================================

if page == "🧬 TumorSpot":
    tumorspot_interface()

elif page == "💧 StainShift":
    stainshift_interface()

elif page == "✨ HeatmapTrust":
    heatmaptrust_interface()

elif page == "🛡️ HonestPath":
    honestpath_interface()

elif page == "ℹ️ About":
    about_interface()

else:
    home_interface()
    
