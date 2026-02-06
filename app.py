"""
PaperBanana — Gradio app for HuggingFace Spaces.

Turns methodology text into publication-ready architecture diagrams
using a 5-agent pipeline (Retriever → Planner → Stylist → Visualizer → Critic).
"""

import os
import json
import tempfile
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional

import gradio as gr
from google import genai
from google.genai import types

from agents import RetrieverAgent, PlannerAgent, StylistAgent, VisualizerAgent, CriticAgent
from aesthetic_guidelines import AESTHETIC_GUIDELINE
import config

# ── Load reference set at startup ───────────────────────────────────────────
REF_SET_PATH = Path("data/spotlight_reference_set.json")
REFERENCE_SET: List[Dict[str, Any]] = []
if REF_SET_PATH.exists():
    with open(REF_SET_PATH) as f:
        REFERENCE_SET = json.load(f)
    print(f"Loaded {len(REFERENCE_SET)} reference examples")

# ── Example gallery images ──────────────────────────────────────────────────
EXAMPLE_IMAGES = {
    "Transformer": "examples/readme/transformer_iter3_0.jpg",
    "ResNet": "examples/readme/resnet_iter3_0.jpg",
    "DDPM": "examples/readme/ddpm_iter3_0.jpg",
}

# ── Preset examples ─────────────────────────────────────────────────────────
PRESET_EXAMPLES = [
    [
        # Transformer
        """The Transformer model follows an encoder-decoder structure using stacked self-attention and fully connected layers.

Encoder: Stack of N=6 identical layers. Each layer has two sub-layers: (1) multi-head self-attention, and (2) position-wise feed-forward network. Residual connections around each sub-layer, followed by layer normalization.

Decoder: Stack of N=6 identical layers. In addition to the two encoder sub-layers, the decoder inserts a third sub-layer for multi-head cross-attention over the encoder output. Masked self-attention prevents attending to subsequent positions.

Multi-Head Attention: Linearly project queries, keys, values h times, perform scaled dot-product attention in parallel, concatenate and project again.

Positional Encoding: Sinusoidal positional encodings added to input embeddings.""",
        "The Transformer — model architecture (Vaswani et al., 2017)",
        2,
    ],
    [
        # ResNet
        """We present a residual learning framework. Instead of learning H(x) directly, layers fit a residual mapping F(x) = H(x) - x. The building block is y = F(x, {W_i}) + x via identity shortcut connections.

Architecture: Input 224×224 → 7×7 conv, 64, stride 2 → BN → ReLU → 3×3 max pool → Stage 1: 3 blocks, 64 filters → Stage 2: 4 blocks, 128 filters → Stage 3: 6 blocks, 256 filters → Stage 4: 3 blocks, 512 filters → Global avg pool → 1000-d FC → softmax.

For deeper networks (50/101/152), bottleneck blocks: 1×1 conv (reduce) → 3×3 conv → 1×1 conv (restore), with shortcut bypassing all three layers.""",
        "Architecture of ResNet with residual learning building blocks (He et al., 2016)",
        2,
    ],
    [
        # DDPM
        """Denoising diffusion probabilistic models (DDPMs): Forward process gradually adds Gaussian noise over T timesteps: q(x_t|x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_tI). After T steps, x_T ≈ N(0,I).

Reverse process learns to denoise: p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t)). Starting from x_T ~ N(0,I), iteratively produces clean x_0.

Denoising network ε_θ(x_t,t) is a U-Net: downsampling with ResNet blocks + self-attention at 16×16, bottleneck with self-attention, upsampling with skip connections. Timestep conditioning via sinusoidal embeddings. Training minimizes L = E[||ε - ε_θ(x_t,t)||²].""",
        "Overview of the denoising diffusion probabilistic model (Ho et al., 2020)",
        2,
    ],
]


# ── Core generation logic (streaming-friendly) ─────────────────────────────
def generate_diagram(
    methodology_text: str,
    caption: str,
    num_iterations: int,
    api_key: str | None = None,
    progress=gr.Progress(track_tqdm=True),
):
    """Run the full PaperBanana pipeline and yield intermediate results."""

    # Resolve API key: user input > env var
    gemini_key = (api_key or "").strip() or config.GEMINI_API_KEY
    if not gemini_key:
        raise gr.Error(
            "No Gemini API key found. Paste one in the field above, "
            "or set GEMINI_API_KEY as a Space secret."
        )

    # Patch config so all agents pick it up
    config.GEMINI_API_KEY = gemini_key

    num_iterations = int(num_iterations)
    logs: list[str] = []

    def log(msg: str):
        logs.append(msg)
        return "\n".join(logs)

    # ── 1. Retriever ────────────────────────────────────────────────────────
    yield None, log("🔍 [1/5] Retriever: finding relevant references…")
    retriever = RetrieverAgent(REFERENCE_SET)
    reference_examples = []
    if REFERENCE_SET:
        reference_examples = retriever.retrieve(
            methodology_text, caption, n=config.NUM_REFERENCE_EXAMPLES
        )
        yield None, log(f"   ✓ Retrieved {len(reference_examples)} references")
    else:
        yield None, log("   ⏭ Skipped (no reference set loaded)")

    # ── 2. Planner ──────────────────────────────────────────────────────────
    yield None, log("📝 [2/5] Planner: creating visual description…")
    planner = PlannerAgent()
    current_description = planner.plan(methodology_text, caption, reference_examples)
    yield None, log(f"   ✓ Description ready ({len(current_description)} chars)")

    # ── 3. Stylist ──────────────────────────────────────────────────────────
    yield None, log("🎨 [3/5] Stylist: applying aesthetic guidelines…")
    stylist = StylistAgent()
    current_description = stylist.refine(current_description)
    yield None, log(f"   ✓ Styled ({len(current_description)} chars)")

    # ── 4/5. Visualize → Critique loop ──────────────────────────────────────
    latest_image_path = None
    critic = CriticAgent()

    for i in range(1, num_iterations + 1):
        yield latest_image_path, log(
            f"🖼️ [4/5] Visualizer: generating image (iteration {i}/{num_iterations})…"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_base = os.path.join(tmpdir, f"iter{i}")
            visualizer = VisualizerAgent(mode="diagram")
            img_path = visualizer.visualize(current_description, out_base)

            if img_path and os.path.exists(img_path):
                # Copy to a persistent temp file so Gradio can serve it
                import shutil

                ext = Path(img_path).suffix or ".jpg"
                persist = tempfile.NamedTemporaryFile(
                    suffix=ext, delete=False, dir=tempfile.gettempdir()
                )
                shutil.copy2(img_path, persist.name)
                latest_image_path = persist.name

        yield latest_image_path, log(f"   ✓ Image generated (iteration {i})")

        # Skip critique on last iteration
        if i >= num_iterations:
            break

        yield latest_image_path, log(
            f"🔬 [5/5] Critic: evaluating (iteration {i})…"
        )
        critique = critic.critique(
            methodology_text, caption, current_description, latest_image_path, i
        )
        n_issues = len(critique["issues"])
        yield latest_image_path, log(f"   ✓ {n_issues} issues found")

        if not critique["should_continue"]:
            yield latest_image_path, log("   ✓ Quality threshold reached — done!")
            break

        # Refine
        yield latest_image_path, log("📝 [2/5] Planner: refining description…")
        refinement_prompt = critic.generate_refinement_prompt(
            current_description, critique
        )
        client = genai.Client(api_key=gemini_key)
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=refinement_prompt)],
            )
        ]
        refined = ""
        for chunk in client.models.generate_content_stream(
            model=config.VLM_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="HIGH")
            ),
        ):
            refined += chunk.text
        current_description = refined.strip()
        yield latest_image_path, log(
            f"   ✓ Refined ({len(current_description)} chars)"
        )

        # Re-style
        yield latest_image_path, log("🎨 [3/5] Stylist: re-applying style…")
        current_description = stylist.refine(current_description)
        yield latest_image_path, log(f"   ✓ Styled ({len(current_description)} chars)")

    yield latest_image_path, log("\n✅ Generation complete!")


# ── Gradio UI ───────────────────────────────────────────────────────────────
DESCRIPTION_MD = """\
# 🍌 PaperBanana

**Turn methodology text into publication-ready architecture diagrams.**

Paste your paper's methodology section + a caption, and PaperBanana's 5-agent pipeline
(Retriever → Planner → Stylist → Visualizer → Critic) will generate a diagram for you.

> Based on [*PaperBanana: Automating Academic Illustration for AI Scientists*](https://arxiv.org/abs/2505.23894) (Zhu et al., NeurIPS 2025).
"""

with gr.Blocks(title="PaperBanana") as demo:
    gr.Markdown(DESCRIPTION_MD)

    # ── Example gallery ─────────────────────────────────────────────────────
    with gr.Accordion("📸 Example outputs (click to expand)", open=False):
        existing = {k: v for k, v in EXAMPLE_IMAGES.items() if Path(v).exists()}
        if existing:
            with gr.Row():
                for name, path in existing.items():
                    with gr.Column(min_width=200):
                        gr.Image(value=path, label=name)

    # ── Inputs ──────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            methodology_input = gr.Textbox(
                label="Methodology text",
                placeholder="Paste your methodology / model description here…",
                lines=12,
            )
            caption_input = gr.Textbox(
                label="Diagram caption",
                placeholder='e.g. "Architecture of our proposed method"',
                lines=2,
            )
            iterations_slider = gr.Slider(
                minimum=1,
                maximum=3,
                value=2,
                step=1,
                label="Refinement iterations",
                info="More iterations = better quality, slower",
            )
            api_key_input = gr.Textbox(
                label="Gemini API key (optional if set as Space secret)",
                type="password",
                placeholder="AIza…",
            )
            generate_btn = gr.Button("🍌 Generate diagram", variant="primary", size="lg")

        # ── Outputs ─────────────────────────────────────────────────────────
        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated diagram", type="filepath")
            output_log = gr.Textbox(label="Pipeline log", lines=18, interactive=False)

    # ── Examples table ──────────────────────────────────────────────────────
    gr.Examples(
        examples=PRESET_EXAMPLES,
        inputs=[methodology_input, caption_input, iterations_slider],
        label="Try a classic paper",
    )

    # ── Wiring ──────────────────────────────────────────────────────────────
    generate_btn.click(
        fn=generate_diagram,
        inputs=[methodology_input, caption_input, iterations_slider, api_key_input],
        outputs=[output_image, output_log],
    )

if __name__ == "__main__":
    demo.queue().launch(
        theme=gr.themes.Soft(primary_hue="amber", secondary_hue="blue"),
        css="footer { display: none !important; }",
    )
