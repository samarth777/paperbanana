# 🍌 PaperBanana

> **Unofficial open-source implementation** of ["PaperBanana: Automating Academic Illustration for AI Scientists"](https://arxiv.org/abs/2601.23265) (Zhu et al.).


[![Try it on HuggingFace Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Samarth0710/PaperBanana)


## Demo

Try PaperBanana directly in your browser — no setup required:

**🔗 [huggingface.co/spaces/Samarth0710/PaperBanana](https://huggingface.co/spaces/Samarth0710/PaperBanana)**


## How It Works

<p align="center">
  <img src="docs/method_diagram.png" width="700" />
</p>

<p align="center"><em>PaperBanana pipeline overview — figure from <a href="https://arxiv.org/abs/2601.23265">Zhu et al., 2025</a></em></p>

## Installation

```bash
pip install -r requirements.txt
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

### Setup

Create a `.env` file with your Gemini API key:

```
GEMINI_API_KEY=your-api-key-here
```

## Quick Start

```python
from paperbanana import generate_illustration
from load_reference_set import load_reference_set

ref_set = load_reference_set()  # 100 curated NeurIPS 2025 diagrams

result = generate_illustration(
    methodology_text="Our model uses a Vision Transformer backbone ...",
    caption="Architecture of our proposed vision-language fusion model",
    reference_set=ref_set,
    output_path="output/my_diagram",
)
print(f"Generated: {result['final_image_path']}")
```

<details>
<summary><strong>Advanced usage</strong></summary>

```python
from paperbanana import PaperBanana
from load_reference_set import load_reference_set

pb = PaperBanana(
    reference_set=load_reference_set(),
    mode="diagram",       # or "plot" for statistical plots
    max_iterations=3,
)

result = pb.generate(
    methodology_text=methodology,
    caption=caption,
    output_path="output/diagram",
)

pb.save_history("output/history.json")
```

</details>

## Project Structure

```
paperbanana/
├── paperbanana.py              # Main orchestration
├── app.py                      # Gradio web UI
├── config.py                   # API keys & model config
├── aesthetic_guidelines.py     # NeurIPS-style visual guidelines
├── utils.py                    # Shared utilities
├── load_reference_set.py       # Reference set loader
├── examples.py                 # Runnable examples
├── agents/
│   ├── retriever.py            # Retriever Agent  (VLM-based ranking)
│   ├── planner.py              # Planner Agent    (methodology → description)
│   ├── stylist.py              # Stylist Agent    (aesthetic refinement)
│   ├── visualizer.py           # Visualizer Agent (image generation)
│   └── critic.py               # Critic Agent     (evaluate & feedback)
├── data/
│   ├── spotlight_reference_set.json
│   └── spotlight_reference_images/
├── docs/                       # Paper figures & notes
├── examples/                   # Generated output images
├── Dockerfile                  # HF Spaces Docker config
└── requirements.txt
```

## Configuration

Edit `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `VLM_MODEL` | `gemini-3-pro-preview` | Reasoning model (Retriever, Planner, Stylist, Critic) |
| `IMAGE_MODEL` | `gemini-3-pro-image-preview` | Image generation model (Visualizer) |
| `MAX_REFINEMENT_ITERATIONS` | `3` | Planner↔Critic loop iterations |
| `NUM_REFERENCE_EXAMPLES` | `10` | References retrieved per generation |

## Citation

This is an unofficial implementation. Please cite the original paper:

```bibtex
@article{zhu2025paperbanana,
  title={PaperBanana: Automating Academic Illustration for AI Scientists},
  author={Zhu, Dawei and Meng, Rui and Song, Yale and Wei, Xiyu and Li, Sujian and Pfister, Tomas and Yoon, Jinsung},
  journal={NeurIPS},
  year={2025}
}
```

## License

MIT — this implementation is for research and educational purposes.
