# PaperBanana Implementation

Implementation of the **PaperBanana** framework from the paper "PaperBanana: Automating Academic Illustration for AI Scientists" (Zhu et al., 2025).

## Overview

PaperBanana is an agentic framework for automated generation of publication-ready academic illustrations. It orchestrates five specialized agents:

1. **Retriever Agent**: Finds relevant reference examples using VLM ranking
2. **Planner Agent**: Translates methodology into comprehensive textual descriptions
3. **Stylist Agent**: Applies aesthetic guidelines for publication quality
4. **Visualizer Agent**: Renders images using Gemini image generation models
5. **Critic Agent**: Provides iterative feedback for refinement

## Installation

```bash
pip install -r requirements.txt
```

## Setup

Set your Gemini API key as an environment variable:

```bash
export GEMINI_API_KEY='your-api-key-here'
```

## Quick Start

### Basic Usage

```python
from paperbanana import generate_illustration

methodology = """
Your methodology description here...
"""

caption = "Architecture of proposed method"

result = generate_illustration(
    methodology_text=methodology,
    caption=caption,
    output_path="output/my_diagram"
)

print(f"Generated: {result['final_image_path']}")
```

### Advanced Usage with All Features

```python
from paperbanana import PaperBanana

# Create reference set (optional)
reference_set = [
    {
        'id': 'ref_001',
        'domain': 'Computer Vision',
        'diagram_type': 'Architecture Diagram',
        'description': 'Description of reference diagram...'
    },
    # ... more references
]

# Initialize framework
pb = PaperBanana(
    reference_set=reference_set,
    mode="diagram",  # or "plot" for statistical plots
    max_iterations=3
)

# Generate illustration
result = pb.generate(
    methodology_text=methodology,
    caption=caption,
    output_path="output/diagram"
)

# Save generation history
pb.save_history("output/history.json")
```

### Generating Statistical Plots

```python
from paperbanana import PaperBanana

pb = PaperBanana(mode="plot")

plot_description = """
Create a line plot showing accuracy vs. epochs...
"""

result = pb.generate(
    methodology_text=plot_description,
    caption="Training accuracy comparison",
    output_path="output/plot",
    data={'epochs': [1,2,3], 'accuracy': [0.7, 0.8, 0.9]}
)

# This generates Python code - run it to create the plot
```

## Examples

Run the included examples:

```bash
python examples.py
```

This demonstrates:
- Basic diagram generation
- Using reference examples
- Ablation studies (skipping components)
- Statistical plot generation
- Full pipeline with history saving

## Project Structure

```
paperbanana_implementation/
├── agents/
│   ├── __init__.py
│   ├── retriever.py       # Retriever Agent
│   ├── planner.py         # Planner Agent
│   ├── stylist.py         # Stylist Agent
│   ├── visualizer.py      # Visualizer Agent
│   └── critic.py          # Critic Agent
├── paperbanana.py         # Main orchestration
├── config.py              # Configuration
├── aesthetic_guidelines.py # Style guide
├── utils.py               # Utilities
├── examples.py            # Example scripts
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Configuration

Edit `config.py` to customize:

- **Models**: VLM and image generation models
- **Iterations**: Maximum refinement iterations (default: 3)
- **Image size**: Output resolution
- **Reference count**: Number of examples to retrieve

## Ablation Studies

Test different configurations:

```python
# Without styling
result = generate_illustration(
    methodology_text=methodology,
    caption=caption,
    output_path="output",
    skip_styling=True
)

# Without refinement
result = generate_illustration(
    methodology_text=methodology,
    caption=caption,
    output_path="output",
    skip_refinement=True
)

# Without retrieval
result = generate_illustration(
    methodology_text=methodology,
    caption=caption,
    output_path="output",
    skip_retrieval=True
)
```

## Models Used

- **VLM**: `gemini-3-pro-preview` (for reasoning tasks)
- **Image Generation**: `gemini-3-pro-image-preview` (referred to as "Nano-Banana-Pro" in the paper)

## Features

✅ Multi-agent architecture  
✅ Iterative refinement with critic feedback  
✅ Aesthetic styling with NeurIPS guidelines  
✅ Reference example retrieval  
✅ Statistical plot generation  
✅ Complete generation history tracking  
✅ Ablation study support  

## Paper Reference

```bibtex
@article{zhu2025paperbanana,
  title={PaperBanana: Automating Academic Illustration for AI Scientists},
  author={Zhu, Dawei and Meng, Rui and Song, Yale and Wei, Xiyu and Li, Sujian and Pfister, Tomas and Yoon, Jinsung},
  journal={NeurIPS},
  year={2025}
}
```

## License

This implementation is for research and educational purposes.

## Notes

- The framework requires a valid Gemini API key
- Image generation may take several seconds per iteration
- Generated images are saved with iteration numbers for comparison
- Plot mode generates Python code that must be executed separately
