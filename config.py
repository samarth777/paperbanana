"""
Configuration for PaperBanana framework.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Gemini API Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Model Configuration
VLM_MODEL = "gemini-3-pro-preview"  # For Retriever, Planner, Stylist, Critic
IMAGE_MODEL = "gemini-3-pro-image-preview"  # For Visualizer (referred to as Nano-Banana-Pro in paper)

# Generation Configuration
MAX_REFINEMENT_ITERATIONS = 3  # As per ablation study
IMAGE_SIZE = "1K"  # Image resolution
THINKING_LEVEL = "HIGH"  # For complex reasoning tasks

# Number of reference examples to retrieve
NUM_REFERENCE_EXAMPLES = 10
