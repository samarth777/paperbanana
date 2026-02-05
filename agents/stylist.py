"""
Stylist Agent for PaperBanana framework.

Acts as a design consultant. Uses automatically synthesized aesthetic 
guidelines to refine initial description into stylistically optimized version.
"""
import os
from google import genai
from google.genai import types
import config
from aesthetic_guidelines import AESTHETIC_GUIDELINE


class StylistAgent:
    """
    Stylist Agent: Refines illustration descriptions using aesthetic guidelines.
    
    Takes initial description P and enhances it with style guidance G 
    to produce stylistically optimized description P*.
    """
    
    def __init__(self, custom_guidelines: str = None):
        """
        Initialize Stylist Agent.
        
        Args:
            custom_guidelines: Optional custom aesthetic guidelines.
                             If None, uses default NeurIPS-style guidelines.
        """
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.model = config.VLM_MODEL
        self.guidelines = custom_guidelines or AESTHETIC_GUIDELINE
    
    def refine(self, initial_description: str) -> str:
        """
        Refine initial description with aesthetic styling.
        
        Args:
            initial_description: Initial textual description P
            
        Returns:
            Stylistically optimized description P*
        """
        prompt = self._create_styling_prompt(initial_description)
        
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        ]
        
        generate_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level=config.THINKING_LEVEL
            )
        )
        
        refined_description = ""
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=generate_config
        ):
            refined_description += chunk.text
        
        return refined_description.strip()
    
    def _create_styling_prompt(self, initial_description: str) -> str:
        """Create prompt for aesthetic refinement."""
        prompt = f"""You are an expert design consultant specializing in academic publication illustrations.

Your task is to take an initial diagram description and enhance it with specific aesthetic and design details 
to create a polished, publication-ready illustration that follows academic standards.

INITIAL DESCRIPTION:
{initial_description}

AESTHETIC GUIDELINES TO FOLLOW:
{self.guidelines}

YOUR TASK:
Refine the initial description by adding specific visual design details:

1. **Color Specifications**: Add specific color choices from the palette (e.g., "soft blue #64B5F6 for the main process boxes")
2. **Shape Details**: Specify exact shapes and their styling (e.g., "rounded rectangles with 10px radius and subtle shadow")
3. **Typography**: Define font choices for different text elements
4. **Visual Hierarchy**: Enhance descriptions of size, weight, and emphasis relationships
5. **Spacing & Layout**: Add details about padding, margins, and alignment
6. **Professional Polish**: Include finishing touches like shadows, borders, gradients

IMPORTANT:
- Preserve ALL content and structural information from the initial description
- Add aesthetic details WITHOUT changing the fundamental design or information flow
- Be specific with measurements, colors (hex codes), and styling parameters
- Ensure the result maintains academic professionalism and clarity
- The output should be suitable for direct input to an image generation model

OUTPUT FORMAT:
Provide the enhanced description as a detailed, flowing paragraph that seamlessly integrates 
the original content with the aesthetic specifications. Make it vivid and precise enough that 
an image generation model can render it accurately.
"""
        return prompt
