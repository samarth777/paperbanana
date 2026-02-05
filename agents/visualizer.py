"""
Visualizer Agent for PaperBanana framework.

Renders academic illustrations using image generation models.
Supports both diagram generation and statistical plot generation.
"""
import os
import mimetypes
from typing import Optional
from google import genai
from google.genai import types
import config
from utils import save_binary_file


class VisualizerAgent:
    """
    Visualizer Agent: Renders illustrations from textual descriptions.
    
    Supports two modes:
    1. Diagram mode: Uses image generation model (Nano-Banana-Pro / Gemini Image)
    2. Plot mode: Generates Python Matplotlib code for statistical plots
    """
    
    def __init__(self, mode: str = "diagram"):
        """
        Initialize Visualizer Agent.
        
        Args:
            mode: Generation mode - "diagram" or "plot"
        """
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.mode = mode
        
        if mode == "diagram":
            self.model = config.IMAGE_MODEL
        elif mode == "plot":
            self.model = config.VLM_MODEL  # Use VLM for code generation
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'diagram' or 'plot'")
    
    def visualize(self, 
                  description: str,
                  output_path: str = "output",
                  data: dict = None) -> str:
        """
        Generate visualization from description.
        
        Args:
            description: Textual description of the illustration
            output_path: Base path for output file (without extension)
            data: Optional data dict for plot mode
            
        Returns:
            Path to generated image file or code file
        """
        if self.mode == "diagram":
            return self._generate_diagram(description, output_path)
        elif self.mode == "plot":
            return self._generate_plot(description, output_path, data)
    
    def _generate_diagram(self, description: str, output_path: str) -> str:
        """
        Generate diagram image using image generation model.
        
        Args:
            description: Detailed visual description
            output_path: Base path for output file
            
        Returns:
            Path to generated image
        """
        # Create prompt for image generation
        prompt = f"""Generate a high-quality academic methodology diagram with the following specifications:

{description}

Requirements:
- Professional academic publication quality
- Clear, readable text and labels
- Consistent styling throughout
- Appropriate use of colors and shapes
- Publication-ready resolution
"""
        
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        ]
        
        generate_config = types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            image_config=types.ImageConfig(
                image_size=config.IMAGE_SIZE
            )
        )
        
        file_index = 0
        saved_path = None
        
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=generate_config
        ):
            if (chunk.candidates is None or 
                chunk.candidates[0].content is None or 
                chunk.candidates[0].content.parts is None):
                continue
            
            # Check for inline image data
            part = chunk.candidates[0].content.parts[0]
            if part.inline_data and part.inline_data.data:
                inline_data = part.inline_data
                data_buffer = inline_data.data
                file_extension = mimetypes.guess_extension(inline_data.mime_type)
                
                if file_extension:
                    file_name = f"{output_path}_{file_index}{file_extension}"
                    saved_path = save_binary_file(file_name, data_buffer)
                    file_index += 1
            else:
                # Print any text output
                if chunk.text:
                    print(chunk.text)
        
        return saved_path or f"{output_path}_0.png"
    
    def _generate_plot(self, description: str, output_path: str, data: dict = None) -> str:
        """
        Generate statistical plot by creating Matplotlib code.
        
        Args:
            description: Description of desired plot
            output_path: Base path for output code file
            data: Optional data dictionary
            
        Returns:
            Path to generated Python code file
        """
        data_context = ""
        if data:
            data_context = f"\n\nDATA PROVIDED:\n{str(data)}\n"
        
        prompt = f"""You are an expert at creating publication-quality statistical plots using Matplotlib.

Generate complete, executable Python code using Matplotlib to create the following plot:

{description}
{data_context}

Requirements:
1. Use professional academic styling (seaborn-paper style or similar)
2. Include clear axis labels with units
3. Add legend if multiple series
4. Use appropriate colors and markers
5. Set figure size for publication (e.g., 6x4 inches)
6. Save as high-resolution PNG (300 dpi minimum)
7. Include error bars if applicable
8. Follow best practices for data visualization

OUTPUT FORMAT:
Provide ONLY the complete Python code, ready to execute. 
Start with necessary imports and end with plt.savefig().
Do not include any explanations outside the code comments.
"""
        
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        ]
        
        generate_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level="MEDIUM"
            )
        )
        
        code = ""
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=generate_config
        ):
            code += chunk.text
        
        # Save code to file
        code_file = f"{output_path}.py"
        with open(code_file, 'w') as f:
            f.write(code.strip())
        
        print(f"Plot code saved to: {code_file}")
        print("Run the code to generate the plot image.")
        
        return code_file
