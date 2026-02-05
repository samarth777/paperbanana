"""
Planner Agent for PaperBanana framework.

Serves as the cognitive core. Translates unstructured methodology data 
into comprehensive textual description of the target illustration.
"""
import os
from typing import List, Dict, Any
from google import genai
from google.genai import types
import config


class PlannerAgent:
    """
    Planner Agent: Translates methodology into comprehensive illustration description.
    
    The cognitive core that interprets source context S and communicative intent C,
    then produces detailed textual description P of the target illustration.
    """
    
    def __init__(self):
        """Initialize Planner Agent."""
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.model = config.VLM_MODEL
    
    def plan(self,
             methodology_text: str,
             caption: str,
             reference_examples: List[Dict[str, Any]] = None) -> str:
        """
        Generate comprehensive textual description of target illustration.
        
        Args:
            methodology_text: Source methodology description (S)
            caption: Diagram caption (part of C)
            reference_examples: Retrieved reference examples (E)
            
        Returns:
            Detailed textual description P of the illustration
        """
        prompt = self._create_planning_prompt(methodology_text, caption, reference_examples)
        
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
        
        description = ""
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=generate_config
        ):
            description += chunk.text
        
        return description.strip()
    
    def _create_planning_prompt(self,
                                methodology_text: str,
                                caption: str,
                                reference_examples: List[Dict[str, Any]] = None) -> str:
        """Create prompt for generating illustration description."""
        
        # Include reference examples if available
        reference_context = ""
        if reference_examples:
            reference_context = "\n\nREFERENCE EXAMPLES (for inspiration):\n"
            for i, ref in enumerate(reference_examples[:3], 1):  # Use top 3
                reference_context += f"\nExample {i}:\n"
                reference_context += f"Domain: {ref.get('domain', 'N/A')}\n"
                reference_context += f"Type: {ref.get('diagram_type', 'N/A')}\n"
                reference_context += f"Description: {ref.get('description', 'N/A')}\n"
        
        prompt = f"""You are an expert at designing academic methodology diagrams for scientific publications.

Your task is to create a COMPREHENSIVE and DETAILED textual description of an illustration that would 
effectively visualize the given methodology. This description will be used to generate the actual diagram.

METHODOLOGY TO VISUALIZE:
{methodology_text}

TARGET DIAGRAM CAPTION:
{caption}
{reference_context}

REQUIREMENTS:
1. **Layout Structure**: Specify the overall layout (left-to-right, top-to-bottom, circular, etc.)
2. **Components**: List all visual elements needed (boxes, arrows, icons, labels, etc.)
3. **Content**: What text/symbols should appear in each component
4. **Connections**: How components connect (arrows, lines, groupings)
5. **Hierarchy**: Which elements are primary vs secondary
6. **Grouping**: How to group related components (containers, background colors)
7. **Flow**: The logical flow of information through the diagram
8. **Key Details**: Important technical details, equations, or annotations

IMPORTANT GUIDELINES:
- Be specific about spatial relationships and positioning
- Describe the logical flow clearly (input → process → output)
- Include any mathematical notation or technical terminology
- Consider the target audience (academic researchers)
- Focus on clarity and information density
- Think about how this supports the paper's narrative

OUTPUT FORMAT:
Provide a detailed paragraph-form description that covers all aspects above. 
Be thorough - this description should be sufficient for someone to create the diagram without seeing the original methodology.
"""
        return prompt
