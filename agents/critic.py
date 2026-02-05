"""
Critic Agent for PaperBanana framework.

Forms closed-loop refinement mechanism by identifying factual misalignments 
or visual glitches and providing feedback for iterative improvement.
"""
import os
from typing import Dict, List
from google import genai
from google.genai import types
import config


class CriticAgent:
    """
    Critic Agent: Provides iterative feedback for refinement.
    
    Identifies factual misalignments, visual glitches, and areas for improvement
    in generated illustrations, enabling closed-loop refinement.
    """
    
    def __init__(self):
        """Initialize Critic Agent."""
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.model = config.VLM_MODEL
    
    def critique(self,
                 methodology_text: str,
                 caption: str,
                 current_description: str,
                 generated_image_path: str = None,
                 iteration: int = 1) -> Dict[str, any]:
        """
        Provide critique and feedback on current illustration.
        
        Args:
            methodology_text: Original methodology description
            caption: Target diagram caption
            current_description: Current textual description
            generated_image_path: Path to generated image (if available)
            iteration: Current iteration number
            
        Returns:
            Dictionary containing:
                - 'feedback': Textual feedback
                - 'issues': List of identified issues
                - 'suggestions': List of improvement suggestions
                - 'should_continue': Boolean indicating if refinement should continue
        """
        prompt = self._create_critique_prompt(
            methodology_text, 
            caption, 
            current_description,
            iteration
        )
        
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        ]
        
        # If we have an image, we could add it to the critique (future enhancement)
        # For now, we critique based on the description
        
        generate_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level=config.THINKING_LEVEL
            )
        )
        
        critique_text = ""
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=generate_config
        ):
            critique_text += chunk.text
        
        # Parse critique into structured feedback
        result = self._parse_critique(critique_text, iteration)
        
        return result
    
    def _create_critique_prompt(self,
                                methodology_text: str,
                                caption: str,
                                current_description: str,
                                iteration: int) -> str:
        """Create prompt for critique generation."""
        prompt = f"""You are an expert reviewer of academic illustrations, specializing in methodology diagrams.

Your task is to critically evaluate a textual description for an academic diagram and provide constructive feedback.

ORIGINAL METHODOLOGY:
{methodology_text}

TARGET CAPTION:
{caption}

CURRENT ILLUSTRATION DESCRIPTION (Iteration {iteration}):
{current_description}

EVALUATION CRITERIA:

1. **Faithfulness**: Does the description accurately represent all key aspects of the methodology?
   - Are all important components mentioned?
   - Is the flow/logic correctly represented?
   - Are there any factual errors or misrepresentations?

2. **Conciseness**: Is the description appropriately detailed without being cluttered?
   - Is information density appropriate?
   - Are there redundant elements?
   - Is anything unnecessarily complex?

3. **Readability**: Will the resulting diagram be easy to understand?
   - Is the layout logical?
   - Are labels clear and informative?
   - Is visual hierarchy appropriate?

4. **Aesthetics**: Does the description specify professional visual design?
   - Are colors, shapes, and typography well-defined?
   - Is there visual consistency?
   - Does it match academic publication standards?

YOUR TASK:
Provide a structured critique covering:

ISSUES FOUND:
- List specific problems (e.g., "Missing connection between X and Y")
- Rate severity: CRITICAL, MAJOR, or MINOR

SUGGESTIONS FOR IMPROVEMENT:
- Provide concrete, actionable suggestions
- Prioritize by impact

OVERALL ASSESSMENT:
- Is this ready for visualization, or does it need refinement?
- If iteration {iteration} < 3, should we continue refining?

OUTPUT FORMAT:
Structure your response as:

ISSUES:
1. [SEVERITY] Issue description
2. [SEVERITY] Issue description
...

SUGGESTIONS:
1. Specific suggestion
2. Specific suggestion
...

DECISION: [READY / NEEDS_REFINEMENT]
REASONING: Brief explanation of the decision
"""
        return prompt
    
    def _parse_critique(self, critique_text: str, iteration: int) -> Dict:
        """Parse critique text into structured format."""
        issues = []
        suggestions = []
        should_continue = True
        
        # Simple parsing - look for key sections
        lines = critique_text.split('\n')
        current_section = None
        
        for line in lines:
            line_upper = line.upper().strip()
            
            if 'ISSUES:' in line_upper:
                current_section = 'issues'
                continue
            elif 'SUGGESTIONS:' in line_upper or 'SUGGESTION' in line_upper:
                current_section = 'suggestions'
                continue
            elif 'DECISION:' in line_upper:
                current_section = 'decision'
                if 'READY' in line_upper and 'NEEDS_REFINEMENT' not in line_upper:
                    should_continue = False
                continue
            
            # Parse content
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if current_section == 'issues' and (line.startswith('-') or line[0].isdigit()):
                issues.append(line.lstrip('-').lstrip('0123456789.').strip())
            elif current_section == 'suggestions' and (line.startswith('-') or line[0].isdigit()):
                suggestions.append(line.lstrip('-').lstrip('0123456789.').strip())
        
        # Don't continue past max iterations
        if iteration >= config.MAX_REFINEMENT_ITERATIONS:
            should_continue = False
        
        return {
            'feedback': critique_text,
            'issues': issues,
            'suggestions': suggestions,
            'should_continue': should_continue
        }
    
    def generate_refinement_prompt(self,
                                   original_description: str,
                                   critique: Dict) -> str:
        """
        Generate prompt for refinement based on critique.
        
        Args:
            original_description: Current description
            critique: Critique dictionary from critique()
            
        Returns:
            Prompt for Planner to refine the description
        """
        issues_text = "\n".join([f"- {issue}" for issue in critique['issues']])
        suggestions_text = "\n".join([f"- {sug}" for sug in critique['suggestions']])
        
        refinement_prompt = f"""CURRENT DESCRIPTION:
{original_description}

IDENTIFIED ISSUES:
{issues_text}

SUGGESTIONS FOR IMPROVEMENT:
{suggestions_text}

Please revise the description to address these issues and incorporate the suggestions.
Maintain all correct elements while fixing the identified problems.
"""
        return refinement_prompt
