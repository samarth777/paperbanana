"""
PaperBanana: Automated Academic Illustration Generation Framework

Main orchestration module that coordinates all agents to generate
publication-ready academic illustrations.
"""
import os
from typing import Dict, List, Any, Optional
from agents import (
    RetrieverAgent,
    PlannerAgent, 
    StylistAgent,
    VisualizerAgent,
    CriticAgent
)
import config


class PaperBanana:
    """
    PaperBanana Framework: Orchestrates specialized agents for academic illustration generation.
    
    Pipeline:
    1. Retriever: Find relevant reference examples
    2. Planner: Create initial description
    3. Stylist: Add aesthetic refinements
    4. Visualizer: Generate image
    5. Critic: Provide feedback (iterate 2-4 if needed)
    """
    
    def __init__(self,
                 reference_set: List[Dict[str, Any]] = None,
                 custom_guidelines: str = None,
                 mode: str = "diagram",
                 max_iterations: int = None):
        """
        Initialize PaperBanana framework.
        
        Args:
            reference_set: List of reference examples for retrieval
            custom_guidelines: Optional custom aesthetic guidelines
            mode: Generation mode - "diagram" or "plot"
            max_iterations: Maximum refinement iterations (default from config)
        """
        self.retriever = RetrieverAgent(reference_set)
        self.planner = PlannerAgent()
        self.stylist = StylistAgent(custom_guidelines)
        self.visualizer = VisualizerAgent(mode)
        self.critic = CriticAgent()
        
        self.max_iterations = max_iterations or config.MAX_REFINEMENT_ITERATIONS
        self.mode = mode
        
        # Store intermediate results for analysis
        self.history = {
            'reference_examples': [],
            'descriptions': [],
            'critiques': [],
            'images': []
        }
    
    def generate(self,
                 methodology_text: str,
                 caption: str,
                 output_path: str = "output",
                 skip_retrieval: bool = False,
                 skip_styling: bool = False,
                 skip_refinement: bool = False,
                 data: dict = None) -> Dict[str, Any]:
        """
        Generate academic illustration from methodology description.
        
        Args:
            methodology_text: Source methodology description (S)
            caption: Target diagram caption (C)
            output_path: Base path for output files
            skip_retrieval: Skip reference retrieval (ablation)
            skip_styling: Skip aesthetic styling (ablation)
            skip_refinement: Skip iterative refinement (ablation)
            data: Optional data for plot mode
            
        Returns:
            Dictionary containing:
                - 'final_image_path': Path to generated image
                - 'final_description': Final textual description
                - 'history': Complete generation history
        """
        print("=" * 80)
        print("PaperBanana: Automated Academic Illustration Generation")
        print("=" * 80)
        
        # Step 1: Retrieve reference examples
        reference_examples = []
        if not skip_retrieval and self.retriever.reference_set:
            print("\n[1/5] Retriever Agent: Finding relevant examples...")
            reference_examples = self.retriever.retrieve(
                methodology_text, 
                caption,
                n=config.NUM_REFERENCE_EXAMPLES
            )
            self.history['reference_examples'] = reference_examples
            print(f"✓ Retrieved {len(reference_examples)} reference examples")
        else:
            print("\n[1/5] Retriever Agent: Skipped (no reference set)")
        
        # Step 2: Initial planning
        print("\n[2/5] Planner Agent: Creating initial description...")
        current_description = self.planner.plan(
            methodology_text,
            caption,
            reference_examples
        )
        self.history['descriptions'].append({
            'iteration': 0,
            'description': current_description,
            'type': 'initial'
        })
        print(f"✓ Generated initial description ({len(current_description)} chars)")
        
        # Step 3: Aesthetic styling
        if not skip_styling:
            print("\n[3/5] Stylist Agent: Applying aesthetic guidelines...")
            current_description = self.stylist.refine(current_description)
            self.history['descriptions'].append({
                'iteration': 0,
                'description': current_description,
                'type': 'styled'
            })
            print(f"✓ Applied styling ({len(current_description)} chars)")
        else:
            print("\n[3/5] Stylist Agent: Skipped")
        
        # Step 4 & 5: Iterative refinement loop
        iteration = 1
        final_image_path = None
        
        while iteration <= self.max_iterations:
            print(f"\n[4/5] Visualizer Agent: Generating image (Iteration {iteration})...")
            
            # Generate visualization
            image_path = self.visualizer.visualize(
                current_description,
                f"{output_path}_iter{iteration}",
                data
            )
            final_image_path = image_path
            self.history['images'].append({
                'iteration': iteration,
                'path': image_path,
                'description': current_description
            })
            print(f"✓ Generated image: {image_path}")
            
            # Skip refinement if requested or on last iteration
            if skip_refinement or iteration >= self.max_iterations:
                break
            
            print(f"\n[5/5] Critic Agent: Evaluating result (Iteration {iteration})...")
            
            # Get critique
            critique = self.critic.critique(
                methodology_text,
                caption,
                current_description,
                image_path,
                iteration
            )
            self.history['critiques'].append(critique)
            
            # Print critique summary
            print(f"✓ Found {len(critique['issues'])} issues")
            if critique['issues']:
                for i, issue in enumerate(critique['issues'][:3], 1):
                    print(f"  {i}. {issue[:80]}...")
            
            # Check if refinement should continue
            if not critique['should_continue']:
                print("✓ Critic: Quality threshold reached, stopping refinement")
                break
            
            if iteration >= self.max_iterations:
                print(f"✓ Reached maximum iterations ({self.max_iterations})")
                break
            
            # Generate refinement prompt
            refinement_prompt = self.critic.generate_refinement_prompt(
                current_description,
                critique
            )
            
            # Refine description (back to Planner)
            print(f"\n[2/5] Planner Agent: Refining description (Iteration {iteration + 1})...")
            
            # Use VLM to refine based on feedback
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=config.GEMINI_API_KEY)
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=refinement_prompt)]
                )
            ]
            
            refined_description = ""
            for chunk in client.models.generate_content_stream(
                model=config.VLM_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="HIGH")
                )
            ):
                refined_description += chunk.text
            
            current_description = refined_description.strip()
            self.history['descriptions'].append({
                'iteration': iteration,
                'description': current_description,
                'type': 'refined'
            })
            print(f"✓ Refined description ({len(current_description)} chars)")
            
            # Re-apply styling if not skipped
            if not skip_styling:
                print(f"\n[3/5] Stylist Agent: Re-applying aesthetic guidelines...")
                current_description = self.stylist.refine(current_description)
                self.history['descriptions'].append({
                    'iteration': iteration,
                    'description': current_description,
                    'type': 'refined_styled'
                })
                print(f"✓ Applied styling ({len(current_description)} chars)")
            
            iteration += 1
        
        print("\n" + "=" * 80)
        print("Generation Complete!")
        print("=" * 80)
        print(f"Final image: {final_image_path}")
        print(f"Total iterations: {iteration}")
        print(f"Total descriptions generated: {len(self.history['descriptions'])}")
        
        return {
            'final_image_path': final_image_path,
            'final_description': current_description,
            'iterations': iteration,
            'history': self.history
        }
    
    def save_history(self, filepath: str):
        """Save generation history to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved to: {filepath}")


# Convenience function
def generate_illustration(methodology_text: str,
                         caption: str,
                         reference_set: List[Dict[str, Any]] = None,
                         output_path: str = "output",
                         mode: str = "diagram",
                         **kwargs) -> Dict[str, Any]:
    """
    Convenience function to generate an illustration.
    
    Args:
        methodology_text: Methodology description
        caption: Diagram caption
        reference_set: Optional reference examples
        output_path: Output file path
        mode: "diagram" or "plot"
        **kwargs: Additional arguments for PaperBanana.generate()
        
    Returns:
        Generation results dictionary
    """
    pb = PaperBanana(reference_set=reference_set, mode=mode)
    return pb.generate(methodology_text, caption, output_path, **kwargs)
