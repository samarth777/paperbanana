"""
Retriever Agent for PaperBanana framework.

Identifies the N most relevant examples from a reference set using VLM ranking.
Matches based on research domain and diagram type.
"""
import os
from typing import List, Dict, Any
from google import genai
from google.genai import types
import config


class RetrieverAgent:
    """
    Retriever Agent: Identifies relevant reference examples from a fixed reference set.
    
    Uses generative retrieval approach where VLM ranks candidates by matching
    research domain and diagram type.
    """
    
    def __init__(self, reference_set: List[Dict[str, Any]] = None):
        """
        Initialize Retriever Agent.
        
        Args:
            reference_set: List of reference examples with metadata
                          Each example should have: {
                              'id': str,
                              'domain': str,
                              'diagram_type': str,
                              'description': str,
                              'image_path': str (optional)
                          }
        """
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.model = config.VLM_MODEL
        self.reference_set = reference_set or []
        
    def retrieve(self, 
                 methodology_text: str, 
                 caption: str,
                 n: int = config.NUM_REFERENCE_EXAMPLES) -> List[Dict[str, Any]]:
        """
        Retrieve the N most relevant reference examples.
        
        Args:
            methodology_text: Source methodology description
            caption: Target diagram caption
            n: Number of examples to retrieve
            
        Returns:
            List of N most relevant reference examples
        """
        if not self.reference_set:
            print("Warning: No reference set provided. Skipping retrieval.")
            return []
        
        # Create retrieval prompt
        prompt = self._create_retrieval_prompt(methodology_text, caption, n)
        
        # Query VLM for ranking
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
        
        response_text = ""
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=generate_config
        ):
            response_text += chunk.text
        
        # Parse the response to extract selected example IDs
        selected_examples = self._parse_retrieval_response(response_text, n)
        
        return selected_examples
    
    def _create_retrieval_prompt(self, methodology_text: str, caption: str, n: int) -> str:
        """Create prompt for retrieving relevant examples."""
        # Create a summary of available references
        reference_summary = "\n".join([
            f"ID: {ref['id']}\nDomain: {ref['domain']}\nType: {ref['diagram_type']}\nDescription: {ref['description']}\n"
            for ref in self.reference_set
        ])
        
        prompt = f"""You are an expert at identifying relevant academic illustration examples.

Given a methodology description and diagram caption, select the {n} most relevant reference examples 
from the provided set. Consider:
1. Research domain similarity (e.g., NLP, Computer Vision, Reinforcement Learning)
2. Diagram type similarity (e.g., architecture diagram, flowchart, pipeline)
3. Conceptual similarity in the methodology

METHODOLOGY:
{methodology_text}

TARGET CAPTION:
{caption}

AVAILABLE REFERENCE EXAMPLES:
{reference_summary}

OUTPUT FORMAT:
Return only the IDs of the {n} most relevant examples, one per line, ranked from most to least relevant.
Example output:
ref_001
ref_005
ref_012
"""
        return prompt
    
    def _parse_retrieval_response(self, response_text: str, n: int) -> List[Dict[str, Any]]:
        """Parse VLM response to extract selected examples."""
        # Extract IDs from response
        lines = response_text.strip().split('\n')
        selected_ids = []
        
        for line in lines:
            line = line.strip()
            # Look for reference IDs
            for ref in self.reference_set:
                if ref['id'] in line:
                    selected_ids.append(ref['id'])
                    break
            if len(selected_ids) >= n:
                break
        
        # Get full reference objects
        selected_examples = []
        for ref_id in selected_ids:
            for ref in self.reference_set:
                if ref['id'] == ref_id:
                    selected_examples.append(ref)
                    break
        
        # If we didn't get enough, just take the first n
        if len(selected_examples) < n:
            selected_examples = self.reference_set[:n]
        
        return selected_examples[:n]
