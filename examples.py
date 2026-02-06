"""
Example usage of PaperBanana framework.

This script demonstrates how to use PaperBanana to generate academic illustrations.
"""
import os
from paperbanana import PaperBanana, generate_illustration

# Example methodology from a hypothetical paper
EXAMPLE_METHODOLOGY = """
Our proposed method consists of three main stages:

1. Feature Extraction: We use a pretrained ResNet-50 backbone to extract visual features 
   from input images. The features are pooled using adaptive average pooling to obtain 
   a fixed-size representation.

2. Attention Mechanism: We apply multi-head self-attention to capture long-range 
   dependencies between different spatial regions. The attention module has 8 heads 
   and uses scaled dot-product attention.

3. Classification Head: The attended features are passed through a two-layer MLP 
   with ReLU activation and dropout (p=0.5) for final classification. The output 
   layer uses softmax activation.

The entire model is trained end-to-end using cross-entropy loss with the Adam optimizer.
"""

EXAMPLE_CAPTION = "Architecture of our proposed attention-based image classification model"

# Example reference set (normally would be loaded from a database)
EXAMPLE_REFERENCE_SET = [
    {
        'id': 'ref_001',
        'domain': 'Computer Vision',
        'diagram_type': 'Architecture Diagram',
        'description': 'CNN architecture with attention modules showing feature extraction, attention layers, and classification head'
    },
    {
        'id': 'ref_002', 
        'domain': 'Computer Vision',
        'diagram_type': 'Pipeline Diagram',
        'description': 'Image processing pipeline from input through multiple stages to output'
    },
    {
        'id': 'ref_003',
        'domain': 'Natural Language Processing',
        'diagram_type': 'Architecture Diagram',
        'description': 'Transformer architecture with self-attention mechanism'
    },
]


def example_basic_usage():
    """Example 1: Basic usage with default settings."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80 + "\n")
    
    result = generate_illustration(
        methodology_text=EXAMPLE_METHODOLOGY,
        caption=EXAMPLE_CAPTION,
        output_path="examples/basic_example"
    )
    
    print(f"\nGenerated image: {result['final_image_path']}")
    print(f"Iterations: {result['iterations']}")


def example_with_references():
    """Example 2: Using reference examples."""
    print("\n" + "="*80)
    print("EXAMPLE 2: With Reference Examples")
    print("="*80 + "\n")
    
    result = generate_illustration(
        methodology_text=EXAMPLE_METHODOLOGY,
        caption=EXAMPLE_CAPTION,
        reference_set=EXAMPLE_REFERENCE_SET,
        output_path="examples/with_references"
    )
    
    print(f"\nGenerated image: {result['final_image_path']}")


def example_ablation_study():
    """Example 3: Ablation study - testing without certain components."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Ablation Study")
    print("="*80 + "\n")
    
    # Without styling
    print("\n--- Without Stylist Agent ---")
    result1 = generate_illustration(
        methodology_text=EXAMPLE_METHODOLOGY,
        caption=EXAMPLE_CAPTION,
        output_path="examples/ablation_no_style",
        skip_styling=True
    )
    
    # Without refinement
    print("\n--- Without Iterative Refinement ---")
    result2 = generate_illustration(
        methodology_text=EXAMPLE_METHODOLOGY,
        caption=EXAMPLE_CAPTION,
        output_path="examples/ablation_no_refinement",
        skip_refinement=True
    )


def example_statistical_plot():
    """Example 4: Generating statistical plots."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Statistical Plot Generation")
    print("="*80 + "\n")
    
    plot_description = """
    Create a line plot comparing accuracy across training epochs for three models:
    - Baseline CNN (blue line)
    - Our method without attention (orange line)  
    - Our full method (green line)
    
    X-axis: Training Epochs (0-100)
    Y-axis: Validation Accuracy (%)
    
    The baseline should plateau around 85%, method without attention around 88%,
    and full method should reach 92%.
    """
    
    # Example data (normally would come from actual experiments)
    plot_data = {
        'epochs': list(range(0, 101, 10)),
        'baseline': [60, 70, 75, 78, 80, 82, 83, 84, 85, 85, 85],
        'no_attention': [65, 75, 80, 83, 85, 86, 87, 87.5, 88, 88, 88],
        'full_method': [70, 80, 85, 87, 89, 90, 91, 91.5, 92, 92, 92]
    }
    
    pb = PaperBanana(mode="plot")
    result = pb.generate(
        methodology_text=plot_description,
        caption="Comparison of validation accuracy across training epochs",
        output_path="examples/accuracy_plot",
        data=plot_data
    )
    
    print(f"\nGenerated plot code: {result['final_image_path']}")
    print("Run the generated Python file to create the plot image.")


def example_with_neurips_references():
    """Example 5b: Using MinerU-parsed NeurIPS reference set."""
    print("\n" + "="*80)
    print("EXAMPLE 5b: With NeurIPS 2025 Reference Set (from MinerU)")
    print("="*80 + "\n")

    from load_reference_set import load_reference_set

    ref_set = load_reference_set()
    if not ref_set:
        print("No reference set found. Ensure data/spotlight_reference_set.json exists.")
        return

    result = generate_illustration(
        methodology_text=EXAMPLE_METHODOLOGY,
        caption=EXAMPLE_CAPTION,
        reference_set=ref_set,
        output_path="examples/neurips_refs"
    )
    print(f"\nGenerated image: {result['final_image_path']}")


def example_full_pipeline():
    """Example 6: Full pipeline with all features and history saving."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Full Pipeline with History")
    print("="*80 + "\n")
    
    pb = PaperBanana(
        reference_set=EXAMPLE_REFERENCE_SET,
        mode="diagram",
        max_iterations=3
    )
    
    result = pb.generate(
        methodology_text=EXAMPLE_METHODOLOGY,
        caption=EXAMPLE_CAPTION,
        output_path="examples/full_pipeline"
    )
    
    # Save generation history for analysis
    pb.save_history("examples/generation_history.json")
    
    print(f"\nFinal image: {result['final_image_path']}")
    print(f"Description versions: {len(result['history']['descriptions'])}")
    print(f"Critiques performed: {len(result['history']['critiques'])}")


def main():
    """Run all examples."""
    # Create examples directory
    os.makedirs("examples", exist_ok=True)
    
    print("\n" + "="*80)
    print("PaperBanana Examples")
    print("="*80)
    print("\nThese examples demonstrate various features of the PaperBanana framework.")
    print("Make sure you have set the GEMINI_API_KEY environment variable.\n")
    
    # Check for API key
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY environment variable not set!")
        print("Please set it with: export GEMINI_API_KEY='your-api-key'")
        return
    
    # Run examples (comment out any you don't want to run)
    try:
        # Example 1: Basic usage
        example_basic_usage()
        
        # Example 2: With references
        # example_with_references()
        
        # Example 3: Ablation study
        # example_ablation_study()
        
        # Example 4: Statistical plots
        # example_statistical_plot()
        
        # Example 5b: With NeurIPS MinerU references
        # example_with_neurips_references()
        
        # Example 6: Full pipeline
        # example_full_pipeline()
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Examples Complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
