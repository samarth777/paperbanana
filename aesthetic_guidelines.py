"""
Aesthetic Guidelines (G) for academic illustration styling.
Based on Appendix F of the PaperBanana paper.
"""

AESTHETIC_GUIDELINE = """
# Academic Illustration Style Guide (NeurIPS Style)

## Color Palette
- **Overall Aesthetic:** Soft Tech & Scientific Pastels ("NeurIPS Look")
- **Background Colors:** Cream (#FFF8E7), Pale Blue (#E3F2FD), Mint (#E8F5E9)
- **Accent Colors:** 
  - Soft Blue (#64B5F6) for primary processes
  - Soft Orange (#FFB74D) for secondary/iterative processes
  - Soft Purple (#9575CD) for highlighting key components
  - Soft Green (#81C784) for success/outputs
- **Use color to group logical components**

## Shapes and Components
- **Process Boxes:** Rounded rectangles with subtle shadows
- **Data/Tensors:** 3D stacks or layered rectangles
- **Databases/Storage:** Cylinders or drum shapes
- **Agents/Models:** Robot or brain icons with labels
- **Inputs/Outputs:** Parallelograms or cloud shapes

## Lines and Arrows
- **Network/Architecture Diagrams:** Orthogonal/Elbow connectors
- **Logic Flow:** Curved arrows for feedback loops
- **Data Flow:** Straight arrows with clear directionality
- **Arrow Styles:** Solid for primary flow, dashed for optional/conditional

## Typography
- **Labels:** Sans-serif fonts (Arial, Roboto, Helvetica)
- **Mathematical Variables:** Serif Italic (Times New Roman) - use LaTeX notation (e.g., $P$, $P^*$)
- **Font Sizes:** 
  - Main labels: 12-14pt
  - Subscript/technical: 10pt
  - Section headers: 16pt bold

## Layout Principles
- **Hierarchy:** Left-to-right or top-to-bottom flow
- **Grouping:** Use containers/boxes with subtle backgrounds to group related components
- **Spacing:** Generous whitespace, consistent padding
- **Alignment:** Grid-based layout, aligned elements
- **Balance:** Visual weight distributed evenly

## Technical Details
- **Line Weight:** 1.5-2pt for main elements, 1pt for details
- **Corner Radius:** 8-12px for rounded rectangles
- **Shadow:** Subtle drop shadow (opacity 10-20%)
- **Icons:** Simple, consistent style throughout

## Diagram-Specific Guidelines
### Architecture Diagrams
- Show clear input → process → output flow
- Use containers to separate phases/stages
- Include feedback loops where applicable

### Methodology Diagrams  
- Emphasize the pipeline structure
- Show agent interactions clearly
- Use consistent icons for similar components
- Annotate with mathematical notation where relevant
"""
