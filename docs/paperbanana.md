# PaperBanana: Automating Academic Illustration for AI Scientists

**Dawei Zhu**$^{1,2,*}$, **Rui Meng**$^2$, **Yale Song**$^2$, **Xiyu Wei**$^1$, **Sujian Li**$^1$, **Tomas Pfister**$^2$ and **Jinsung Yoon**$^2$  
$^1$Peking University, $^2$Google Cloud AI Research  
[https://dwzhu-pku.github.io/PaperBanana/](https://dwzhu-pku.github.io/PaperBanana/)

---

## Abstract
Despite rapid advances in autonomous AI scientists powered by language models, generating publication-ready illustrations remains a labor-intensive bottleneck in the research workflow. To lift this burden, we introduce **PaperBanana**, an agentic framework for automated generation of publication-ready academic illustrations. Powered by state-of-the-art VLMs and image generation models, PaperBanana orchestrates specialized agents to retrieve references, plan content and style, render images, and iteratively refine via self-critique. To rigorously evaluate our framework, we introduce **PaperBananaBench**, comprising 292 test cases for methodology diagrams curated from NeurIPS 2025 publications, covering diverse research domains and illustration styles. Comprehensive experiments demonstrate that PaperBanana consistently outperforms leading baselines in faithfulness, conciseness, readability, and aesthetics. We further show that our method effectively extends to the generation of high-quality statistical plots. Collectively, PaperBanana paves the way for the automated generation of publication-ready illustrations.

---

## 1. Introduction
Autonomous scientific discovery is a long-standing pursuit of artificial general intelligence. While LLMs have demonstrated potential in literature review, idea generation, and experiment iteration, current autonomous AI scientists struggle to visually communicate discoveries, especially for generating illustrations (diagrams and plots) that adhere to rigorous academic standards.

In this paper, we introduce **PaperBanana**, an agentic framework designed to bridge this gap. Given a methodology description and diagram caption as input, PaperBanana orchestrates specialized agents to retrieve reference examples, devise detailed plans, render images, and iteratively refine via self-critique.

---

## 2. Task Formulation
We formalize the task of automated academic illustration generation as learning a mapping from a source context $S$ and a communicative intent $C$ to a visual representation $I$:
$$I = f(S, C)$$
To further guide the mapping, the input can be augmented by a set of $N$ reference examples $\mathcal{E} = \{E_n\}_{n=1}^N$. The unified task formulation becomes:
$$I = f(S, C, \mathcal{E})$$

---

## 3. Methodology
PaperBanana orchestrates a collaborative team of five specialized agents:

1.  **Retriever Agent:** Identifies the $N$ most relevant examples $\mathcal{E}$ from a fixed reference set $\mathcal{R}$ using a generative retrieval approach where the VLM ranks candidates by matching research domain and diagram type.
2.  **Planner Agent:** Serves as the cognitive core. It translates unstructured data from $S$ into a comprehensive textual description $P$ of the target illustration.
3.  **Stylist Agent:** Acts as a design consultant. It uses an automatically synthesized **Aesthetic Guideline** $\mathcal{G}$ to refine the initial description into a stylistically optimized version $P^*$.
4.  **Visualizer Agent:** Renders the academic illustration using an image generation model based on description $P_t$. For statistical plots, it converts descriptions into executable Python Matplotlib code.
5.  **Critic Agent:** Forms a closed-loop refinement mechanism by identifying factual misalignments or visual glitches and providing feedback for the next iteration $P_{t+1}$.

---

## 4. Benchmark Construction
We introduce **PaperBananaBench**, curated from NeurIPS 2025 methodology diagrams.

### 4.1. Data Curation
*   **Collection & Parsing:** Sampled 2,000 papers from NeurIPS 2025, extracting methodology sections and diagrams using the MinerU toolkit.
*   **Filtering:** Restricted aspect ratios to [1.5, 2.5] and discarded papers without methodology diagrams, yielding 610 valid candidates.
*   **Categorization:** Diagrams are categorized into: *Agent & Reasoning*, *Vision & Perception*, *Generative & Learning*, and *Science & Applications*.
*   **Human Curation:** Final verification resulting in a test set ($N=292$) and a reference set ($N=292$).

### 4.2. Evaluation Protocol
We utilize **VLM-as-a-Judge** (Gemini-3-Pro) for referenced comparison against human-drawn diagrams across four dimensions:
*   **Content:** Faithfulness & Conciseness
*   **Presentation:** Readability & Aesthetics

---

## 5. Experiments

### 5.1. Baseline Methods
We compare PaperBanana against:
1.  **Vanilla:** Direct prompting of image generation models (GPT-Image-1.5, Nano-Banana-Pro).
2.  **Few-shot:** Prompting augmented with 10 examples.
3.  **Paper2Any:** An agentic framework for high-level idea presentation.

### 5.2. Main Results

**Table 1: Main results on PaperBananaBench (Best score in bold)**

| Method | Faithfulness ↑ | Conciseness ↑ | Readability ↑ | Aesthetic ↑ | Overall ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Vanilla Settings** | | | | | |
| GPT-Image-1.5 | 4.5 | 37.5 | 30.0 | 37.0 | 11.5 |
| Nano-Banana-Pro | 43.0 | 43.5 | 38.5 | 65.5 | 43.2 |
| Few-shot Nano-Banana-Pro | 41.6 | 49.6 | 37.6 | 60.5 | 41.8 |
| **Agentic Frameworks** | | | | | |
| Paper2Any (w/ Nano-Banana-Pro) | 6.5 | 44.0 | 20.5 | 40.0 | 8.5 |
| **PaperBanana (Ours)** | | | | | |
| w/ GPT-Image-1.5 | 16.0 | 65.0 | 33.0 | 56.0 | 19.0 |
| w/ Nano-Banana-Pro | **45.8** | **80.7** | **51.4** | **72.1** | **60.2** |
| *Human* | *50.0* | *50.0* | *50.0* | *50.0* | *50.0* |

### 5.3. Ablation Study

**Table 2: Ablation study on PaperBananaBench**

| # | Retriever | Planner | Stylist | Visualizer | Critic | Faithfulness ↑ | Conciseness ↑ | Readability ↑ | Aesthetic ↑ | Overall ↑ |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| ① | ✓ | ✓ | ✓ | ✓ | 3 iters | **45.8** | **80.7** | **51.4** | **72.1** | **60.2** |
| ② | ✓ | ✓ | ✓ | ✓ | 1 iter | 38.3 | 75.2 | 50.6 | 68.9 | 51.8 |
| ③ | ✓ | ✓ | ✓ | ✓ | - | 30.7 | 79.2 | 47.0 | 72.1 | 45.6 |
| ④ | ✓ | ✓ | - | ✓ | - | 39.2 | 61.7 | 47.9 | 67.4 | 49.2 |
| ⑤ | Random | ✓ | - | ✓ | - | 37.3 | 62.7 | 51.1 | 65.6 | 48.3 |
| ⑥ | - | ✓ | - | ✓ | - | 41.9 | 58.6 | 43.1 | 62.9 | 44.2 |

---

## 6. Discussion
*   **Enhancing Aesthetics of Human-Drawn Diagrams:** The summarized aesthetic guidelines $\mathcal{G}$ can be used to refine existing human-drawn figures. In experiments, refined diagrams achieved a 56.2% win rate against originals.
*   **Statistical Plots:** PaperBanana extends to statistical plots by converting descriptions into code. This approach outperforms direct image generation in terms of numerical faithfulness.

---

## 7. Limitations and Future Directions
*   **Raster vs. Vector:** Currently outputs raster images; future work could involve generating editable vector graphics (e.g., SVG, TikZ).
*   **Fine-Grained Faithfulness:** Challenges remain in precise connectivity (e.g., arrow start/end points).
*   **Style Diversity:** The unified style guide may reduce stylistic diversity.

---

## Appendices (Summary)

### Appendix E: Textual Description of Methodology Diagram
Contains the detailed prompt used to generate Figure 2, specifying layout, containers (Linear Planning Phase, Iterative Refinement Loop), agent icons (robots), and styling (soft blue/orange accents, LaTeX-style variables).

### Appendix F: Auto Summarized Style Guide
*   **"NeurIPS Look":** Soft Tech & Scientific Pastels.
*   **Color Palettes:** Use color to group logic. Backgrounds: Cream, Pale Blue, Mint.
*   **Shapes:** Rounded rectangles for processes, 3D stacks for tensors, cylinders for databases.
*   **Lines:** Orthogonal/Elbow for networks; Curved for logic/feedback.
*   **Typography:** Sans-serif (Arial/Roboto) for labels; Serif Italic (Times New Roman) for variables.

### Appendix G & H: System and Evaluation Prompts
Contains the exact system prompts for the Retriever, Planner, Stylist, Visualizer, and Critic agents, as well as the evaluation rubrics for the VLM-as-a-Judge.