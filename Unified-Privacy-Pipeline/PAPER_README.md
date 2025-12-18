# Research Paper - PRVCYPPLN

## Compiling the LaTeX Document

### Requirements

You need a LaTeX distribution installed:
- **macOS**: Install MacTeX - `brew install --cask mactex`
- **Linux**: Install TeX Live - `sudo apt-get install texlive-full`
- **Windows**: Install MiKTeX from https://miktex.org/

### Compile the Paper

#### Method 1: Using pdflatex (Command Line)

```bash
cd "/Users/ronitdhansoia/Desktop/Generative Artificial Intelligence/Unified-Privacy-Pipeline"

# Compile (run twice for references)
pdflatex research_paper.tex
pdflatex research_paper.tex

# Or compile with bibliography
pdflatex research_paper.tex
bibtex research_paper
pdflatex research_paper.tex
pdflatex research_paper.tex
```

#### Method 2: Using Overleaf (Online)

1. Go to https://www.overleaf.com/
2. Create new project → Upload Project
3. Upload `research_paper.tex`
4. Click "Recompile"
5. Download PDF

#### Method 3: Using VS Code with LaTeX Workshop

1. Install VS Code extension: "LaTeX Workshop"
2. Open `research_paper.tex`
3. Press `Cmd+Alt+B` (macOS) or `Ctrl+Alt+B` (Windows/Linux)
4. PDF will generate automatically

### Output

The compilation will generate:
- `research_paper.pdf` - The final paper
- `research_paper.aux`, `.log`, `.out` - Auxiliary files (can be deleted)

## Paper Structure

The paper includes:

### Sections
1. **Abstract** - 200 words summary
2. **Introduction** - Motivation and contributions
3. **Related Work** - DP, unlearning, influence functions, privacy attacks
4. **System Architecture** - Three-layer design
5. **Implementation** - Technology stack, models, data pipeline
6. **Experimental Evaluation** - Results and analysis
7. **Discussion** - Findings, limitations, future work
8. **Regulatory Compliance** - GDPR and HIPAA
9. **Reproducibility** - Code and data availability
10. **Conclusion** - Summary and impact
11. **References** - 15 citations

### Key Results Documented

- **Face Recognition**: 54.3% test accuracy (158 classes)
- **Health Prediction**: 92.2% test accuracy
- **Unlearning**: 93% forget quality, 4× faster than retraining
- **Privacy Protection**: 86.4% against MIA attacks
- **Privacy Budget**: ε=1.0, δ=10^{-5}

### Tables Included

- Table I: Face Recognition Performance
- Table II: Health Prediction Results
- Table III: Unlearning Method Comparison
- Table IV: Privacy Attack Resistance

## Customization

### Change Authors

Edit lines 19-31 in `research_paper.tex`:

```latex
\IEEEauthorblockN{Your Name}
\IEEEauthorblockA{\textit{Your Department}\\
Email: your.email@example.edu}
```

### Add Your Institution

Update the affiliation blocks:

```latex
\IEEEauthorblockA{\textit{Your University}\\
\textit{Department Name}\\
City, Country\\
Email: name@university.edu}
```

### Add Figures

Place figures in the same directory and add:

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.48\textwidth]{figure_name.pdf}
\caption{Your caption here}
\label{fig:label}
\end{figure}
```

### Add More References

Add to the bibliography section:

```latex
\bibitem{author2023}
A. Author, ``Title of paper,'' in \textit{Conference Name}, 2023, pp. 1--10.
```

## Paper Format

- **Template**: IEEE Conference format
- **Columns**: Two-column layout
- **Page Limit**: ~10 pages
- **Font**: 10pt Times Roman
- **Paper Size**: US Letter (8.5" × 11")

## Submission Ready

The paper is formatted for submission to:
- IEEE conferences (CVPR, ICCV, NeurIPS, ICML)
- ACM conferences (KDD, SIGMOD, CCS)
- Privacy conferences (USENIX Security, S&P, PETS)

## Citations

The paper includes 15 key references:
- Differential Privacy (Dwork, Abadi, Opacus)
- Machine Unlearning (Bourtoule, Cao, Golatkar)
- Influence Functions (Koh & Liang)
- Privacy Attacks (Shokri, Tramèr)
- Regulations (GDPR, HIPAA)

## License

This research paper is provided for academic and educational purposes.

---

**For questions about the paper, contact:**
- Ronit Dhansoia: ronit.dhansoia@example.edu
- Zohaib Hussain: zohaib.hussain@example.edu
- Edwin Roy Cheriyan: edwin.cheriyan@example.edu
