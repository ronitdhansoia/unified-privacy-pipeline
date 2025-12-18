# Unified Privacy Pipeline - Tutorials

This directory contains Jupyter notebooks that demonstrate the capabilities of the Unified Privacy Pipeline.

## Available Notebooks

### 1. Getting Started
**File**: `01_getting_started.ipynb`

Introduction to the core concepts and basic usage:
- Setting up the environment
- Training a simple model
- Basic machine unlearning
- Influence function computation
- Privacy evaluation with MIA

**Recommended for**: Beginners, first-time users

### Future Notebooks (Templates Available)

2. **Advanced Unlearning Techniques** - Different unlearning methods and their trade-offs
3. **Differential Privacy Training** - DP-augmented training workflows
4. **Influence Functions Deep Dive** - Advanced influence analysis
5. **Face Recognition Example** - Privacy-preserving face recognition
6. **Health Prediction Example** - HIPAA-compliant healthcare AI

## Running the Notebooks

### Prerequisites
```bash
cd Unified-Privacy-Pipeline
source venv/bin/activate  # Activate virtual environment
pip install jupyter ipywidgets matplotlib
```

### Launch Jupyter
```bash
jupyter notebook notebooks/
```

### Running in Google Colab

Upload the notebook to Google Colab and add this cell at the beginning:

```python
# Clone repository
!git clone https://github.com/your-org/Unified-Privacy-Pipeline.git
%cd Unified-Privacy-Pipeline

# Install dependencies
!pip install -r requirements.txt
```

## Notebook Structure

Each notebook follows this structure:

1. **Introduction** - Overview and objectives
2. **Setup** - Imports and environment configuration
3. **Examples** - Step-by-step code examples with explanations
4. **Exercises** - Optional practice problems
5. **Summary** - Key takeaways
6. **Next Steps** - Links to related tutorials

## Tips for Learning

- **Run cells sequentially** - Each notebook is designed to be run from top to bottom
- **Experiment** - Try modifying parameters and see how results change
- **Read comments** - Code is heavily commented to explain concepts
- **Ask questions** - Open issues on GitHub if you get stuck

## Example Usage

```python
# Quick start example
from influence_functions.influence_computation import create_influence_computer
from machine_unlearning.unlearning_methods import create_unlearner

# Create components
influence_computer = create_influence_computer(method="lissa")
unlearner = create_unlearner("gradient_ascent")

# Use in your workflow
influences = influence_computer.compute_influence_lissa(...)
results = unlearner.unlearn(model, forget_loader, retain_loader)
```

## Contributing

Have ideas for new tutorials? See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## Support

- **Documentation**: `docs/` directory
- **Issues**: GitHub issue tracker
- **Discussions**: GitHub Discussions

Happy learning! 
