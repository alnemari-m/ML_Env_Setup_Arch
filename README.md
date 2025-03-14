# Machine Learning Engineering Environment Setup for Arch Linux

This comprehensive guide outlines the complete machine learning engineering environment setup on Arch Linux, detailing each component and its purpose in creating a robust workflow for ML development.

## System Setup and Dependencies

### System Packages Installation
The environment begins with essential development tools and scientific computing dependencies installed through Arch Linux's package manager (pacman). These foundational packages provide the necessary building blocks for advanced computational work.

### CUDA Installation
The setup intelligently detects the presence of an NVIDIA GPU and automatically installs CUDA and related packages to enable GPU acceleration. This optimization dramatically speeds up training for deep learning models by leveraging specialized hardware.

### Anaconda Installation
If not already present, the script downloads and installs Anaconda, which serves as a robust platform for scientific computing and machine learning. Anaconda's package management capabilities ensure compatibility between libraries and simplify environment management.

## ML Environment Configuration

### Conda Environment Creation
A dedicated environment called "ml-engineering" is created with Python 3.10, providing isolation for ML-specific packages and preventing conflicts with other Python applications on the system.

### Scientific Computing Packages
The environment includes fundamental libraries that form the backbone of scientific computing:
- NumPy for numerical operations
- SciPy for advanced mathematical functions
- Pandas for data manipulation and analysis
- Matplotlib and Seaborn for data visualization

### Machine Learning Libraries
Both PyTorch and TensorFlow are installed (with GPU support where available), along with a comprehensive suite of ML libraries:

- **Deep learning frameworks**: PyTorch, TensorFlow
- **ML tools**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Experiment tracking**: MLflow, Weights & Biases
- **Hyperparameter optimization**: Optuna, Ray[tune]
- **Model explainability**: SHAP, ELI5
- **Development tools**: Black, pylint, pytest

## Experiment Logging System

### Experiment Logger
A custom tool called `mllog` is created to structure and document ML experiments, ensuring reproducibility and organization:

- Automatically creates a timestamped directory for each experiment
- Saves metadata (experiment name, description, environment details)
- Creates a standardized directory structure (code, data, results, models)
- Generates a Markdown notebook for documenting the experiment process, goals, and outcomes
- Automatically copies Python files from the current directory to the experiment folder
- Facilitates tracking changes and progress across multiple experiments

## Project Structure and Templates

### ML Project Template
A standardized template for new ML projects is established with:

- A well-organized directory structure (data, notebooks, src, tests, results, models)
- Pre-configured Python modules for common ML tasks:
  - `data_loader.py`: Functions to load and preprocess data
  - `model.py`: Class that wraps ML models with training, evaluation, and saving functionality
  - `visualization.py`: Functions for creating common ML visualizations (confusion matrices, feature importance, learning curves)
- A starter Jupyter notebook for data exploration
- Git integration with appropriate `.gitignore` settings for ML projects

### Project Creator Script
The environment includes `new-ml-project`, a command-line tool designed to:

- Generate new projects based on the template
- Customize the README with project name and description
- Set up an initial requirements.txt file
- Initialize a Git repository with appropriate settings

## Usability Enhancements

### Aliases and Helper Functions
Several shortcuts are created to streamline the workflow:

- `mla`: Quickly activate the ML environment
- `mld`: Deactivate the current conda environment
- `jl`: Launch Jupyter Lab
- `mlnew`: Create a new ML project
- `mlvenv`: Create and activate a new conda environment
- `mlexp`: Start a new experiment with logging
- `mltrack`: Launch an MLflow tracking server for experiment tracking

### Git Configuration
Git is configured with specialized settings for ML work:

- Special handling for Jupyter notebooks
- Global `.gitignore` with appropriate settings for ML work

## How to Use the Environment

After running the setup script and restarting your terminal (or sourcing your bashrc), you can:

### Start a New Project
```bash
mla                           # Activate the ML environment
mlnew my_regression_project   # Create a new project
cd ~/ml_projects/my_regression_project
jupyter lab                   # Start working in Jupyter
```

### Log Your Experiments
```bash
mlexp initial_model "Testing a random forest baseline"
```
This creates a structured directory at `~/ml_experiments/<experiment_name>_<timestamp>` with everything needed to document your work.

### Use the Template Code
The template includes reusable code for common ML tasks, allowing you to focus on your specific problem rather than rewriting boilerplate code.

## Benefits of This Setup

- **Reproducibility**: All experiments are automatically documented with their environment details
- **Organization**: Consistent project structure makes it easier to navigate your work
- **Efficiency**: Templates and helper functions reduce the time spent on setup
- **Best Practices**: Built-in Git integration and logging encourage good habits
- **Flexibility**: Works for both deep learning and traditional ML tasks
- **GPU Support**: Automatically configures for GPU acceleration when available

This environment helps you write better machine learning code and maintain a structured approach to your experiments, making your workflow more efficient and your results more reproducible.
