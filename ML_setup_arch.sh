#!/bin/bash

# ========================================================
# Machine Learning Engineering Environment Setup Script
# For Arch Linux with Anaconda
# ========================================================

# Set script to exit on error
set -e

# Terminal colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ========================================================
# Helper functions
# ========================================================

# Log messages with timestamp
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Display section headers
section() {
    echo -e "\n${GREEN}=== $1 ===${NC}"
}

# Display errors
error() {
    echo -e "${RED}ERROR: $1${NC}"
    exit 1
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Create experiment logging function
setup_experiment_logger() {
    local log_dir="$HOME/ml_experiments"
    mkdir -p "$log_dir"
    
    # Create the experiment logger script
    cat > "$HOME/bin/mllog" << 'EOL'
#!/bin/bash

# ML Experiment Logger
# Usage: mllog experiment_name [description]

set -e

if [ $# -lt 1 ]; then
    echo "Usage: mllog experiment_name [description]"
    exit 1
fi

LOG_DIR="$HOME/ml_experiments"
EXPERIMENT="$1"
DESCRIPTION="${2:-No description provided}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_DIR="$LOG_DIR/${EXPERIMENT}_${TIMESTAMP}"

# Create experiment directory
mkdir -p "$EXPERIMENT_DIR"
mkdir -p "$EXPERIMENT_DIR/code"
mkdir -p "$EXPERIMENT_DIR/data"
mkdir -p "$EXPERIMENT_DIR/results"
mkdir -p "$EXPERIMENT_DIR/models"

# Create experiment metadata
cat > "$EXPERIMENT_DIR/metadata.json" << EOF
{
    "name": "$EXPERIMENT",
    "timestamp": "$(date +"%Y-%m-%d %H:%M:%S")",
    "description": "$DESCRIPTION",
    "environment": {
        "python_version": "$(python --version 2>&1)",
        "conda_env": "$CONDA_DEFAULT_ENV"
    }
}
EOF

# Copy current directory Python files to experiment
cp *.py "$EXPERIMENT_DIR/code/" 2>/dev/null || echo "No Python files found in current directory"

# Create experiment notebook
cat > "$EXPERIMENT_DIR/experiment_notebook.md" << EOF
# Experiment: $EXPERIMENT
**Date:** $(date +"%Y-%m-%d %H:%M:%S")
**Description:** $DESCRIPTION

## Goals

- [ ] Goal 1
- [ ] Goal 2

## Methods

## Results

## Conclusions

## Next Steps

EOF

echo "Experiment '$EXPERIMENT' initialized at $EXPERIMENT_DIR"
echo "Use 'cd $EXPERIMENT_DIR' to navigate to experiment directory"
EOL

    # Make the script executable
    chmod +x "$HOME/bin/mllog"
    
    log "Experiment logger created at $HOME/bin/mllog"
    log "Usage: mllog experiment_name [description]"
}

# ========================================================
# System package installation
# ========================================================
section "Installing system packages"

# Update system
log "Updating system packages..."
sudo pacman -Syu --noconfirm || error "Failed to update system packages"

# Install essential development tools
log "Installing development tools..."
sudo pacman -S --needed --noconfirm \
    base-devel \
    git \
    cmake \
    vim \
    neovim \
    tmux \
    htop \
    wget \
    curl \
    unzip \
    p7zip \
    openssh \
    python-pip \
    python-virtualenv \
    python-setuptools \
    python-wheel || error "Failed to install development tools"

# Install scientific computing dependencies
log "Installing scientific computing dependencies..."
sudo pacman -S --needed --noconfirm \
    openblas \
    lapack \
    hdf5 \
    graphviz \
    tkinter \
    ffmpeg \
    libxml2 \
    libxslt \
    freetype2 || error "Failed to install scientific computing dependencies"

# Install CUDA if an NVIDIA GPU is present
if lspci | grep -i nvidia > /dev/null; then
    log "NVIDIA GPU detected, installing CUDA and related packages..."
    sudo pacman -S --needed --noconfirm \
        cuda \
        cudnn \
        nvidia \
        nvidia-utils || error "Failed to install CUDA packages"
    
    # Set up environment variables for CUDA
    if ! grep -q 'CUDA_HOME' ~/.bashrc; then
        echo 'export CUDA_HOME=/opt/cuda' >> ~/.bashrc
        echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    fi
    log "CUDA environment variables added to ~/.bashrc"
else
    log "No NVIDIA GPU detected, skipping CUDA installation"
fi

# ========================================================
# Anaconda installation
# ========================================================
section "Installing Anaconda"

if command_exists conda; then
    log "Anaconda is already installed"
else
    log "Downloading Anaconda installer..."
    ANACONDA_INSTALLER="Anaconda3-2023.09-0-Linux-x86_64.sh"
    wget https://repo.anaconda.com/archive/$ANACONDA_INSTALLER -O /tmp/$ANACONDA_INSTALLER || error "Failed to download Anaconda"
    
    log "Installing Anaconda..."
    bash /tmp/$ANACONDA_INSTALLER -b -p $HOME/anaconda3 || error "Failed to install Anaconda"
    
    # Initialize conda for bash
    $HOME/anaconda3/bin/conda init bash || error "Failed to initialize conda for bash"
    
    # Clean up installer
    rm /tmp/$ANACONDA_INSTALLER
    
    log "Anaconda installed successfully at $HOME/anaconda3"
    
    # Source bashrc to use conda immediately
    source ~/.bashrc
fi

# ========================================================
# Creating ML environment with conda
# ========================================================
section "Setting up ML environment with conda"

# Create conda environment for ML
ENV_NAME="ml-engineering"

if conda info --envs | grep -q $ENV_NAME; then
    log "Conda environment '$ENV_NAME' already exists"
else
    log "Creating conda environment '$ENV_NAME'..."
    conda create -y -n $ENV_NAME python=3.10 || error "Failed to create conda environment"
fi

# Activate the environment
log "Activating conda environment '$ENV_NAME'..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME || error "Failed to activate conda environment"

# Install ML and scientific computing packages
section "Installing ML and scientific packages"

log "Installing basic scientific computing packages..."
conda install -y -c conda-forge \
    numpy \
    scipy \
    pandas \
    matplotlib \
    seaborn \
    jupyterlab \
    ipywidgets \
    scikit-learn \
    sympy \
    statsmodels || error "Failed to install basic scientific packages"

log "Installing deep learning packages..."
if lspci | grep -i nvidia > /dev/null; then
    log "Installing PyTorch with CUDA support..."
    conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia || error "Failed to install PyTorch with CUDA"
else
    log "Installing PyTorch without CUDA support..."
    conda install -y pytorch torchvision torchaudio cpuonly -c pytorch || error "Failed to install PyTorch"
fi

log "Installing TensorFlow..."
if lspci | grep -i nvidia > /dev/null; then
    pip install tensorflow || error "Failed to install TensorFlow"
else
    pip install tensorflow-cpu || error "Failed to install TensorFlow CPU"
fi

log "Installing additional ML tools and libraries..."
pip install \
    transformers \
    datasets \
    huggingface-hub \
    optuna \
    wandb \
    ray[tune] \
    eli5 \
    shap \
    xgboost \
    lightgbm \
    catboost \
    mlflow \
    fastapi \
    uvicorn \
    streamlit \
    gradio \
    dvc \
    black \
    isort \
    pylint \
    mypy \
    pytest \
    pytest-cov \
    jupytext \
    jupyter-black \
    jupyter_contrib_nbextensions || error "Failed to install additional ML tools"

# Install Jupyter Lab extensions
log "Installing Jupyter Lab extensions..."
pip install jupyterlab-git || log "Warning: Failed to install jupyterlab-git"
jupyter labextension install jupyterlab-plotly || log "Warning: Failed to install jupyterlab-plotly"

# ========================================================
# Setting up experiment logging and project structure
# ========================================================
section "Setting up experiment logging and project structure"

# Create bin directory for custom scripts
mkdir -p "$HOME/bin"
if ! echo $PATH | grep -q "$HOME/bin"; then
    echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
fi

# Create the experiment logger
setup_experiment_logger

# Create ML projects directory
mkdir -p "$HOME/ml_projects"

# Create a template for new ML projects
log "Creating ML project template..."
mkdir -p "$HOME/ml_templates/basic_ml_project"
cat > "$HOME/ml_templates/basic_ml_project/README.md" << 'EOL'
# Project Name

## Description
Brief description of the project.

## Setup
```bash
conda activate ml-engineering
pip install -r requirements.txt
```

## Directory Structure
- `data/`: Raw and processed data
- `notebooks/`: Jupyter notebooks for exploration and experimentation
- `src/`: Source code for the project
  - `data/`: Data loading and processing scripts
  - `models/`: Model definitions and training code
  - `utils/`: Utility functions
- `tests/`: Unit tests
- `results/`: Saved results, plots, etc.
- `models/`: Saved model checkpoints

## Usage
Instructions for running the code.
EOL

# Create project structure
mkdir -p "$HOME/ml_templates/basic_ml_project/data/raw"
mkdir -p "$HOME/ml_templates/basic_ml_project/data/processed"
mkdir -p "$HOME/ml_templates/basic_ml_project/notebooks"
mkdir -p "$HOME/ml_templates/basic_ml_project/src/data"
mkdir -p "$HOME/ml_templates/basic_ml_project/src/models"
mkdir -p "$HOME/ml_templates/basic_ml_project/src/utils"
mkdir -p "$HOME/ml_templates/basic_ml_project/tests"
mkdir -p "$HOME/ml_templates/basic_ml_project/results"
mkdir -p "$HOME/ml_templates/basic_ml_project/models"

# Create template Python files
cat > "$HOME/ml_templates/basic_ml_project/src/data/data_loader.py" << 'EOL'
#!/usr/bin/env python3
"""
Data loading utilities.
"""
import pandas as pd
from pathlib import Path


def load_data(data_path):
    """
    Load data from the specified path.
    
    Parameters:
    -----------
    data_path : str or Path
        Path to the data file
        
    Returns:
    --------
    DataFrame or dict
        Loaded data
    """
    data_path = Path(data_path)
    
    if data_path.suffix == '.csv':
        return pd.read_csv(data_path)
    elif data_path.suffix == '.parquet':
        return pd.read_parquet(data_path)
    elif data_path.suffix == '.json':
        return pd.read_json(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")


def preprocess_data(df):
    """
    Preprocess the data.
    
    Parameters:
    -----------
    df : DataFrame
        Raw data
        
    Returns:
    --------
    DataFrame
        Preprocessed data
    """
    # Implement preprocessing steps
    # Example: Handle missing values
    df = df.copy()
    df = df.dropna()
    
    return df
EOL

cat > "$HOME/ml_templates/basic_ml_project/src/models/model.py" << 'EOL'
#!/usr/bin/env python3
"""
Model definition and training utilities.
"""
import pickle
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class MLModel:
    """
    Wrapper class for machine learning models.
    """
    
    def __init__(self, model=None):
        """
        Initialize the model.
        
        Parameters:
        -----------
        model : BaseEstimator, optional
            Scikit-learn compatible model
        """
        self.model = model
        
    def train(self, X_train, y_train, **kwargs):
        """
        Train the model.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        **kwargs
            Additional arguments to pass to the model's fit method
            
        Returns:
        --------
        self
        """
        self.model.fit(X_train, y_train, **kwargs)
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters:
        -----------
        X : array-like
            Features
            
        Returns:
        --------
        array-like
            Predictions
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model.
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        return metrics
    
    def save(self, filepath):
        """
        Save the model to disk.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to save the model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    @classmethod
    def load(cls, filepath):
        """
        Load the model from disk.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the saved model
            
        Returns:
        --------
        MLModel
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        return cls(model)
EOL

cat > "$HOME/ml_templates/basic_ml_project/src/utils/visualization.py" << 'EOL'
#!/usr/bin/env python3
"""
Visualization utilities.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 8), cmap='Blues'):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    classes : list, optional
        Class names
    figsize : tuple, optional
        Figure size
    cmap : str, optional
        Colormap
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax)
    
    if classes is not None:
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    return fig


def plot_feature_importance(feature_importance, feature_names, figsize=(12, 8)):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    feature_importance : array-like
        Feature importance scores
    feature_names : list
        Feature names
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Sort features by importance
    indices = np.argsort(feature_importance)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.barh(range(len(indices)), feature_importance[indices], align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    
    return fig


def plot_learning_curve(train_scores, test_scores, train_sizes, figsize=(10, 6)):
    """
    Plot learning curve.
    
    Parameters:
    -----------
    train_scores : array-like
        Training scores
    test_scores : array-like
        Test scores
    train_sizes : array-like
        Training set sizes
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    
    ax.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    
    ax.set_xlabel('Training examples')
    ax.set_ylabel('Score')
    ax.set_title('Learning Curve')
    ax.legend(loc='best')
    
    return fig
EOL

# Create a script to create new ML projects from the template
cat > "$HOME/bin/new-ml-project" << 'EOL'
#!/bin/bash
# Script to create a new ML project from template

if [ $# -lt 1 ]; then
    echo "Usage: new-ml-project <project_name> [description]"
    exit 1
fi

PROJECT_NAME="$1"
DESCRIPTION="${2:-No description provided}"
TEMPLATE_DIR="$HOME/ml_templates/basic_ml_project"
PROJECT_DIR="$HOME/ml_projects/$PROJECT_NAME"

# Check if project already exists
if [ -d "$PROJECT_DIR" ]; then
    echo "Error: Project '$PROJECT_NAME' already exists at $PROJECT_DIR"
    exit 1
fi

# Copy template to new project directory
cp -r "$TEMPLATE_DIR" "$PROJECT_DIR"

# Update README with project name and description
sed -i "s/# Project Name/# $PROJECT_NAME/" "$PROJECT_DIR/README.md"
sed -i "s/Brief description of the project./$DESCRIPTION/" "$PROJECT_DIR/README.md"

# Create requirements.txt file
cat > "$PROJECT_DIR/requirements.txt" << EOF
# Data processing
numpy
pandas
scikit-learn

# Visualization
matplotlib
seaborn

# Development
pytest
black
isort
pylint
mypy

# Add project-specific dependencies below
EOF

# Create initial notebook
cat > "$PROJECT_DIR/notebooks/01_exploration.ipynb" << EOF
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# $PROJECT_NAME: Initial Data Exploration\n",
    "\n",
    "**Description:** $DESCRIPTION\n",
    "\n",
    "**Date:** $(date +"%Y-%m-%d")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import libraries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Set plot style\n",
    "plt.style.use('seaborn-v0_8-whitegrid')\n",
    "sns.set_context('notebook')\n",
    "\n",
    "# Display settings\n",
    "%matplotlib inline\n",
    "pd.set_option('display.max_columns', None)\n",
    "pd.set_option('display.max_rows', 100)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Add code to load your data here\n",
    "# data = pd.read_csv('../data/raw/your_data.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Data Overview"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Examine data structure\n",
    "# data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# View basic statistics\n",
    "# data.describe(include='all')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Exploratory Data Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Add visualizations here"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Next Steps"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- [ ] Task 1\n",
    "- [ ] Task 2\n",
    "- [ ] Task 3"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Create Git repository
cd "$PROJECT_DIR"
git init
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Jupyter
.ipynb_checkpoints
*/.ipynb_checkpoints/*

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Models
models/*
!models/.gitkeep

# Distribution / packaging
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Testing
.coverage
.pytest_cache/
htmlcov/

# OS specific
.DS_Store
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
EOF

# Add placeholders to keep empty directories in git
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch models/.gitkeep
touch results/.gitkeep

# Initial commit
git add .
git commit -m "Initial project structure"

echo "Project '$PROJECT_NAME' created at $PROJECT_DIR"
echo "To get started:"
echo "  cd $PROJECT_DIR"
echo "  conda activate ml-engineering"
echo "  jupyter lab notebooks/01_exploration.ipynb"
EOL

# Make the script executable
chmod +x "$HOME/bin/new-ml-project"

# ========================================================
# Create aliases and configurations
# ========================================================
section "Creating aliases and configurations"

# Create aliases for common ML tasks
cat >> ~/.bashrc << 'EOL'

# ML Engineering aliases and functions
alias mla='conda activate ml-engineering'
alias mld='conda deactivate'
alias jl='jupyter lab'
alias mlnew='new-ml-project'

# Function to quickly create and activate a virtual environment
mlvenv() {
    if [ $# -lt 1 ]; then
        echo "Usage: mlvenv <env_name> [python_version]"
        return 1
    fi
    
    ENV_NAME="$1"
    PYTHON_VERSION="${2:-3.10}"
    
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
    conda activate "$ENV_NAME"
    echo "Environment '$ENV_NAME' created and activated with Python $PYTHON_VERSION"
}

# Function to create a new experiment
mlexp() {
    if [ $# -lt 1 ]; then
        echo "Usage: mlexp <experiment_name> [description]"
        return 1
    fi
    
    mllog "$@"
}

# Function to start tracking with MLflow
mltrack() {
    if [ ! -d "mlruns" ]; then
        mlflow ui --host 127.0.0.1 --port 5000 &
        echo "MLflow tracking server started at http://127.0.0.1:5000"
    else
        echo "MLflow tracking already running"
    fi
}
EOL

# Configure git for ML projects
git config --global diff.tool.jupyternotebook.command "jupyter nbdiff-web"
git config --global core.excludesfile ~/.gitignore_global

# Create global gitignore
cat > ~/.gitignore_global << 'EOL'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Jupyter
.ipynb_checkpoints
*/.ipynb_checkpoints/*

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS specific
.DS_Store
Thumbs.db
EOL

# ========================================================
# Final setup and verification
# ========================================================
section "Verifying installation"

# Print summary of installed tools
echo -e "${YELLOW}=== ML Engineering Environment Summary ===${NC}"
echo "System packages: Installed"
echo "Anaconda: Installed"
echo "ML Environment: ml-engineering (Python 3.10)"
echo "Deep Learning Frameworks: PyTorch, TensorFlow"
echo "Experiment Logger: $HOME/bin/mllog"
echo "Project Template: $HOME/ml_templates/basic_ml_project"
echo "Project Creator: $HOME/bin/new-ml-project"
echo "Aliases and Functions: Added to ~/.bashrc"

# Final instructions
section "Final instructions"

echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "To get started:"
echo "  1. Close and reopen your terminal, or run: source ~/.bashrc"
echo "  2. Activate the ML environment: mla"
echo "  3. Create a new ML project: mlnew my_first_project \"Description of project\""
echo "  4. Create a new experiment: mlexp experiment_name \"Description of experiment\""
echo ""
echo "For help, use the following commands:"
echo "  - new-ml-project --help: Help with creating new projects"
echo "  - mllog --help: Help with logging experiments"
echo ""
echo "Happy Machine Learning!"
