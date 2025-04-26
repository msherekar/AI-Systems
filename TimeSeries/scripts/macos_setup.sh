#!/bin/bash
# macOS setup script for TimeSeries project - Compatible with Apple Silicon (M1/M2)

echo "Setting up TimeSeries project for macOS with Apple Silicon..."

# Check if Homebrew is installed
if ! command -v brew &> /dev/null
then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "Homebrew is already installed."
    brew update
fi

# Install OpenMP and other dependencies needed for scientific computing
echo "Installing system dependencies..."
brew install libomp cmake openblas

# Check if Python 3.9+ is installed
if ! command -v python3 &> /dev/null || [[ $(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2) < "3.9" ]]
then
    echo "Installing Python 3.9..."
    brew install python@3.9
    echo 'export PATH="/usr/local/opt/python@3.9/bin:$PATH"' >> ~/.zshrc
    source ~/.zshrc
else
    echo "Python 3.9+ is already installed."
fi

# Create a virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch specially for Apple Silicon
echo "Installing PyTorch for Apple Silicon..."
pip install --no-binary=torch torch torchvision torchaudio

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

# Create directories if they don't exist
mkdir -p data/raw data/processed models logs

echo "Setup complete! Activate the virtual environment with:"
echo "source venv/bin/activate" 