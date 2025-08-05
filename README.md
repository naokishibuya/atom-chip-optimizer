# Atom Chip Optimizer

Credit: This project is inspired by the work of  the University of Sussex's BEC lab's MATLAB scripts.

## Installation

### 1. Clone the repository:

   ```bash
   git clone https://github.com/naokishibuya/atom-chip-optimizer.git
   cd atom-chip-optimizer
   ```

### 2. Set up a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # .venv\Scripts\activate   # On Windows
   ```

### 3. Install dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### 4. Install pre-commit hooks (Optional):

   ```bash
   pip install -r requirements-dev.txt  # Ruff, PyTest, etc.
   pre-commit install   # Install pre-commit hooks
   ```
