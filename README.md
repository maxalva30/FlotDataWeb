# Flotation Data Analysis Assistant

##  Quick Start

### Prerequisites
- Python 3.11 or 3.12
- pip

### Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd FlotDataWeb

# 2. Create virtual environment
python -m venv venv_flotdata

# 3. Activate virtual environment
venv_flotdata\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run application
python index.py
```

Open browser: **http://localhost:8050**

## Project Structure

```
FlotDataWeb/
├── index.py                 # Main application
├── requirements.txt         # Dependencies
├── assets/
│   ├── custom.css          # Styles
│   └── *.jpg, *.png        # Images
└── pages/
    ├── home.py             # Home page (data upload & config)
    └── plots.py            # Analysis & visualization
```