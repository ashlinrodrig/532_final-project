# Formula 1 Pit Stop Analysis

This project analyzes Formula 1 pit stop data to predict the optimal tire change window using various machine learning models. 
This README outlines the steps required to set up the environment, install dependencies, and download data from Kaggle.

## Pre-requisites
- Python 3.x
- Kaggle account

## Project Setup
- **Create a virtual environment**
    - Navigate to your project directory - `cd /path/to/your/project`
    - Create a virtual environment - `python3 -m venv ~/kaggle-env`
    - Activate the virtual environment - `source ~/kaggle-env/bin/activate`

- **Set Up Kaggle API**
    - Sign in to your Kaggle account. Go to Account settings.
    - Under the API section, click Create New API Token. This will download a file named `kaggle.json`.
    - Move `kaggle.json` to the `.kaggle directory` in your home folder.
      - `mkdir -p ~/.kaggle`
      - `mv ~/Downloads/kaggle.json ~/.kaggle/`
      - `chmod 600 ~/.kaggle/kaggle.json`
    - Install the Kaggle API - `pip install kaggle`
    
- **Download Dataset from Kaggle**
    - Download Dataset from Kaggle
        - `kaggle datasets download cjgdev/formula-1-race-data-19502017`
        - `kaggle datasets download jtrotman/formula-1-race-events/50`




