# Formula 1 Pit Stop Analysis

This project analyzes Formula 1 pit stop data to predict the optimal tire change window using various machine learning models. 
This README outlines the steps required to set up the environment, install dependencies, and download data from Kaggle.

## Pre-requisites
- Python 3.x
- Kaggle account

## Project Setup
- **Create a virtual environment**
    - Navigate to your project directory - `cd 532_final-project`
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
    - Use the following commands to download the datasets from Kaggle:
        - `kaggle datasets download cjgdev/formula-1-race-data-19502017`
        - `kaggle datasets download jtrotman/formula-1-race-events/50`
    - After downloading, unzip the downloaded files to access the data.

- As the dataset is huge, training the Flan-T5 model takes a lot of time, so, I have added the training data csv file, 80% of which has been used to train the model and have added the zip file of the model here - https://drive.google.com/file/d/1mZARX-LWDJv0jeNX2CkufnUhJraMzTAT/view?usp=sharing. Please use your UMass email to access the model.


## Instructions to Run the LSTM Model

- Open the Notebook: Navigate to the LSTM_532.ipynb notebook located in this repository.

- Download the Dataset: Download the f1dataset.csv file from the link provided in the repository.

- Update Dataset Path: In the notebook, locate the section where the dataset path is specified and update it to the location where you have saved the f1dataset.csv file.

- Run the Notebook: After setting the correct dataset path, run all the cells in the notebook to begin training and evaluating the model.
  
## Results:

Flan T5 <br>
Memory Usage: <br>
<img width="946" alt="image" src="https://github.com/user-attachments/assets/ef5554e1-03d3-4772-9a0c-52b71af862c4">

<br>
LSTM: <br>
Latency: <br>
<img width="816" alt="image" src="https://github.com/user-attachments/assets/140590cf-cc3f-4bfa-9cc0-5517b67ae73c">
<br>
Throughput: <br>
<img width="894" alt="image" src="https://github.com/user-attachments/assets/76e5c74c-61c7-452c-9835-b6611d2a6263">

## Installation:
git clone https://github.com/username/project-name.git  
cd 532_final-project

## Usage:

- **Run preprocessing:**
  python preprocess_data.py  

- **Run Flan-T5 model:**
  python flan_t5_model.py

- **Run LSTM:**


