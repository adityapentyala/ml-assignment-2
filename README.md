# ML ASSIGNMENT 2 - Patient Emergency Risk Analytics

# USAGE INSTRUCTIONS

- Clone this repository onto your local system

- Download the dataset from the provided google drive link as part of the assignment and place it in the root directory of your clone

- Create a new virtual environment with the command `python -m venv env`

- Activate the environment: 
    - `source env/bin/activate` on Mac/Linux
    - `env/Scripts/activate.ps1` on Windows Powershell
    - `env/Scripts/activate.bat` on Windows cmd

- Once inside the environment, install the requirements using `pip install -r requirements.txt`

- Run the notebook _eda.ipynb_ to populate the _data/_ directory (you may have to create an empty directory at first)

- Run the command `streamlit run dashboard.py`


## Target
HIGH_EMERGENCY_RISK: >3 emergency room encounters logged

## Features
- Age
- Medications
- Conditions
- Allergies
- Income
- Healthcare Coverage
- Procedure Cost

## Models
- Decision Tree
- Neural Network (MLP)
- Support Vector Machine

## Metrics
- Precision
- Accuracy
- Recall
- F1