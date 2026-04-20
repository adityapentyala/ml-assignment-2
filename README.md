# ML ASSIGNMENT 2 - Patient Emergency Risk Classification

# TEAM 7
- Saharsh Misra    - 2022A7PS0074H
- Aarav Haran      - 2022B3A70880H
- Vaishnu Kanna    - 2022B3A71608H
- Aditya Pentyala  - 2022B3A70522H

# USAGE INSTRUCTIONS
0. Ensure you have Python>=3.8 on your system
1. Clone this repository onto your local system
2. Download the dataset from the provided google drive link as part of the assignment and place it in the root directory of your clone ***(SKIP IF REPO DOWNLOADED WITH THE DATA DIRECTORY INTACT)***
3. Create a new virtual environment with the command `python -m venv env`
4. Activate the environment: 
    - `source env/bin/activate` on Mac/Linux
    - `env/Scripts/activate.ps1` on Windows Powershell
    - `env/Scripts/activate.bat` on Windows cmd
5. Once inside the environment, install the requirements using `pip install -r requirements.txt`
6. Run the notebook _eda.ipynb_ to populate the _data/_ directory (you may have to create an empty directory at first) ***(SKIP IF REPO DOWNLOADED WITH THE DATA DIRECTORY INTACT)***
7. Run the command `streamlit run dashboard.py`

# EVALUATION RUBRIC
1. Data Preprocessing and Exploratory Data Analysis (EDA) (5 Marks)
    - Done in `eda.ipynb` and through plots in `dashboard.py`
2. Feature Engineering and Representation (5 Marks)
    - Done in `eda.ipynb`
3. Classification Models (5 Marks)
    - Done in `models/`
4. Hyperparameter Tuning and Model Optimization, Complexity, Generalization, and Interpretation (5 Marks)
    - Done in `model_tests.ipynb` and saved to `saved_models/`
    - Hyperparameter tuning done through _GridSearchCV_ (sparse search space due to computational constraints)
    - Complexity & generalization shown through outputs within notebook and dashboard
    - Interpretation given in `INTERPRETATION.md`
5. Overall ML Pipeline and Automation (5 Marks)
    - `eda.ipynb` (ETL Pipeline) -> `model_tests.ipynb` (train models) -> `dashboard.py` (interpret & visualize) 
6. Visualization and Video Reporting (5 Marks)
    - Done in `dashboard.py` and `Team07_Assignment2_Video.mp4`
7. Code Demo and Viva (10 Marks - Individual Assessment)

# PROBLEM 
Classify persons into being high or low risk for having an emergency room admission.

## Target
- HIGH_EMERGENCY_RISK: Positive (1) if >2 emergency room encounters logged, else Negative (0)

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