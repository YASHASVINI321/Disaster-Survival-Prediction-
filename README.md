# Disaster Survival Prediction

## Project Overview

This project is a machine learning web application that predicts whether a passenger survived a disaster based on passenger information.

The application uses a Random Forest Classifier and is developed using Python and Flask.

## Technologies Used

- Python
- Flask
- Pandas
- Scikit-learn
- Random Forest Classifier
- HTML

## Features

- Accepts passenger details through a web interface
- Preprocesses the input data
- Uses a trained Random Forest machine learning model
- Predicts whether the passenger survived or did not survive

## Dataset

The project uses the Titanic passenger dataset (`train.csv`) for training the machine learning model.

## How to Run

1. Install the required Python libraries:

```bash
pip install flask pandas scikit-learn

python app.py

http://127.0.0.1:5000

Disaster-Survival-Prediction
│
├── app.py
├── train.csv
└── templates
    └── index.html

## Machine Learning Model

The project uses a Random Forest Classifier to predict passenger survival based on features such as:

- Passenger class
- Sex
- Age
- Number of siblings/spouses
- Number of parents/children
- Fare
- Embarked location
