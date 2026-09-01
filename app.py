from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load and preprocess data
data = pd.read_csv('train.csv')

# Clean data (corrected chaining warning)
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Encode categorical variables
le_sex = LabelEncoder()
le_embarked = LabelEncoder()
data['Sex'] = le_sex.fit_transform(data['Sex'])
data['Embarked'] = le_embarked.fit_transform(data['Embarked'])

# Features and labels
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = data[features]
y = data['Survived']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form
    try:
        input_data = pd.DataFrame([{
            'Pclass': int(form_data['Pclass']),
            'Sex': le_sex.transform([form_data['Sex']])[0],
            'Age': float(form_data['Age']),
            'SibSp': int(form_data['SibSp']),
            'Parch': int(form_data['Parch']),
            'Fare': float(form_data['Fare']),
            'Embarked': le_embarked.transform([form_data['Embarked']])[0]
        }])

        prediction = model.predict(input_data)[0]
        result = "Survived" if prediction == 1 else "Did not survive"
        return render_template('index.html', prediction=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
