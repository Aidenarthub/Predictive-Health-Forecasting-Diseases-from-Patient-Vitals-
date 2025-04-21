import joblib
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load pre-trained models
diabetes_model_file = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(diabetes_model_file, 'rb'))

cancer_model_file = 'model.pkl'
model = pickle.load(open(cancer_model_file, 'rb'))

# Initialize Flask app
app = Flask(__name__)

# Define routes
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/help')
def help():
    return render_template("help.html")

@app.route('/terms')
def terms():
    return render_template("tc.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/disindex")
def disindex():
    return render_template("disindex.html")

@app.route("/cancer")
def cancer():
    return render_template("cancer.html")

@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")

@app.route("/liver")
def liver():
    return render_template("liver.html")

# Function to predict liver disease
def ValuePred(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    
    if size == 7:
        loaded_model = joblib.load('liver_model.pkl')
        result = loaded_model.predict(to_predict)
        return result[0]
    
    return None

@app.route('/predictliver', methods=["POST"])
def predictliver():
    if request.method == "POST":
        to_predict_list = list(request.form.values())
        to_predict_list = list(map(float, to_predict_list))

        if len(to_predict_list) == 7:
            result = ValuePred(to_predict_list, 7)

            if result == 1:
                prediction = "Patient has a high risk of Liver Disease, please consult your doctor immediately."
            else:
                prediction = "Patient has a low risk of Liver Disease."

            return render_template("liver_result.html", prediction_text=prediction)

    return render_template("liver.html", prediction_text="Invalid input data!")

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [int(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape', 'marginal_adhesion',
                     'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)

    res_val = "a high risk of Breast Cancer" if output == 4 else "a low risk of Breast Cancer"

    return render_template('cancer_result.html', prediction_text=f'Patient has {res_val}')

# Load and preprocess diabetes dataset
df1 = pd.read_csv('diabetes.csv')

# Rename columns
df1.rename(columns={'DiabetesPedigreeFunction': 'DPF'}, inplace=True)

# Replace zero values with NaN in specific columns
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df1[cols_with_zeros] = df1[cols_with_zeros].replace(0, np.NaN)

# Fill NaN values with appropriate statistics
df1['Glucose'].fillna(df1['Glucose'].mean(), inplace=True)
df1['BloodPressure'].fillna(df1['BloodPressure'].mean(), inplace=True)
df1['SkinThickness'].fillna(df1['SkinThickness'].median(), inplace=True)
df1['Insulin'].fillna(df1['Insulin'].median(), inplace=True)
df1['BMI'].fillna(df1['BMI'].median(), inplace=True)

# Prepare training data
X = df1.drop(columns='Outcome')
y = df1['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Train and save the RandomForest model for diabetes prediction
diabetes_classifier = RandomForestClassifier(n_estimators=20)
diabetes_classifier.fit(X_train, y_train)

# Save the model
pickle.dump(diabetes_classifier, open(diabetes_model_file, 'wb'))

@app.route('/predictt', methods=['POST'])
def predictt():
    if request.method == 'POST':
        try:
            preg = float(request.form['pregnancies'])
            glucose = float(request.form['glucose'])
            bp = float(request.form['bloodpressure'])
            st = float(request.form['skinthickness'])
            insulin = float(request.form['insulin'])
            bmi = float(request.form['bmi'])
            dpf = float(request.form['dpf'])
            age = float(request.form['age'])

            data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
            my_prediction = classifier.predict(data)

            prediction_text = "Diabetes Positive" if my_prediction[0] == 1 else "Diabetes Negative"

            return render_template('diab_result.html', prediction=prediction_text)

        except Exception as e:
            return render_template('diabetes.html', prediction="Error: Invalid input data!")

# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)
