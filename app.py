from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("thyroid_model.pkl", "rb"))

@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")


@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        Age = int(request.form["age"])
        T3 = float(request.form["T3"])
        TT4 = float(request.form["TT4"])
        T4U = float(request.form["T4U"])
        FTI = float(request.form["FTI"])
        sex = request.form['sex']
        if (sex == "sex_M"):
            sex_M = 1
        else:
            sex_M = 0

        sick = request.form['sick']
        if (sick == 'sick_t'):
            sick_t = 1
        else:
            sick_t = 0

        pregnant = request.form['pregnant']
        if (pregnant == 'pregnant_t'):
            pregnant_t = 1
        else:
            pregnant_t = 0

        thyroid_surgery = request.form['thyroid_surgery']
        if (thyroid_surgery == 'thyroid_surgery_t'):
            thyroid_surgery_t = 1
        else:
            thyroid_surgery_t = 0

        goitre = request.form['goitre']
        if(goitre == 'goitre_t'):
            goitre_t = 1
        else:
            goitre_t = 0

        tumor = request.form['tumor']
        if (tumor == 'tumor_t'):
            tumor_t = 1
        else:
            tumor_t = 0

       
        prediction = model.predict([[Age,
                                     T3,
                                     TT4,
                                     T4U,
                                     FTI,
                                     sex_M,
                                     sick_t,
                                     pregnant_t,
                                     thyroid_surgery_t,
                                     goitre_t,
                                     tumor_t]])

    output = prediction[0]

    if output == 0:
        return render_template('home.html', prediction_text='Thyroid_Result : Hyperthyroid')
    elif output == 1:
        return render_template('home.html', prediction_text='Thyroid_Result : Hypothyroid')
    elif output == 2:
        return render_template('home.html', prediction_text= 'Thyroid_Result : Negative')
    else:
        return render_template('home.html', prediction_text='Thyroid_Result : Sick')


if __name__ == '__main__':
    app.run(debug=True)






























