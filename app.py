from flask import Flask,render_template,request
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline  import CustomData,PredictionPipeline
from src.exception import CustomException
import sys

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict():
    try:
        if request.method == 'GET':
            return render_template('form.html')
        
        else:
            form_data=request.form
            data= CustomData(

               gender = form_data["gender"],
               race_ethnicity = form_data["race_ethnicity"],
               parental_level_of_education = form_data["parental_level_of_education"],
               lunch = form_data["lunch"],
               test_preparation_course = form_data["test_preparation_course"],
               reading_score = int(form_data["reading_score"]),
               writing_score = int(form_data["writing_score"])
             )

            data_frame=data.get_data_as_dataframe()
            print(data_frame)

            predict_pipeline= PredictionPipeline()
            result=predict_pipeline.predict(data_frame)
            return render_template('form.html',result=result[0])
    except Exception as e:
        raise CustomException(e,sys)      

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)



