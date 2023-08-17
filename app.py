from flask import Flask,request,render_template
import dill
import numpy as np



app=Flask('__name__')
@app.route('/')
def read_main():
    return render_template('index.html')

@app.route('/predict',methods=['GET'])
def generate_output():
    json_data = False
    input_data = request.args.get('data')
    if input_data==None:
        input_data = request.get_json()
        json_data = True
    # input_text = SPX_USO_SLV_EUR_USD_comma_separated_values
    GLD = process_and_predict(input_text=input_data,json_data=json_data)
    return {'predicted':GLD}

def process_and_predict(input_text,json_data):
    if(json_data==True):
        output_text = [float(item) for item in input_text['data'].split(',')]
    else:
        output_text = [float(item) for item in input_text.split(',')]
    with open('src/models/preprocessor.pkl', 'rb') as p:
        preprocessor = dill.load(p)
    for i in range(13):
        if(i==9):
            output_text[i] = output_text[i]
        else:
            output_text[i] = int(output_text[i])
    output_text = np.array(output_text).reshape(1, -1)
    output_text_dims = preprocessor.transform(output_text)
    with open('src/models/model.pkl', 'rb') as m:
        model = dill.load(m)
    heart_disease_prediction = model.predict(output_text_dims)
    if heart_disease_prediction[0] == 1.0:
        return "Heart Disease"
    else:
        return "Healthy Heart"
if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000)