from flask import Flask,request,jsonify,render_template
import pandas as pd
import os
import json
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
import pickle



app = Flask(__name__)
app.config['DEBUG'] = True

# filename = os.path.join(app.static_folder,'data.json')


@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/calculate',methods=['GET'])
def calculate():
    # df = pd.read_json(filename)
    p_loc = request.args.get('p_loc')
    p_lost_num = request.args.get('p_lost_num')
    p_plant_num = request.args.get('p_plant_num')
    if int(p_plant_num) <=0:
        return '產量分級(4級):1'+'<br>'+'預測產量(公噸):[0.0]'
    p_sunshine = request.args.get('p_sunshine')
    p_temp = request.args.get('p_temp')
    file_path = 'static/data/100_108_Banana_price.csv'
    dt = pd.read_csv(file_path)
    df_features = dt[['plant_number','location_id','lost_amount','sunshine_hours','temp_mean']]
    df_features.loc[-1] = [p_plant_num, p_loc, p_lost_num, p_sunshine, p_temp]  # adding a row
    df_features.index = df_features.index + 1  # shifting index
    df_features = df_features.sort_index()
    new_dt = df_features.iloc[0,:]
    X_p = df_features
    scaler = preprocessing.StandardScaler().fit(X_p)
    x = scaler.transform(X_p)
    x2 = pd.DataFrame(data=x,columns=X_p.columns)
    new_dt_a = np.array(new_dt)
    new_dt_as = new_dt_a.reshape(1,-1)
    new_dt_scl = scaler.transform(new_dt_as)
    dnn_model = tf.keras.models.load_model('static/data/dnn_trained_model')
    new_p = pd.DataFrame(data=new_dt_scl,columns=df_features.columns)
    new_y = np.argmax(dnn_model.predict(new_p), axis=-1)
    ####
    path_h = "static/data/Reg_"
    path_t = ".pickle.dat"
    level = str(new_y[0])
    file_path = path_h + level + path_t
    model_name = pickle.load(open(file_path, "rb"))
    pre_V = model_name.predict(new_p)
    return '產量分級(4級):'+str(new_y[0])+'<br>'+'預測產量(公噸):'+str(pre_V)

app.run()




