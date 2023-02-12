from flask import Flask,request,jsonify,render_template
import numpy as np
import pickle


#load the model from pickle 
model = pickle.load(open('./model.pkl','rb'))
#Write the app 
app = Flask(__name__)

@app.route('/',methods=['GET'])
def home():
    return render_template("index.html")




#Post the Predict Route 
@app.route('/predict',methods=['POST'])
def index():
    
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    
    data5 = np.array([[data1,data2,data3,data4]],dtype=int)
    result = model.predict(data5)
    
    return render_template("after.html",data=result)
    
   
    
    # 'Outlook','Temparature','Humidity','Wind','Play Tennis'




if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
   

