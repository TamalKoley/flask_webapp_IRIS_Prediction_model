from flask import request,render_template,Flask;
import pickle;

application=Flask(__name__);
app=application;

grid_log_model=pickle.load(open('models/grid_log_model.pkl','rb'))
le=pickle.load(open('models/label_encoder.pkl','rb'))

@app.route('/',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        sepal_length=float(request.form.get('sepal_length'))
        sepal_width=float(request.form.get('sepal_width'))
        petal_length=float(request.form.get('petal_length'))
        petal_width=float(request.form.get('petal_width'))
        y_pred=grid_log_model.predict([[sepal_length,sepal_width,petal_length,petal_width]])
        result=le.classes_[y_pred[0]];
        return render_template('index.html',result=result)
    else:
        return render_template('index.html');

if __name__=='__main__':
    app.run(host='0.0.0.0')



