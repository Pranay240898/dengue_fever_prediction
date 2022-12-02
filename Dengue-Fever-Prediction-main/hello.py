from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import statsmodels.api as sm

#just for the sake of this blog post!
from warnings import filterwarnings
from flask import Flask,render_template,url_for,request
app = Flask(__name__)
@app.route('/')
def home():
	return render_template("home.html")

@app.route('/results',methods=['POST','GET'])
def results():
	path1= request.form['path1']
	path2= request.form['path2']

	filterwarnings('ignore')
	train_features = pd.read_csv(path1,
	                             index_col=[0,1,2])
	train_labels = pd.read_csv(path2,
	                           index_col=[0,1,2])
	sj_train_features = train_features.loc['sj']
	sj_train_labels = train_labels.loc['sj']
	iq_train_features = train_features.loc['iq']
	iq_train_labels = train_labels.loc['iq']
	sj_train_features.head()
	sj_train_features.drop('week_start_date', axis=1, inplace=True)
	iq_train_features.drop('week_start_date', axis=1, inplace=True)
	pd.isnull(sj_train_features).any()

	sj_train_features.fillna(method='ffill', inplace=True)
	iq_train_features.fillna(method='ffill', inplace=True)
	sj_train_features['total_cases'] = sj_train_labels.total_cases
	iq_train_features['total_cases'] = iq_train_labels.total_cases
	sj_correlations = sj_train_features.corr()
	iq_correlations = iq_train_features.corr()


	def preprocess_data(data_path, labels_path=None):
	    df = pd.read_csv(data_path, index_col=[0, 1, 2])
	    features = ['reanalysis_specific_humidity_g_per_kg', 
	                 'reanalysis_dew_point_temp_k', 
	                 'station_avg_temp_c', 
	                 'station_min_temp_c']
	    df = df[features]
	    df.fillna(method='ffill', inplace=True)
	    if labels_path:
	        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
	        df = df.join(labels)
	    sj = df.loc['sj']
	    iq = df.loc['iq']
	    return sj, iq 
	sj_train, iq_train = preprocess_data(path1,
	                                    labels_path=path2)
	sj_train.describe()
	iq_train.describe()
	sj_train_subtrain = sj_train.head(800)
	sj_train_subtest = sj_train.tail(sj_train.shape[0] - 800)

	iq_train_subtrain = iq_train.head(400)
	iq_train_subtest = iq_train.tail(iq_train.shape[0] - 400)
	from statsmodels.tools import eval_measures
	import statsmodels.formula.api as smf

	def get_best_model(train, test):
	    # Step 1: specify the form of the model
	    model_formula = "total_cases ~ 1 + " \
	                    "reanalysis_specific_humidity_g_per_kg + " \
	                    "reanalysis_dew_point_temp_k + " \
	                    "station_min_temp_c + " \
	                    "station_avg_temp_c"
	    
	    grid = 10 ** np.arange(-8, -3, dtype=np.float64)
	                    
	    best_alpha = []
	    best_score = 1000

	    for alpha in grid:
	        model = smf.glm(formula=model_formula,
	                        data=train,
	                        family=sm.families.NegativeBinomial(alpha=alpha))

	        results = model.fit()
	        predictions = results.predict(test).astype(int)
	        score = eval_measures.meanabs(predictions, test.total_cases)

	        if score < best_score:
	            best_alpha = alpha
	            best_score = score

	    print('best alpha = ', best_alpha)
	    print('best score = ', best_score)
	            
	    # Step 3: refit on entire dataset
	    full_dataset = pd.concat([train, test])
	    model = smf.glm(formula=model_formula,
	                    data=full_dataset,
	                    family=sm.families.NegativeBinomial(alpha=best_alpha))

	    fitted_model = model.fit()
	    return fitted_model    
	sj_best_model = get_best_model(sj_train_subtrain, sj_train_subtest)
	iq_best_model = get_best_model(iq_train_subtrain, iq_train_subtest)
	sj_test, iq_test = preprocess_data('DengAI_Predicting_Disease_Spread_-_Test_Data_Features.csv')

	sj_predictions = sj_best_model.predict(sj_test).astype(int)
	iq_predictions = iq_best_model.predict(iq_test).astype(int)

	submission = pd.read_csv("DengAI_Predicting_Disease_Spread_-_Submission_Format.csv",
	                         index_col=[0, 1, 2])

	submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
	submission.to_csv("benchmark.csv")
	s=[]
	import csv
	with open("benchmark.csv",'r') as f:
		s=list(csv.reader(f))
	return render_template("results.html",s=s,path1=path1,path2=path2)

@app.route('/stats')
def stats():
	return render_template("stats.html",)
if __name__ == '__main__':
    app.run(debug=True)