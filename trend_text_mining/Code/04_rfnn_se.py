#!/usr/bin/python

'''Python code to fit RF and NN model using lpi texts and trends.
	Testing top 4 trend categorisations as identified from RF models.
	Evaluating impact of varying se_t
'''

# Local env: conda activate py3_env


# Load packages
import os
import os.path
import numpy as np
import pandas as pd
import random
import datetime
import re

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.porter import PorterStemmer as ps

import nltk.data
from nltk import sent_tokenize

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.utils.multiclass import type_of_target

import itertools

import tensorflow as tf
import tensorflow_hub as hub

from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, GlobalMaxPooling1D, Lambda
from keras.layers import Conv1D, MaxPooling1D, Embedding, SpatialDropout1D
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model

from keras.layers import Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils
from keras import regularizers

from sklearn.preprocessing import LabelBinarizer, LabelEncoder


########################################################################
# Functins/constants
########################################################################



bold_re = re.compile(r'</*b>')
italic_re = re.compile(r'</*i>')
typewrite_re = re.compile(r'</*tt>')
under_re = re.compile(r'</*u>')
emph_re = re.compile(r'</*em>')
big_re = re.compile(r'</*big>')
small_re = re.compile(r'</*small>')
strong_re = re.compile(r'</*strong>')
sub_re = re.compile(r'</*sub>')
sup_re = re.compile(r'</*sup>')
inf_re = re.compile(r'</*inf>')

regex_list = [bold_re, italic_re, typewrite_re, under_re, emph_re, big_re, small_re, strong_re, sub_re, sup_re, inf_re] 

def rm_html_tags(txt):

	for i in regex_list:
		txt = i.sub('', txt)
	
	return txt

def rm_auth_suff(txt):
	tags = ["-from Authors", "-from Author", "-Authors", "-Author"]
	for tag in tags:
		txt = txt.replace(tag, "")
	return(txt)


def sp_rm_stop(text):
	if type(text) == str:
		text = remove_stopwords(" ".join(simple_preprocess(text, deacc = True, min_len = 3)))
	
	return(text)	


def expand_grid(data_dict):
	rows = itertools.product(*data_dict.values())
	return (pd.DataFrame.from_records(rows, columns=data_dict.keys()))


NN classifier...
def nn_build(nlayers, nnodes, dropout, kern_reg, n_out, out_type):
	inp = Input(shape=(512,), dtype = "float32")

	x = Dense(nnodes, activation='relu',
								kernel_regularizer = kern_reg)(inp)
	x = Dropout(dropout)(x)
	
	for i in range(1,nlayers):
		x = Dense(nnodes, activation='relu',
								kernel_regularizer = kern_reg)(x)
		x = Dropout(dropout)(x)
	
	if out_type == "class":
		pred = Dense(n_out, activation='softmax')(x)
		model = Model(inputs=[inp], outputs=pred)
		if n_out>1:
			model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		elif n_out == 1:
			model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	elif out_type == "cont":
		pred = Dense(n_out)(x) 	
		model = Model(inputs=[inp], outputs=pred)
		model.compile(loss='mean_squared_error', optimizer='adam')
	
	return(model)
        

def encode(le, labels):
	enc = le.transform(labels)
	return utils.to_categorical(enc)

def decode(le, one_hot):
	dec = np.argmax(one_hot, axis=1)
	return le.inverse_transform(dec)


# K-fold cross validation of NN models
def cv_keras(k, df, seed, class_col, se_t):
	out_ls = []

	# Subset to se threshold
	df = df.loc[(df.se <= se_t) | (pd.isna(df.se))].reset_index(drop = True)

	cv = StratifiedKFold(n_splits=k, shuffle = True, random_state = 1)
	out_type = "class"

	# Drop varied texts
	df = df.loc[df[class_col]!="Varied"].reset_index(drop = True)

	# Find min class count 
	n_samp = min(df[class_col].value_counts())

	# Sub sample
	stab = df.loc[df[class_col]=="Stable"].sample(n=n_samp, random_state=seed)
	incr = df.loc[df[class_col]=="Increase"].sample(n=n_samp, random_state=seed)
	decl = df.loc[df[class_col]=="Decline"].sample(n=n_samp, random_state=seed)
	
	df = pd.concat([stab, incr, decl]).reset_index(drop = True)

	x = df["TA"]
	y = df[class_col]

	# embed txt once here, then use indices from cv.split to index embedding array...
	x_emb = embed(x)

	fold_id = 1
	for (tr, te) in cv.split(x, y):
	
		print ("Fold: ", str(fold_id))

		# encode y values
		
		le = LabelEncoder()
		le.fit(y)
		tr_y = np.asarray(encode(le, y[tr]))
		n_y = tr_y.shape[1]
		
		# create nn/cnn mod
		nn = nn_build(2, 256, 0.5, "l2", n_y, out_type)

		tr_emb = np.array(tf.gather(x_emb, tr))
		y_ = tr_y
		
		te_emb = np.array(tf.gather(x_emb, te))
		
		nn.fit(tr_emb, y_, epochs = 20, validation_split = 0)

		# test model	
		nn_tr_pred = decode(le, nn.predict(tr_emb)).tolist()
		nn_te_pred = decode(le, nn.predict(te_emb)).tolist()

		nn_tr_true = decode(le, y_).tolist()

		# store output to df
		out_df = pd.DataFrame({
								"N_folds"		:	k,
								"Fold_id"		:	fold_id,
								"N_train"		:	len(nn_tr_true),
								"N_test"		:	len(y[te]),
								"Class_type"	: 	class_col,
								"se_t"			:   se_t,
								"Seed"			:   seed,
								"Classifier"	:	"nn",
								"Train_class"	:	str(nn_tr_true),
								"Test_class"	:	str(y[te].tolist()),
								"Train_pred"	:	str(nn_tr_pred),
								"Test_pred"		:	str(nn_te_pred)},
								index = [0])

		out_ls.append(out_df) 

		fold_id += 1

	# bind dfs and return
	out_df = pd.concat(out_ls)
	return(out_df)


# K-fold cross validation of RF model
def cv_skl(k, df, seed, class_col, se_t):
	out_ls = []

	# Subset to se threshold
	df = df.loc[(df.se <= se_t) | (pd.isna(df.se))].reset_index(drop = True)


	cv = StratifiedKFold(n_splits=k, shuffle = True, random_state = 1)
	
	# Drop varied texts
	df = df.loc[df[class_col]!="Varied"].reset_index(drop = True)

	# Find min class count 
	n_samp = min(df[class_col].value_counts())

	# Sub sample
	stab = df.loc[df[class_col]=="Stable"].sample(n=n_samp, random_state=seed)
	incr = df.loc[df[class_col]=="Increase"].sample(n=n_samp, random_state=seed)
	decl = df.loc[df[class_col]=="Decline"].sample(n=n_samp, random_state=seed)
	
	df = pd.concat([stab, incr, decl]).reset_index(drop = True)

	x = df["TA_rf"]
	y = df[class_col]

	fold_id = 1
	for (tr, te) in cv.split(x, y):
		print ("Fold: ", str(fold_id))
	
		# create tfidf matrix
		vect = None
		vect = TfidfVectorizer(analyzer = 'word', ngram_range = (1,2))

		tr_tfidf = vect.fit_transform(x[tr])
		y_ = y[tr].tolist()
		n_tr = len(y_)

		te_tfidf = vect.transform(x[te])
		te_y = y[te]

		# fit rf to train
		rf = RandomForestClassifier(random_state = 1).fit(tr_tfidf, y_)
		
		rf_tr_pred = rf.predict(tr_tfidf).tolist()
		
		# apply model	
		rf_te_pred = rf.predict(te_tfidf).tolist()
		
		# store output to df
		out_df = pd.DataFrame({
								"N_folds"		:	k,
								"Fold_id"		:	fold_id,
								"N_train"		:	n_tr,
								"N_test"		:	len(te_y),
								"Class_type"	: 	class_col,
								"se_t"			:   se_t,
								"Seed"			:   seed,
								"Classifier"	:	"rf",
								"Train_class"	:	str(y_),
								"Test_class"	:	str(te_y.tolist()),
								"Train_pred"	:	str(rf_tr_pred),
								"Test_pred"		:	str(rf_te_pred)
								}, index = [0])
		
		out_ls.append(out_df) 

		fold_id += 1

	# bind dfs and return
	out_df = pd.concat(out_ls)
	return(out_df)


########################################################################
# Main code
########################################################################

# Code to generate specification df, run locally...
class_col_ls = ["Trend_sig_01_60","Trend_sig_005_60",
				"Trend_sig_01_maj","Trend_sig_005_maj"] 

seed_ls = np.arange(1,11).tolist()  
se_t_ls = [0.3,0.2,0.1,0.05,0.025,0.0125]	

spec_df = expand_grid({"mod"		:	["rf", "nn"],
						"class_col"	:	class_col_ls,
						"se_t"		:   se_t_ls,
						"seed"		:   seed_ls
						})

spec_df["id"] = [i for i in range(spec_df.shape[0])]
# spec_df.to_csv("../Data/se_specs.csv", index = False)
# 480


#####
# Load texts
txt_dat = pd.read_csv("../Data/lpi_texts.csv", encoding = "latin-1")
# Load lpi trend data...
trend_dat = pd.read_csv("../Data/lpi_trends.csv")
# Merge by rn
dat = txt_dat.loc[:,["RN", "Title_eng", "AB_eng"]].merge(trend_dat, on = "RN")

# Drop texts which are in the manually verified sample...
holdout = pd.read_csv("../Data/compiled_trend_text1.csv",
						encoding = "latin-1")
dat = dat.loc[~dat["RN"].isin(holdout["RN"])].reset_index(drop = True)

# Remove records with question marks ...
# [2843, 2587, 2812]
dat = dat.loc[~dat.RN.isin([2587, 2812, 2843])].reset_index(drop = True)


dat["Title"] = dat.Title_eng.copy()
dat["Title"].fillna("", inplace=True)

# remove author ending
dat["AB"] = dat.AB_eng.apply(rm_auth_suff)

# rm html
dat["AB"] = dat.AB.apply(rm_html_tags)
dat["Title"] = dat.Title.apply(rm_html_tags)

# drop texts with AB < 200 characters
dat["AB_len"] = dat.AB.apply(len)
dat["Title_len"] = dat.Title.apply(len)

dat = dat.loc[dat.AB_len>=200]
dat = dat.reset_index(drop = True)

dat["TA"] = dat.Title + " " + dat.AB

# Simple preprocess+stop word removal
dat["TA"] = dat.TA.apply(sp_rm_stop)

dat["TA_rf"] = dat.TA.apply(ps().stem_sentence)

embed = hub.load("..//Data/universal-sentence-encoder-large_5") 


for i in range(spec_df.shape[0]):
	f = "res_"+str(i)+".csv"
	fp = "../Results/rfnn_se/"+f

	cv_spec = spec_df.loc[i,]

	# 10-fold cv
	if cv_spec["mod"] == "rf":
		
		out_df = cv_skl(10, dat.copy(), cv_spec.seed, cv_spec.class_col, cv_spec.se_t)

	if cv_spec["mod"] == "nn":
		
		out_df = cv_keras(10, dat.copy(), cv_spec.seed, cv_spec.class_col, cv_spec.se_t)

	out_df = out_df.reset_index(drop = True)
	out_df["id"] = cv_spec.id
	out_df.to_csv(fp, index = False)

