#!/usr/bin/python

'''Python code to fit NN (based on Google's sentence encoder) 
   using lpi texts and trends.
   Testing top trend categorisations as identified from RF models.
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
# Functions/constants
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


## NN classifier construction
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


def cv_keras(k, df, seed, class_col, incl_varied_bool):
	out_ls = []

	cv = StratifiedKFold(n_splits=k, shuffle = True, random_state = 1)
	out_type = "class"

	if incl_varied_bool != True:
		# Drop varied texts
		df = df.loc[df[class_col]!="Varied"].reset_index(drop = True)

	# Find min class count 
	n_samp = min(df[class_col].value_counts())

	# Sub sample
	stab = df.loc[df[class_col]=="Stable"].sample(n=n_samp, random_state=seed)
	incr = df.loc[df[class_col]=="Increase"].sample(n=n_samp, random_state=seed)
	decl = df.loc[df[class_col]=="Decline"].sample(n=n_samp, random_state=seed)
	
	if "Varied" in df[class_col].unique():
		vari = df.loc[df[class_col]=="Varied"].sample(n=n_samp, random_state=seed)
	
		# combine
		df = pd.concat([stab, incr, decl, vari]).reset_index(drop = True)
	else:
		df = pd.concat([stab, incr, decl]).reset_index(drop = True)

	x = df["TA"]
	y = df[class_col]

	# embed txt once here, then use indices from cv.split to index embedding array...
	x_emb = embed(x)

	fold_id = 1

	for (tr, te) in cv.split(x, y):
		# print(tr)
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
								"Incl_varied"	:   incl_varied_bool,
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



########################################################################
# Main code
########################################################################

# # Code to generate specification df, run locally...
class_col_ls = ["Trend_sig_01_60","Trend_sig_005_60",
				"Trend_sig_01_maj","Trend_sig_005_maj"] 


incl_varied_ls = [True, False]

seed_ls = np.arange(1,11).tolist()  

# # # make df
spec_df = expand_grid({"mod"		:	["nn"],
						"class_col"	:	class_col_ls,
						"incl_varied":  incl_varied_ls,
						"seed"		:   seed_ls
						})

spec_df["id"] = [i for i in range(spec_df.shape[0])]
# spec_df.to_csv("../Data/nn_specs.csv", index = False)
# 80


#####
# Load texts
txt_dat = pd.read_csv("../Data/lpi_texts.csv", encoding = "latin-1")

# Load lpi trend data...
trend_dat = pd.read_csv("../Data/lpi_trends.csv")

# Merge by rn
dat = txt_dat.loc[:,["RN", "Title_eng", "AB_eng"]].merge(trend_dat, on = "RN")

# Load holdout to ignore
holdout = pd.read_csv("../Data/compiled_trend_text1.csv", encoding = "latin-1")

dat = dat.loc[~dat["RN"].isin(holdout["RN"])].reset_index(drop = True)

# Remove records with question marks ...
# [2843, 2587, 2812]
dat = dat.loc[~dat.RN.isin([2587, 2812, 2843])].reset_index(drop = True)

# 
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

dat = dat.loc[dat.AB_len>=200].reset_index(drop = True)

# Combine title and abstract
dat["TA"] = dat.Title + " " + dat.AB

# Simple preprocess+stop word removal
dat["TA"] = dat.TA.apply(sp_rm_stop)

# Note: google's large senetence encoder should be downladed to run this.
embed = hub.load("../Data/universal-sentence-encoder-large_5") 
	

# 10-fold cv
for i in range(spec_df.shape[0]):
	f = "res_"+str(i)+".csv"
	fp = "../Results/nn_init/"+f

	cv_spec = spec_df.loc[i,]

	out_df = cv_keras(10, dat.copy(), cv_spec.seed, cv_spec.class_col, cv_spec.incl_varied)

	out_df = out_df.reset_index(drop = True)
	out_df["id"] = cv_spec.id
	out_df.to_csv(fp, index = False)

#####
