#!/usr/bin/python

'''Python code to fit RF trend classifications using lpi texts and trends.
   Testing various trend categorisations.
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

# Remove html tags in text
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


# Conduct k-fold cv of an RF clasifier, based on the specified categorisation columns etc.
def cv_skl(k, df, seed, class_col, incl_varied_bool):
	out_ls = []

	# cv seed should remain constant (1), can then compare across sample seeds for 
	# the same cv splits
	cv = StratifiedKFold(n_splits=k, shuffle = True, random_state = 1)
	
	if incl_varied_bool != True:
		# Drop varied texts
		df = df.loc[df[class_col]!="Varied"].reset_index(drop = True)

	# Find min class count - for undersmpling
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

	fold_id = 1
	# For each fold, 
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
								"Incl_varied"	:   incl_varied_bool,
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

# # Code to generate specification df, run locally...
class_col_ls = [ 
				"Trend_01_60","Trend_02_60","Trend_05_60",
				"Trend_01_maj","Trend_02_maj","Trend_05_maj",
				"Trend_sig_01_60","Trend_sig_005_60",
				"Trend_sig_01_maj","Trend_sig_005_maj",
				"Trend_meta_01","Trend_meta_005"] #

incl_varied_ls = [True, False]

seed_ls = np.arange(1,11).tolist()  

# make df
spec_df = expand_grid({"mod"		:	["rf"],
						"class_col"	:	class_col_ls,
						"incl_varied":  incl_varied_ls,
						"seed"		:   seed_ls
						})

spec_df["id"] = [i for i in range(spec_df.shape[0])]
# spec_df.to_csv("../Data/rf_specs.csv", index = False)
# 240


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

# Stemming
dat["TA"] = dat.TA.apply(ps().stem_sentence)


# 10-fold cv
for i in range(spec_df.shape[0]):
	f = "res_"+str(i)+".csv"
	fp = "../Results/rf_init/"+f

	cv_spec = spec_df.loc[i,]

	out_df = cv_skl(10, dat.copy(), cv_spec.seed, cv_spec.class_col, cv_spec.incl_varied)

	out_df = out_df.reset_index(drop = True)
	out_df["id"] = cv_spec.id
	out_df.to_csv(fp, index = False)

#####
