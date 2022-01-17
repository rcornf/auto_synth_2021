#!/usr/bin/python

'''Python code to fit selected (RF) classifier using lpi texts and trends.
	Then apply this to the with-held test set...
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


##------------------------------------------------------------------------------
## EDA.py code
# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

import random
from random import shuffle
random.seed(1)

#stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', '']

#cleaning up text
import re
def get_only_chars(line):

	clean_line = ""

	line = line.replace("’", "")
	line = line.replace("'", "")
	line = line.replace("-", " ") #replace hyphens with spaces
	line = line.replace("\t", " ")
	line = line.replace("\n", " ")
	line = line.lower()

	for char in line:
		if char in 'qwertyuiopasdfghjklzxcvbnm ':
			clean_line += char
		else:
			clean_line += ' '

	clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
	if clean_line[0] == ' ':
		clean_line = clean_line[1:]
	return clean_line

########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################

#for the first time you use wordnet
#import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet 

def synonym_replacement(words, n):
	new_words = words.copy()
	random_word_list = list(set([word for word in words if word not in stop_words]))
	random.shuffle(random_word_list)
	num_replaced = 0
	for random_word in random_word_list:
		synonyms = get_synonyms(random_word)
		if len(synonyms) >= 1:
			synonym = random.choice(list(synonyms))
			new_words = [synonym if word == random_word else word for word in new_words]
			#print("replaced", random_word, "with", synonym)
			num_replaced += 1
		if num_replaced >= n: #only replace up to n words
			break

	#this is stupid but we need it, trust me
	sentence = ' '.join(new_words)
	new_words = sentence.split(' ')

	return new_words

def get_synonyms(word):
	synonyms = set()
	for syn in wordnet.synsets(word): 
		for l in syn.lemmas(): 
			synonym = l.name().replace("_", " ").replace("-", " ").lower()
			synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
			synonyms.add(synonym) 
	if word in synonyms:
		synonyms.remove(word)
	return list(synonyms)

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):

	#obviously, if there's only one word, don't delete it
	if len(words) == 1:
		return words

	#randomly delete words with probability p
	new_words = []
	for word in words:
		r = random.uniform(0, 1)
		if r > p:
			new_words.append(word)

	#if you end up deleting all words, just return a random word
	if len(new_words) == 0:
		rand_int = random.randint(0, len(words)-1)
		return [words[rand_int]]

	return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
	new_words = words.copy()
	for _ in range(n):
		new_words = swap_word(new_words)
	return new_words

def swap_word(new_words):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0
	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words
	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
	return new_words

########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n):
	new_words = words.copy()
	for _ in range(n):
		add_word(new_words)
	return new_words

def add_word(new_words):
	synonyms = []
	counter = 0
	while len(synonyms) < 1:
		random_word = new_words[random.randint(0, len(new_words)-1)]
		synonyms = get_synonyms(random_word)
		counter += 1
		if counter >= 10:
			return
	random_synonym = synonyms[0]
	random_idx = random.randint(0, len(new_words)-1)
	new_words.insert(random_idx, random_synonym)

########################################################################
# main data augmentation function
########################################################################

def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
	
	sentence = get_only_chars(sentence)
	words = sentence.split(' ')
	words = [word for word in words if word is not '']
	num_words = len(words)
	
	augmented_sentences = []
	num_new_per_technique = int(num_aug/4)+1
	n_sr = max(1, int(alpha_sr*num_words))
	n_ri = max(1, int(alpha_ri*num_words))
	n_rs = max(1, int(alpha_rs*num_words))

	#sr
	for _ in range(num_new_per_technique):
		a_words = synonym_replacement(words, n_sr)
		augmented_sentences.append(' '.join(a_words))

	#ri
	for _ in range(num_new_per_technique):
		a_words = random_insertion(words, n_ri)
		augmented_sentences.append(' '.join(a_words))

	#rs
	for _ in range(num_new_per_technique):
		a_words = random_swap(words, n_rs)
		augmented_sentences.append(' '.join(a_words))

	#rd
	for _ in range(num_new_per_technique):
		a_words = random_deletion(words, p_rd)
		augmented_sentences.append(' '.join(a_words))

	augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
	shuffle(augmented_sentences)

	#trim so that we have the desired number of augmented sentences
	if num_aug >= 1:
		augmented_sentences = augmented_sentences[:num_aug]
	else:
		keep_prob = num_aug / len(augmented_sentences)
		augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

	#append the original sentence
	augmented_sentences.append(sentence)

	return augmented_sentences
##------------------------------------------------------------------------------


# Function to remove first and last sentence of abstract
def sent_lrstrip(txt):
	s_ls = sent_tokenize(txt) 
	n_s = len(s_ls) 
	if n_s>2:
		out = " ".join(s_ls[1:(n_s-1)])
	else: 
		out = txt
	return(out)


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

def rm_specific_wrds(txt, wrds_to_rm):
	for wrd in wrds_to_rm:
		txt = txt.replace(wrd, "")
	return(txt)


def text_prep(dat):
	dat["Title"] = dat.Title_eng.copy()
	dat["Title"].fillna("", inplace=True)

	# remove author ending
	dat["AB"] = dat.AB_eng.apply(rm_auth_suff)

	# rm html
	dat["AB"] = dat.AB.apply(rm_html_tags)
	dat["Title"] = dat.Title.apply(rm_html_tags)
	return(dat)


# text_process func...
def text_process(dat, m_row, sp_nms_dat, loc_nms_dat):
		# Sentence stripping
	if m_row.Sent_strp == True:
		# Rm 1st/last sent of abstract
		dat["AB"] = dat.AB.apply(sent_lrstrip).tolist()
	
	dat["TA"] = dat.Title + " " + dat.AB
	
	# if sp rm
	if m_row.Sp_rm == True:
		for i in range(dat.shape[0]):
			tmp_sp_nms = sp_nms_dat.loc[sp_nms_dat.RN==dat.RN[i],"original"].tolist()
			if len(tmp_sp_nms)>0:
				# dat.AB[i] = rm_specific_wrds(dat.AB[i], tmp_sp_nms)
				dat.TA[i] = rm_specific_wrds(dat.TA[i], tmp_sp_nms)
	
	# if loc rm
	if m_row.Loc_rm == True:
		for i in range(dat.shape[0]):
			tmp_loc_nms = loc_nms_dat.loc[loc_nms_dat.RN==dat.RN[i],"source.string"].tolist()
			if len(tmp_loc_nms)>0:
				# dat.AB[i] = rm_specific_wrds(dat.AB[i], tmp_loc_nms)
				dat.TA[i] = rm_specific_wrds(dat.TA[i], tmp_loc_nms)

	# Simple preprocess+stop word removal
	dat["TA"] = dat.TA.apply(sp_rm_stop)
	# Stemming
	# dat["TA"] = dat.TA.apply(ps().stem_sentence)

	return(dat)


# Given 10-fold cv, set to 0.05, 12
def skl_eda(k, train, test, seed, class_col, aug_bool, alpha, aug_n, 
			incl_varied_bool, m_nm):
	out_ls = []

	if incl_varied_bool != True:
		# Drop varied texts
		train = train.loc[train[class_col]!="Varied"].reset_index(drop = True)

	# Find min class count 
	n_samp = min(train[class_col].value_counts())

	# Sub sample
	stab = train.loc[train[class_col]=="Stable"].sample(n=n_samp, random_state=seed)
	incr = train.loc[train[class_col]=="Increase"].sample(n=n_samp, random_state=seed)
	decl = train.loc[train[class_col]=="Decline"].sample(n=n_samp, random_state=seed)
	
	if "Varied" in train[class_col].unique():
		vari = train.loc[train[class_col]=="Varied"].sample(n=n_samp, random_state=seed)
	
		# combine
		train = pd.concat([stab, incr, decl, vari]).reset_index(drop = True)
	else:
		train = pd.concat([stab, incr, decl]).reset_index(drop = True)


	# Do stemming here
	x = train.TA.apply(ps().stem_sentence)
	y = train[class_col]

	te_x = test.TA.apply(ps().stem_sentence)

	
	# create tfidf matrix
	vect = None
	vect = TfidfVectorizer(analyzer = 'word', ngram_range = (1,2))
		
	if aug_bool == True:

		tr_txt = x.tolist()
		tr_y = y.tolist()

		aug_txt = x.tolist()
		y_ = y.tolist()

		for i in range(len(tr_txt)):
			tmp_txt = eda(tr_txt[i], alpha_sr=alpha, alpha_ri=alpha, 
							alpha_rs=alpha, p_rd=alpha, num_aug = int(aug_n))
			tmp_y = [tr_y[i]]*len(tmp_txt)

			aug_txt.extend(tmp_txt)
			y_.extend(tmp_y)

		tr_tfidf = vect.fit_transform(aug_txt)
		n_tr = len(aug_txt)

	elif aug_bool == False:
		tr_tfidf = vect.fit_transform(x)
		y_ = y.tolist()
		n_tr = len(y_)



	te_tfidf = vect.transform(te_x)


	# fit rf to train
	rf = RandomForestClassifier(random_state = 1).fit(tr_tfidf, y_)

	# apply models
	rf_te_pred = rf.predict(te_tfidf).tolist()
	
	# add pred to col in test
	# need to sort out column name (classifier + "_" + seed)
	col_nm = m_nm + "_" + str(seed)
	test[col_nm] = rf_te_pred

	return(test)



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



def keras_eda(k, train, test, seed, class_col, aug_bool, alpha, n_aug, 
			incl_varied_bool, m_nm):

	out_type = "class"

	if incl_varied_bool != True:
		# Drop varied texts
		train = train.loc[train[class_col]!="Varied"].reset_index(drop = True)

	# Find min class count 
	n_samp = min(train[class_col].value_counts())

	# Sub sample
	stab = train.loc[train[class_col]=="Stable"].sample(n=n_samp, random_state=seed)
	incr = train.loc[train[class_col]=="Increase"].sample(n=n_samp, random_state=seed)
	decl = train.loc[train[class_col]=="Decline"].sample(n=n_samp, random_state=seed)
	
	if "Varied" in train[class_col].unique():
		vari = train.loc[train[class_col]=="Varied"].sample(n=n_samp, random_state=seed)
	
		# combine
		train = pd.concat([stab, incr, decl, vari]).reset_index(drop = True)
	else:
		train = pd.concat([stab, incr, decl]).reset_index(drop = True)

	x = train["TA"]
	y = train[class_col]

	te_x = test["TA"]

	
	if aug_bool == True:
		
		tr_txt = x.tolist()
		tr_y = y.tolist()

		aug_txt = x.tolist()
		y_ = y.tolist()

		for i in range(len(tr_txt)):
			tmp_txt = eda(tr_txt[i], alpha_sr=alpha, alpha_ri=alpha, 
							alpha_rs=alpha, p_rd=alpha, num_aug = int(n_aug))
			tmp_y = [tr_y[i]]*len(tmp_txt)

			aug_txt.extend(tmp_txt)
			y_.extend(tmp_y)


	elif aug_bool == False:
		aug_txt = x.tolist()
		y_ = y.tolist()
	
	# encode y values
	le = LabelEncoder()
	le.fit(y_)
	tr_y = np.asarray(encode(le, y_))
	n_y = tr_y.shape[1]


	# create nn/cnn mod
	nn = nn_build(2, 256, 0.5, "l2", n_y, out_type)

	# embed tr and te 
	tr_emb = np.array(embed(aug_txt))
	te_emb = np.array(embed(te_x))
	

	nn.fit(tr_emb, tr_y, epochs = 20, validation_split = 0)
	
	# test model
	nn_te_pred = decode(le, nn.predict(te_emb)).tolist()

	# add pred to col in test
	# need to sort out column name (classifier + "_" + seed)
	col_nm = m_nm + "_" + str(seed)
	test[col_nm] = nn_te_pred

	return(test)


def m_fit_apply(train, test, m_row, sp_nms_dat, loc_nms_dat):
	train = text_process(train, m_row, sp_nms_dat, loc_nms_dat)
	test = text_process(test, m_row, sp_nms_dat, loc_nms_dat)

	# determine type of classifer and run relevant eda function
	if m_row.Classifier == "rf":	
		for i in range(1,11):
			test = skl_eda(10, train, test, i, m_row.Class_type, 
							m_row.Augmented, m_row.Alpha_Sigma, 
							m_row.N_aug, m_row.Incl_varied, m_row["mod"])
			print ("seed: " + str(i))

	elif m_row.Classifier == "nn":
		for i in range(1,11):
			test = keras_eda(10, train, test, i, m_row.Class_type, 
							m_row.Augmented, m_row.Alpha_Sigma, 
							m_row.N_aug, m_row.Incl_varied, m_row["mod"])
			print ("seed: " + str(i))

	# save csv
	print("Saving predictions!")

	fp = "../Results/holdout_preds/" + m_row["mod"] + ".csv"

	test.to_csv(fp, index = False, encoding = "latin-1")
	return(None)
	# 



####
## Main Code
####

# rf, sig 0.05 a., n_aug == 1, + sentence removal 
rf_top_spec = pd.DataFrame({"mod"			: ["rf1"],
						    "Class_type" 	: ["Trend_sig_005_maj"],
					   		"Classifier" 	: ["rf"],
					   "Incl_varied" 	: [False],
					   "Sent_strp" 		: [True],
					   "Sp_rm" 			: [False],
					   "Loc_rm" 		: [False],
					   "Augmented" 		: [False],
					   "Alpha_Sigma" 	: [0.05],
					   "N_aug" 			: [1]})

# Load texts
txt_dat = pd.read_csv("../Data/lpi_texts.csv", encoding = "latin-1")

# Load lpi trend data...
trend_dat = pd.read_csv("../Data/lpi_trends.csv")

# Merge by rn
dat = txt_dat.loc[:,["RN", "Title_eng", "AB_eng"]].merge(trend_dat, on = "RN")

# Drop texts which are in the manually verified sample...
holdout = pd.read_csv("../Data/compiled_trend_text1.csv",
						encoding = "latin-1")
holdout_ = dat.loc[dat["RN"].isin(holdout["RN"])].reset_index(drop = True)
dat = dat.loc[~dat["RN"].isin(holdout["RN"])].reset_index(drop = True)

dat = dat.loc[~dat.RN.isin([2587, 2812, 2843])].reset_index(drop = True)

# Load sp and loc data
sp_nms_dat = pd.read_csv("../Data/lpi_txts_sp_nms.csv")
loc_nms_dat = pd.read_csv("../Data/lpi_txts_loc_nms.csv")
# drop na rows 
sp_nms_dat = sp_nms_dat.dropna(subset=['original'])
loc_nms_dat = loc_nms_dat.dropna(subset=['source.string'])

dat = text_prep(dat)
holdout_ = text_prep(holdout_)

# drop texts with AB < 200 characters
dat["AB_len"] = dat.AB.apply(len)
dat["Title_len"] = dat.Title.apply(len)

holdout_["AB_len"] = holdout_.AB.apply(len)
holdout_["Title_len"] = holdout_.Title.apply(len)

dat = dat.loc[dat.AB_len>=200]
dat = dat.reset_index(drop = True)

# Fit specified model
m_fit_apply(dat.copy(), holdout_.copy(), rf_top_spec.loc[0], sp_nms_dat, loc_nms_dat)

