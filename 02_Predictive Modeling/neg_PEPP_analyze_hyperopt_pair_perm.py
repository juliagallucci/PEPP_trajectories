# -*- coding: utf-8 -*-
"""

Collect iterations for a permuted repeated nested stratified k-fold cross validation 
for paired one-vs-one classification using the selected RF classifier with 
hyperparameters previously optimized for the performance metric of choice and extract 
one-sided p-values for the classifier performance and feature importance. For SANS measure.

Usage: 
    neg_PEPP_score_hyperopt_pair_perm.py <curr_cat> <classifier> <scorfunc> <rep_size> <outer_size> <inner_size> <nperm> 
    
Arguments:
    
    <curr_cat> Current positive category
    <classifier> Classifier
    <scorfunc> Scoring function performance metric for hyperopt
    <rep_size> Number of repetitions
    <outer_size> Number of outer folds
    <inner_size> Number of inner folds
    <nperm> Number of permutations

"""

import time, os, random
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, average_precision_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from docopt import docopt

#Read in data to generate mean summary scores and feature importance across folds.
def score_and_feat(nrk,nfeat,nrep,outer_k,curr_cat,rk_labs,xlabs,permpath,permlab):

    #Collect the summary scores.
    scorelabs = ['accuracy','balanced_accuracy',
                'f1','f1_weighted',
                'roc_auc','prc_auc']
    nscores = len(scorelabs)
    allscore_collect = pd.DataFrame(np.zeros((nrk,nscores)),index=rk_labs,columns=scorelabs)

    #Collect the feature importances.
    feat_collect = pd.DataFrame(np.zeros((nrk,nfeat)),index=rk_labs,columns=xlabs)

    #Fill in collectors.
    for ridx in range(nrep):
        for outidx in range(outer_k):
            rk_lab = 'r'+str(ridx+1)+'_f'+str(outidx+1)

            #Open store.
            infile = (permpath+'/'+curr_cat+'_pair_'+rk_lab+permlab+'.h5')
            instore = pd.HDFStore(infile,'r')

            #Extract true, predicted, and predicted probability labels.
            inlab = 'y_true'
            inkey = ('/'+inlab+'_'+rk_lab)
            y_true = instore.select(inkey)
            inlab = 'y_predict'
            inkey = ('/'+inlab+'_'+rk_lab)
            y_predict = instore.select(inkey)
            inlab = 'y_proba'
            inkey = ('/'+inlab+'_'+rk_lab)
            y_proba = instore.select(inkey)

            #Accuracy versions.
            acc = accuracy_score(y_true,y_predict)
            balacc = balanced_accuracy_score(y_true,y_predict)

            #F1 versions.
            f1 = f1_score(y_true,y_predict)
            f1_weighted = f1_score(y_true,y_predict,average='weighted')

            #ROC versions.
            try:
                roc_auc = roc_auc_score(y_true,y_proba.iloc[:,1])
            except:
                roc_auc = 0

            #PRC versions.
            prc_auc = average_precision_score(y_true,y_proba.iloc[:,1])

            #Put together scores and append.
            allscore = pd.Series([acc,balacc,
                                f1,f1_weighted,
                                roc_auc,prc_auc],
                                index=['accuracy','balanced_accuracy',
                                    'f1','f1_weighted',
                                    'roc_auc','prc_auc'])
            allscore_collect.loc[rk_lab,allscore.index] = allscore.values

            #Extract feature importance and append.
            inlab = 'featimp'
            inkey = ('/'+inlab+'_'+rk_lab)
            inmat = instore.select(inkey)
            feat_collect.loc[rk_lab,inmat.index] = inmat.values

            #Close.
            instore.close()

    #Average summary scores and feature importance.
    meanscore = allscore_collect.mean(axis=0)
    meanfeat = feat_collect.mean(axis=0)

    #Return the summary scores and feature importance.
    return meanscore, meanfeat

#One-sided significance for features.
def onep_feat(cfeat,cperm):

    #Extract.
    [nperm,nfeat] = np.shape(cperm)
    featlabs = cfeat.index

    #Go through columns.
    feat_pval = pd.DataFrame(np.zeros((1,nfeat)))
    feat_pval.columns = featlabs
    for fidx in range(nfeat):

        #Extract.
        featlab = featlabs[fidx]
        onetruefeat = cfeat.loc[featlab]
        onepermfeat = cperm.loc[:,featlab]

        #If positive or negative.
        if onetruefeat >= 0:
            permsig = ((sum(onepermfeat >= onetruefeat)) + 1)/(nperm + 1)   
        else:
            permsig = ((sum(onepermfeat <= onetruefeat)) + 1)/(nperm + 1)
            
        #Append.
        feat_pval.loc[0,featlab] = permsig
    
    #Return p-values.
    return feat_pval

#Set classifier and hyperparameter optimization method.
args = docopt(__doc__)
curr_cat = args['<curr_cat>']
classifier = args['<classifier>']
scorfunc = args['<scorfunc>']
rep_size = args['<rep_size>']
outer_size = args['<outer_size>']
inner_size = args['<inner_size>']
nperm = args['<nperm>']
print(curr_cat,classifier,scorfunc,rep_size,outer_size,inner_size,nperm)

#Set paths.
basepath = ('pair_neg_'+classifier+'_'+scorfunc+'_r'+rep_size+'_fo'+outer_size+'_fi'+inner_size)
outpath = (basepath+'/gather')
os.makedirs(outpath,exist_ok=True)

#Set main seed and set numeric arguments including hyperopt evaluations, current 
#paired categories, CV repetitions, CV outer fold number, and CV inner fold number.
fullseed = 12345
nevals = 500
[ccat1,ccat2] = [int(x) for x in curr_cat.split('v')]
nrep = int(rep_size)
outer_k = int(outer_size)
inner_k = int(inner_size)

#Read in the input matrix.
infile = '/projects/jgallucci/project_3/PEPP_SANS_XY.csv'
inmat = pd.read_csv(infile)
tr_labels = inmat['class_sans3']

#Map trajectories to indices for input into the classifier later.
label_mapping = {
   'Low' : 0,
   'Remitting' : 1,
   'Persistent-High' : 2,
}
tr_idx = tr_labels.map(label_mapping)

#Produce labels and extract dimensions.
ylabs = ['class_sans3'] 
xlabs = [x for x in inmat.columns if x not in ylabs]
nfeat = len(xlabs)
ycat = ['Low','Remitting','Persistent-High']
ncat = len(ycat)

#Divide input matrix into X features and Y label. First value is zero category,
#second value is positive category. Extract only the relevant subjects. Reset indices.
data_X = inmat.loc[:,xlabs]
data_Y = tr_idx.loc[(tr_idx==ccat1)|(tr_idx==ccat2)]
label_mapping = {
    ccat1: 0,
    ccat2: 1,
}
data_Y = data_Y.map(label_mapping)
data_X = data_X.loc[data_Y.index,:]
data_Y.reset_index(drop=True,inplace=True)
data_X.reset_index(drop=True,inplace=True)

#Set discrete and continuous variables of interest.
disc_x = ['gender','Vmin','NEET','INDPT','REL','mode','dx_0']
cont_x = ['ageonset','dui','ageentry','SOFAS_0','SAPS_0','HAS_0']

#Impute discrete variables.
data_X_disc = data_X.loc[:,disc_x]
imp_disc = SimpleImputer(strategy='most_frequent')
data_X_disc_imp = pd.DataFrame(
    imp_disc.fit_transform(data_X_disc),
    columns=disc_x,
    index=data_X_disc.index
)

#Encode discrete variables.
enc_disc = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
data_X_disc_enc = pd.DataFrame(
    enc_disc.fit_transform(data_X_disc_imp),
    columns=enc_disc.get_feature_names_out(disc_x),
    index=data_X_disc_imp.index
)

#Impute the continuous variables.
data_X_cont = data_X.loc[:,cont_x]
imp_cont = IterativeImputer(random_state=12345,estimator=RandomForestRegressor(),
                            max_iter=1)
data_X_cont_imp = pd.DataFrame(
    imp_cont.fit_transform(data_X_cont),
    columns=cont_x,
    index=data_X_cont.index
)

#Combine continuous and discrete back together.
data_X = pd.concat([data_X_cont_imp,data_X_disc_enc],axis=1)

#Redefine labels.
xlabs = data_X.columns.tolist()
nfeat = len(xlabs)

#Define repetition and outer fold labels.
rk_labs = []
for ridx in range(nrep):
    for outidx in range(outer_k):
        rk_labs.append('r'+str(ridx+1)+'_f'+str(outidx+1))
nrk = len(rk_labs)

#Collect the real summary scores and features.
permpath = basepath
permlab = ''
truescore, truefeat = score_and_feat(nrk,nfeat,nrep,outer_k,curr_cat,rk_labs,xlabs,permpath,permlab)

#Go through each permutation and collect the summary scores and features.
permscore = np.zeros((int(nperm),truescore.shape[0]))
permfeat = np.zeros((int(nperm),truefeat.shape[0]))
for pidx in range(int(nperm)):
    permidx = str(pidx+1)
    print(permidx)
    permpath = (basepath+'/perm')
    permlab = '_p'+permidx
    cscore, cfeat = score_and_feat(nrk,nfeat,nrep,outer_k,curr_cat,rk_labs,xlabs,permpath,permlab)
    permscore[pidx,:] = cscore.values
    permfeat[pidx,:] = cfeat.values

#Add labels.
scorelabs = ['accuracy','balanced_accuracy',
            'f1','f1_weighted',
            'roc_auc','prc_auc']
permscore = pd.DataFrame(permscore,columns=scorelabs)
permfeat = pd.DataFrame(permfeat,columns=xlabs)

#Find the one-sided permutation p-values.
score_OneP = permscore.ge(truescore,axis=1).sum(axis=0).add(1).div((int(nperm)+1))
feat_OneP = onep_feat(truefeat,permfeat)

#Save the permutations and p-values.
outfile = (outpath+curr_cat+'_perm_summary_scores.csv')
permscore.to_csv(outfile,index=False)
outfile = (outpath+curr_cat+'_perm_feature.csv')
permfeat.to_csv(outfile,index=False)
outfile = (outpath+curr_cat+'_summary_scores_OneP.csv')
score_OneP.to_csv(outfile,index=True,header=False)
outfile = (outpath+curr_cat+'_feature_OneP.csv')
feat_OneP.T.to_csv(outfile,index=True,header=False)
print('Permutation done.')
