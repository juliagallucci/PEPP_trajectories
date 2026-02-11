# -*- coding: utf-8 -*-
"""

Take in features and trajectory labels and, for speed from parallelization, conduct 
one specific iteration of a repeated nested stratified k-fold cross validation 
for paired one-vs-one classification using the selected RF classifier with 
hyperparameters optimized for the performance metric of choice. For SANS measure.

Usage: 
    neg_PEPP_score_hyperopt_pair.py <curr_cat> <classifier> <scorfunc> <rep_size> <outer_size> <inner_size> <replab> <outerlab>
    
Arguments:
    
    <curr_cat> Current positive category
    <classifier> Classifier
    <scorfunc> Scoring function performance metric for hyperopt
    <rep_size> Number of repetitions
    <outer_size> Number of outer folds
    <inner_size> Number of inner folds
    <replab> Current repetition
    <outerlab> Current fold

"""

import time, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from hyperopt import fmin, tpe, hp
from docopt import docopt

#Set current category pairs being considered, classifier, metric to optimize hyperparameters, 
#CV parameters, and current CV repetition and outer fold.
args = docopt(__doc__)
curr_cat = args['<curr_cat>']
classifier = args['<classifier>']
scorfunc = args['<scorfunc>']
rep_size = args['<rep_size>']
outer_size = args['<outer_size>']
inner_size = args['<inner_size>']
replab = args['<replab>']
outerlab = args['<outerlab>']
print(curr_cat,classifier,scorfunc,rep_size,outer_size,inner_size,replab,outerlab)

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

#Set constant parameters for RF.
if classifier == 'RF':

    #Number of trees.
    ntrees = 500

#Produce counts.
nsample = data_Y.shape[0]

#Set up outer CV train-test indices for every repetition. Specifically, give
#a label to test indices for each outer CV iteration for each repetition.
outercv_test = []

#Set the seed to initialize. While we don't have all the repetition indices,
#run the loop.
np.random.seed(fullseed)
while len(outercv_test) < nrep:

    #Sample to get a new seed to input.
    repseed = np.random.randint(1,12345)

    #Set up outer CV generator from this seed.
    outer_kf = StratifiedKFold(n_splits=outer_k,shuffle=True,random_state=repseed)

    #Pull out the indices.
    allidx = np.zeros((nsample))
    for outidx, (_,test_idx) in enumerate(outer_kf.split(data_X,data_Y)):
        allidx[test_idx] = outidx + 1
    
    #If the indices don't exist in the list yet, append.
    idx_exist = any(np.array_equal(allidx,arr) for arr in outercv_test)
    if not idx_exist:
        outercv_test.append(allidx)
    else:
        print('Indices exist.')

#Convert to array.
outercv_test = np.array(outercv_test).T

#Set number of test and train samples for later use with hyperparameters.
ntest = int(np.ceil(nsample/outer_k))
ntrain = nsample - ntest

#Define repetition and outer fold labels.
rk_labs = []
for ridx in range(nrep):
    for outidx in range(outer_k):
        rk_labs.append('r'+str(ridx+1)+'_f'+str(outidx+1))
nrk = len(rk_labs)

#Define all CV repetition seeds from the main seed for use in random processes.
np.random.seed(fullseed)
repcv_list = np.random.randint(1,12345,nrep).tolist()

#Start this repetition.
start1 = time.time()

#Set current CV repetition and outer CV iteration from the labels.
ridx = int(replab) - 1
outidx = int(outerlab) - 1

#Set the seed for this CV repetition for use in random processes.
repseed = repcv_list[ridx]

#Define outer CV seeds from the CV repetition seed for use in random processes.
np.random.seed(repseed)
outcv_list = np.random.randint(1,12345,outer_k).tolist()

#Extract outer CV subject indices for this repetition.
outercollect = outercv_test[:,ridx]

#Label the current iteration.
rk_lab = 'r'+str(ridx+1)+'_f'+str(outidx+1)
print(rk_lab)

#Set the seed for this outer CV iteration for use in random processes.
outerseed = outcv_list[outidx]

#Extract train and test subject indices.
train_index = (np.where(outercollect!=(outidx+1))[0]).tolist()
test_index = (np.where(outercollect==(outidx+1))[0]).tolist()

#Extract train and test X and Y.
X_train, X_test = data_X.iloc[train_index,:], data_X.iloc[test_index,:]
Y_train, Y_test = data_Y.iloc[train_index], data_Y.iloc[test_index]

#Set discrete and continuous variables of interest.
disc_x = ['gender','Vmin','NEET','INDPT','REL','mode','dx_0']
cont_x = ['ageonset','dui','ageentry','SOFAS_0','SAPS_0','HAS_0']

#Impute discrete variables for train and test.
X_train_disc = X_train.loc[:,disc_x]
X_test_disc = X_test.loc[:,disc_x]
imp_disc = SimpleImputer(strategy='most_frequent')
X_train_disc_imp = pd.DataFrame(
    imp_disc.fit_transform(X_train_disc),
    columns=disc_x,
    index=X_train_disc.index
)
X_test_disc_imp = pd.DataFrame(
    imp_disc.transform(X_test_disc),
    columns=disc_x,
    index=X_test_disc.index,
)

#Encode discrete variables for train and test.
enc_disc = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
X_train_disc_enc = pd.DataFrame(
    enc_disc.fit_transform(X_train_disc_imp),
    columns=enc_disc.get_feature_names_out(disc_x),
    index=X_train_disc_imp.index
)
X_test_disc_enc = pd.DataFrame(
    enc_disc.transform(X_test_disc_imp),
    columns=enc_disc.get_feature_names_out(disc_x),
    index=X_test_disc_imp.index
)

#Impute the continuous variables for train and test.
X_train_cont = X_train.loc[:,cont_x]
X_test_cont = X_test.loc[:,cont_x]
imp_cont = IterativeImputer(random_state=12345,estimator=RandomForestRegressor(),
                            max_iter=500)
X_train_cont_imp = pd.DataFrame(
    imp_cont.fit_transform(X_train_cont),
    columns=cont_x,
    index=X_train_cont.index
)
X_test_cont_imp = pd.DataFrame(
    imp_cont.transform(X_test_cont),
    columns=cont_x,
    index=X_test_cont.index
)

#Combine continuous and discrete back together.
X_train = pd.concat([X_train_cont_imp,X_train_disc_enc],axis=1)
X_test = pd.concat([X_test_cont_imp,X_test_disc_enc],axis=1)

#Initialize inner CV for hyperparameter optimization.
inner_kf = StratifiedKFold(n_splits=inner_k,shuffle=True,random_state=outerseed)

#Random forest hyperparameter optimization.
if classifier == 'RF':

    #Define a RF function for scoring for hyperopt, which we want to minimize.
    def rf_cv_score(params,outerseed=outerseed,inner_kf=inner_kf,X_train=X_train,Y_train=Y_train):

        #Gets hyperparameters.
        params = {'criterion': params['criterion'],
                'class_weight': params['class_weight'],
                'max_depth': params['max_depth'], 
                'max_features': params['max_features'],
                'min_samples_leaf': params['min_samples_leaf'],
                'min_samples_split': params['min_samples_split']
                }
        
        #Use these hyperparameters with classifier.
        clf = RandomForestClassifier(random_state=outerseed,n_estimators=ntrees,**params)
    
        #Conduct inner CV and retrieve the score which we want to minimize.
        cv_score = -cross_val_score(clf,X_train,Y_train,cv=inner_kf,scoring=scorfunc).mean()
        return cv_score
    
    #Define space of hyperparameters we want to explore.
    nfeat_slice = X_train.shape[1]
    criterion_list = ['gini','entropy','log_loss']
    class_weight_list = [None,'balanced','balanced_subsample']
    max_depth_list = [None] + [int(np.round(x)) for x in (np.linspace(0.1,1.0,10) * ntrain)]
    max_depth_list = [i for i in max_depth_list if i != 0]
    max_depth_list = list(dict.fromkeys(max_depth_list))
    max_features_list = ['sqrt',1] + [int(np.round(x)) for x in (np.linspace(0.1,1.0,10) * nfeat_slice)]
    max_features_list = [i for i in max_features_list if i != 0]
    max_features_list = list(dict.fromkeys(max_features_list))
    min_samples_leaf_list = [1] + [int(np.round(x)) for x in (np.linspace(0.1,1.0,10) * ntrain)]
    min_samples_leaf_list = [i for i in min_samples_leaf_list if i != 0]
    min_samples_leaf_list = list(dict.fromkeys(min_samples_leaf_list))
    min_samples_split_list = [2] + [int(np.round(x)) for x in (np.linspace(0.1,1.0,10) * ntrain)]
    min_samples_split_list = [i for i in min_samples_split_list if i != 0]
    min_samples_split_list = list(dict.fromkeys(min_samples_split_list))
    space = {'criterion' : hp.choice('criterion',criterion_list),
            'class_weight': hp.choice('class_weight',class_weight_list),
            'max_depth': hp.choice('max_depth',max_depth_list),
            'max_features': hp.choice('max_features',max_features_list),
            'min_samples_leaf': hp.choice('min_samples_leaf',min_samples_leaf_list),
            'min_samples_split': hp.choice('min_samples_split',min_samples_split_list)
            }

    #Fit minimizer for best hyperparameters, selecting TPE algorithm.
    best_min = fmin(fn=rf_cv_score,
                space=space, 
                algo=tpe.suggest,
                max_evals=int(nevals),
                rstate=np.random.default_rng(outerseed),
                return_argmin=False)
    
    #Produce classifier with the best hyperparameters.
    chyper = RandomForestClassifier(random_state=outerseed,
                                    n_estimators=ntrees,
                                    criterion=best_min['criterion'],
                                    class_weight=best_min['class_weight'],
                                    max_depth=best_min['max_depth'],
                                    max_features=best_min['max_features'],
                                    min_samples_leaf=best_min['min_samples_leaf'],
                                    min_samples_split=best_min['min_samples_split'])

#Fit the classifier with the best hyperparameters.
chyper.fit(X_train,Y_train)

#Generate best hyperparameters.
best_hyper = pd.Series(best_min.values(),index=best_min.keys())

#Generate predicted labels and predicted label probabilities.
Y_test_predict = pd.Series(chyper.predict(X_test),index=Y_test.index)
Y_test_proba = pd.DataFrame(chyper.predict_proba(X_test),index=Y_test.index)

#Generate impurity-based feature importance only, for now.
if classifier == 'RF':
    featimp = pd.Series(chyper.feature_importances_,index=X_train.columns)

#Set output path.
outpath = ('pair_neg_'+classifier+'_'+scorfunc+'_r'+rep_size+'_fo'+outer_size+'_fi'+inner_size+'/')
os.makedirs(outpath,exist_ok=True)

#Save everything into h5 file to store.
outfile = (outpath+curr_cat+'_pair_'+rk_lab+'.h5')
savelist = [Y_test,Y_test_predict,Y_test_proba,best_hyper,featimp,X_train,X_test]
savelabs = ['y_true','y_predict','y_proba','hyper','featimp','x_train','x_test']
nsave = len(savelist)
for saidx in range(nsave):
    savestore = pd.HDFStore(outfile)
    savemat = savelist[saidx]
    savelab = savelabs[saidx]
    savekey = ('/'+savelab+'_'+rk_lab)
    savestore.put(savekey,savemat)
    savestore.close()

#Display time.
end1 = time.time()
print('Iteration done:',end1-start1)
