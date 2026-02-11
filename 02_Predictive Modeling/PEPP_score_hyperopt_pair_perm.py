# -*- coding: utf-8 -*-
"""

Take in features and trajectory labels, permute trajctory labels, and, for speed 
from parallelization, conduct one specific iteration of a repeated nested 
stratified k-fold cross validation for paired one-vs-one classification using the 
selected RF classifier with hyperparameters optimized for the performance metric 
of choice. For LV1 measure.

Usage: 
    PEPP_score_hyperopt_pair_perm.py <curr_cat> <classifier> <scorfunc> <rep_size> <outer_size> <inner_size> <replab> <outerlab> <nperm> <permidx>
    
Arguments:
    
    <curr_cat> Current positive category
    <classifier> Classifier
    <scorfunc> Scoring function performance metric for hyperopt
    <rep_size> Number of repetitions
    <outer_size> Number of outer folds
    <inner_size> Number of inner folds
    <replab> Current repetition
    <outerlab> Current fold
    <nperm> Number of permutations
    <permidx> Current permutation index

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

#Set classifier and hyperparameter optimization method.
args = docopt(__doc__)
curr_cat = args['<curr_cat>']
classifier = args['<classifier>']
scorfunc = args['<scorfunc>']
rep_size = args['<rep_size>']
outer_size = args['<outer_size>']
inner_size = args['<inner_size>']
replab = args['<replab>']
outerlab = args['<outerlab>']
nperm = args['<nperm>']
permidx = args['<permidx>']
print(curr_cat,classifier,scorfunc,rep_size,outer_size,inner_size,replab,outerlab,nperm,permidx)

#Set main seed and set numeric arguments including hyperopt evaluations, current 
#paired categories, CV repetitions, CV outer fold number, and CV inner fold number.
fullseed = 12345
nevals = 500
[ccat1,ccat2] = [int(x) for x in curr_cat.split('v')]
nrep = int(rep_size)
outer_k = int(outer_size)
inner_k = int(inner_size)

#Read in the input matrix.
infile = '/projects/jgallucci/project_3/PEPP_LV1_XY.csv'
inmat = pd.read_csv(infile)
tr_labels = inmat['class_lv1_3']

#Map trajectories to indices for input into the classifier later.
label_mapping = {
   'Low' : 0,
   'Remitting' : 1,
   'Persistent-High' : 2,
}
tr_idx = tr_labels.map(label_mapping)

#Produce labels and extract dimensions.
ylabs = ['class_lv1_3'] 
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

#Set constant parameters.
if classifier == 'RF':

    #Number of trees. As high as you can go is ok. 
    #Talked about in class.
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

#Set number of test and train samples.
ntest = int(np.ceil(nsample/outer_k))
ntrain = nsample - ntest

#Define repetition and outer fold labels.
rk_labs = []
for ridx in range(nrep):
    for outidx in range(outer_k):
        rk_labs.append('r'+str(ridx+1)+'_f'+str(outidx+1))
nrk = len(rk_labs)

#Define all CV repetition seeds from the full seed.
np.random.seed(fullseed)
repcv_list = np.random.randint(1,12345,nrep).tolist()

#Go through each repetition.
start1 = time.time()

#Set current CV repetition and outer CV iteration from the labels.
ridx = int(replab) - 1
outidx = int(outerlab) - 1

#Set the seed for this CV repetition.
repseed = repcv_list[ridx]

#Define outer CV seeds from the CV repetition seed.
np.random.seed(repseed)
outcv_list = np.random.randint(1,12345,outer_k).tolist()

#Extract outer CV indices for this repetition.
outercollect = outercv_test[:,ridx]

#Go through outer CV iterations.
rk_lab = 'r'+str(ridx+1)+'_f'+str(outidx+1)
print(rk_lab)

#Read in the hyperparameters from the fitted model on the true data.
inpath = ('pair_'+classifier+'_'+scorfunc+'_r'+rep_size+'_fo'+outer_size+'_fi'+inner_size+'/')
savelab = 'hyper'
infile = (inpath+curr_cat+'_pair_'+rk_lab+'.h5')
savestore = pd.HDFStore(infile,'r')
savekey = ('/'+savelab+'_'+rk_lab)
best_min = savestore.select(savekey)
savestore.close()

#Set the seed for this outer CV iteration.
outerseed = outcv_list[outidx]

#Extract indices.
train_index = (np.where(outercollect!=(outidx+1))[0]).tolist()
test_index = (np.where(outercollect==(outidx+1))[0]).tolist()

#Generate permutation indices for all the permutations.
permseed = 12345
random.seed(permseed)
lorig = tuple(range(nsample))
lnum = list(range(nsample))
pset = set()
pset.add(tuple(lnum))
while len(pset) < (int(nperm)+1):
    random.shuffle(lnum)
    pset.add(tuple(lnum))
pset = list(pset)  
pset.remove(lorig)

#Define permutation set for the current permutation.
oneset = list(pset[(int(permidx)-1)])

#Extract permuted Y and relabel.
perm_Y = data_Y.iloc[oneset]
perm_Y.index = data_Y.index
Y_train, Y_test = perm_Y.iloc[train_index], perm_Y.iloc[test_index]

#Read train and test X previously already imputed and encoded.
inpath = ('pair_'+classifier+'_'+scorfunc+'_r'+rep_size+'_fo'+outer_size+'_fi'+inner_size+'/')
infile = (inpath+curr_cat+'_pair_'+rk_lab+'.h5')
savestore = pd.HDFStore(infile,'r')
savelab = 'x_train'
savekey = ('/'+savelab+'_'+rk_lab)
X_train = savestore.select(savekey)
savelab = 'x_test'
savekey = ('/'+savelab+'_'+rk_lab)
X_test = savestore.select(savekey)
savestore.close()

#Produce actual classifier with the best hyperparameters.
chyper = RandomForestClassifier(random_state=outerseed,
                                n_estimators=ntrees,
                                criterion=best_min['criterion'],
                                class_weight=best_min['class_weight'],
                                max_depth=best_min['max_depth'],
                                max_features=best_min['max_features'],
                                min_samples_leaf=best_min['min_samples_leaf'],
                                min_samples_split=best_min['min_samples_split'])

#Fit.
chyper.fit(X_train,Y_train)

#Generate predicted labels, predicted label probabilities.
Y_test_predict = pd.Series(chyper.predict(X_test),index=Y_test.index)
Y_test_proba = pd.DataFrame(chyper.predict_proba(X_test),index=Y_test.index)

#Generate impurity-based feature importance.
if classifier == 'RF':
    featimp = pd.Series(chyper.feature_importances_,index=X_train.columns)

#Set output path.
outpath = ('pair_'+classifier+'_'+scorfunc+'_r'+rep_size+'_fo'+outer_size+'_fi'+inner_size+'/perm/')
os.makedirs(outpath,exist_ok=True)

#Save everything into h5 file to store.
outfile = (outpath+curr_cat+'_pair_'+rk_lab+'_p'+permidx+'.h5')
savelist = [Y_test,Y_test_predict,Y_test_proba,featimp]
savelabs = ['y_true','y_predict','y_proba','featimp']
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
print('Permutation done:',end1-start1)
