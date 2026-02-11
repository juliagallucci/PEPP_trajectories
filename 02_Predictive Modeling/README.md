# 02_Predictive Modeling
This folder contains Python scripts for predicting depressive and negative symptom trajectory classes in the PEPP dataset using Random Forests, cross-validation, and permutation-based feature importance.

## Script Summaries
1. PEPP_cvmaker.py

Take in cross-validation repetition number and outer fold number then output
test fold subject indices for each outer fold and repetition.

2. PEPP_score_hyperopt_pair.py

Take in features and trajectory labels and, for speed from parallelization, conduct 
one specific iteration of a repeated nested stratified k-fold cross validation 
for paired one-vs-one classification using the selected RF classifier with 
hyperparameters optimized for the performance metric of choice. For LV1 measure.

3. PEPP_analyze_hyperopt_pair.py

Collect iterations for a repeated nested stratified k-fold cross validation 
for paired one-vs-one classification using the selected RF classifier with 
hyperparameters optimized for the performance metric of choice and extract 
classifier performance and feature importance. For LV1 measure.

4. PEPP_score_hyperopt_pair_perm.py

Take in features and trajectory labels, permute trajctory labels, and, for speed 
from parallelization, conduct one specific iteration of a repeated nested 
stratified k-fold cross validation for paired one-vs-one classification using the 
selected RF classifier with hyperparameters optimized for the performance metric 
of choice. For LV1 measure.

5. PEPP_analyze_hyperopt_pair_perm.py

Collect iterations for a permuted repeated nested stratified k-fold cross validation 
for paired one-vs-one classification using the selected RF classifier with 
hyperparameters previously optimized for the performance metric of choice and extract 
one-sided p-values for the classifier performance and feature importance. For LV1 measure.

6. neg_PEPP_score_hyperopt_pair.py

Take in features and trajectory labels and, for speed from parallelization, conduct 
one specific iteration of a repeated nested stratified k-fold cross validation 
for paired one-vs-one classification using the selected RF classifier with 
hyperparameters optimized for the performance metric of choice. For SANS measure.

7. neg_PEPP_analyze_hyperopt_pair.py

Collect iterations for a repeated nested stratified k-fold cross validation 
for paired one-vs-one classification using the selected RF classifier with 
hyperparameters optimized for the performance metric of choice and extract 
classifier performance and feature importance. For SANS measure.

8. neg_PEPP_score_hyperopt_pair_perm.py

Take in features and trajectory labels, permute trajctory labels, and, for speed 
from parallelization, conduct one specific iteration of a repeated nested 
stratified k-fold cross validation for paired one-vs-one classification using the 
selected RF classifier with hyperparameters optimized for the performance metric 
of choice. For SANS measure.

9. neg_PEPP_analyze_hyperopt_pair_perm.py

Collect iterations for a permuted repeated nested stratified k-fold cross validation 
for paired one-vs-one classification using the selected RF classifier with 
hyperparameters previously optimized for the performance metric of choice and extract 
one-sided p-values for the classifier performance and feature importance. For SANS measure.
