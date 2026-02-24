# 01_Latent Class & Growth Mixture Modeling
This folder contains R Markdown scripts for analyzing latent class and growth mixture models in the PEPP dataset. The analyses focus on identifying symptom trajectories: general depressive (class_lv1_3) and negative (class_sans3) symptom classes.

## Script Summaries
- gmms_final.Rmd This script performs growth mixture modeling using the lcmm package in R to identify and evaluate latent longitudinal trajectories of general depressive (LV1) and negative (SANS) symptoms over 18 months, comparing model fit, classification quality (entropy, APP), and conducting bootstrap likelihood ratio tests.
- lcga_final.Rmd This script performs latent class growth analysis using the lcmm package in R to identify and evaluate latent longitudinal trajectories of general depressive (LV1) and negative (SANS) symptoms over 18 months, comparing model fit, classification quality (entropy, APP), and conducting bootstrap likelihood ratio tests.
