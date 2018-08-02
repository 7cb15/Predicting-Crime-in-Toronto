# Predicting-Crime-in-Toronto
Using data sourced from the Toronto Police (http://data.torontopolice.on.ca/pages/open-data), I construct a multi-class classification model using a Random Forest classifier to predict the type of major crime committed based on time of day, neighbourhood, division, year, month, etc. The dataset includes every major crime committed from 2014-2017* in the city of Toronto, with detailed information about the location and time of offence. The data contains only categorical variables so the modeling process tests both numeric encoding and OneHot encoding, with some improvement with the latter approach. 

The model performs reasonably well on F1-score (precision and recall) for a five-class classification problem. Though the data set is somewhat unbalanced towards assaults (higher volume), balancing class weights does not materially impact model performance.

*excludes sexual assaults / sexual crimes, which are unpublished, and homicides which are not published at the same level of detail.
