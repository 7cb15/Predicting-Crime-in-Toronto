# Predicting-Crime-in-Toronto
Using data sourced from the Toronto Police (http://data.torontopolice.on.ca/pages/open-data), I construct a multi-class classification model to predict the type of major crime committed based on time of day, neighbourhood, division, year, month, etc. The dataset includes every major crime committed from 2014-2017* in the city of Toronto, with detailed information about the location and time of offence.

The modeling approach is to use a Decision Tree classifier with AdaBoost for improved accuracy. A RandomForest classifier is also tested to test for improved accuracy.

*excludes sexual assaults / sexual crimes, which are unpublished, and homicides which are not published at the same level of detail.
