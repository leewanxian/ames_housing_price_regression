# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 2: Ames Housing Price Regression<br>
Author: Lee Wan Xian

---
## Executive Summary

The purpose of this project is to create a machine learning (ML) regression model to assist a real estate agency in predicting the property prices in Ames, Iowa. There are 2 objectives in this project report. Firstly, it is to create a ML regression model that can predict property sale prices based on the unique features of the property. Next, it is pinpoint and highlight what are the key features that have significant effect on sale prices.<br> 

This report has 3 Jupyter notebooks in total. Book 1 covers Data Cleaning and Exploratory Data Analysis. Book 2 covers Preprocessing and Features Engineering. Book 3 covers ML Modelling, Conclusion and Recommendation.<br>

This report shows that the best performing ML model is the ElasticNet regression model. To add on, features relating to the neighborhood the property resides in, the building class the proprty falls under and overall condition/quality of the property have significant effect on prices. Thus, making them suitable predictors for predict sale prices.<br>

In conclusion, we recommend that the <font color="blue">ElasticNet regression model</font> should be used for predicting property prices.

---
## Problem Statement

Our client is a real estate agency that wants to predict property prices in the city of Ames, Iowa. With a good prediction on property prices, they would be able to channel their sales and marketing resources to the right customer base who have the means to afford it. The team is tasked to build a regression model that can predict the sale prices based on the property's features. The client would also like to know which key features have significant impact on sale prices.

---
## Data Dictionary
The data dictionary of the train & test datasets is available in the below link.

[Data dictionary](https://www.kaggle.com/competitions/dsi-us-11-project-2-regression-challenge/data)

---
## Data Cleaning Summary

* Missing values were imputed as $0$ or `NA`, based on the nature of the variable
* [Robust Scaling](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html) was used to scale the X variables. That way, we can reduce the impact of outliers affecting the model's performance.

---
## EDA Summary

There are 20 variables considered as the predictor features for this model. Amongst these features, it contains a mix of categorical and numerical variables.

As for the rest of the variables, they are not considered as suitable predictor features due to the below reasons:
1. Some of them have the same data observation value for more than 85% of the train data observations (Lack of variance in observation value)
2. Some of them have datapoints that are too concentrated to each other with respect to the property sale price
3. `garage_yr_blt` have a outlier that would cause more noise to the model. That outlier has been proven to be erroneous. We can exclude this variable as it does not have a linear relationship with sale price.

---
## Modelling Summary

Model|$R_2$ Train|$R_2$ Test|Root Mean Square Error (RMSE)
---|---|---|---
Baseline|n.a.|-0.000238|79061.81
Linear|-2.4471e+18|-2.2705e+22|3.3994e+14
Ridge|0.85299|0.87313|28157.35
Lasso|0.85465|0.87186|28297.80
ElasticNet|0.85338|0.87298|28173.75

Based on the metrics tables above, the linear regression model has performed way worse than the baseline model. Their negative $R_2$ scores deemed that they are very underfitted and the difference in $R_2$ scores and RMSE is huge. 

Among Ridge, Lasso & ElasticNet models, all 3 models do not show signs of underfitting or overfitting. Their $R_2$ scores for both train & test dataframes are close to 1. Their respective $R_2$ train scores are lower than their respective $R_2$ test scores. Hence, all 3 models have a good tradeoff balance between bias and variance. The difference in respective $R_2$ train score & $R_2$ test score is the smallest for both Lasso and ElasticNet models.

Looking at RMSE, the Ridge model is the best out of the 3. RMSE represents the approximate average distance squared of the actual value from the predicted value. Thus the lower the RMSE, the lesser likely the model predicts the sale price too far from the actual sale price.

Hence, we chose ElasticNet regression model as the most suitable model for predicting house prices. Reasons are stated as per below:
1. Second best model in terms of RMSE
2. $R_2$ scores are in a good range (within 0.85 to 0.88)
3. Difference in $R_2$ score for train dataset vs test dataset is the second smallest amongst the models (0.019)
4. $R_2$ score for train dataset is lower than that of test dataset

---
## Conclusion

**Prediction Model**

ElasticNet regression model, where regularization strength value = 43.1 and L1 to L2 penalty = 1.0 (i.e. Lasso penalty), is the most suitable model to use for predicting property prices. This model is able to explain more than 85% of the variability in Sale Price, based on the key features of the property. Reasons are stated as per below:
1. Second best model in terms of RMSE
2. $R_2$ scores are in a good range (within 0.85 to 0.88)
3. Difference in $R_2$ score for train dataset vs test dataset is the second smallest amongst the models (0.019)
4. $R_2$ score for train dataset is lower than that of test dataset

**Key Features**

Relating back to the problem statement, the key features that have the most significant effect on Sale Price are the neighborhood, property area, building class and overall condition and quality of the property.<br> For instance, if the property is located in Green Hills, Stone Brook, Northridge or Northridge Heights neighborhood, it adds more value to the property. To add on, the property area has a positive relationship with the value of the property. Thus, the bigger the property area, the higher the sale price. If the property falls under 2-STORY PUD - 1946 & NEWER or 1-STORY PUD (Planned Unit Development) - 1946 & NEWER building class or located in Old Town, Edwards or Iowa DOT and Rail Road neighborhood, the property sale price would mostly likely be low.

**Future Improvements**

Further improvements can be done on the prediction model by including the below variables as predictor features,
1. Annual income of the buyer as the wealthier buyers are able to tolerate higher prices
2. Macroeconomic factors (i.e. interest rate of home loans)
3. Number of years of lease left for leasehold properties
4. Crime rate of the neighborhood where the property resides

---
## Recommendation

We recommend that the client should use our <font color="blue">ElasticNet regression model</font> for predicting property prices in Ames, Iowa. This model can be generalized to other cities, provided that the categorical features are properly encoded into numerical data beforehand.

The client should take note of <font color="blue">the neighborhood where the property resides and the building class of the property</font> to gauge its sale price. It is also good to note that <font color="blue">the property area and overall condition/quality of property</font> have strong effect on sale prices.

Thus, the client should promote houses located in Green Hills, Stone Brook, Northridge or Northridge Heights neighborhood to high-income buyers. As for low-income buyers, the client can promote houses that fall under 2-STORY PUD - 1946 & NEWER or 1-STORY PUD (Planned Unit Development) - 1946 & NEWER building class.

---
## Python Libraries

* Python = 3.8
* Pandas
* Numpy
* Matplotlib
* Seaborn
* Scikit-learn