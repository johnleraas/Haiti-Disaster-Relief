# Project Overview

This project was done as part of a class on machine learning and used actual satellite data on Haiti following the earthquake in 2010. 

The primary learning outcomes were (i) gaining familiarity with and hyperparameter tuning of different ML models, (ii) comparison of different models and understanding model capacity, and (iii) thinking through the objective function in detail. Notably, very little instruction was provided and certain assumptions and considerations are discussed below.

This project was coded in R. <ins> See the full knit HTML here:</ins> [Haiti Disaster Relief Project](https://github.com/johnleraas/Haiti-Disaster-Relief/blob/main/Haiti_Disaster_Relief.html)

# Introduction
On January 12, 2010 a 7.0 magnitude earthquake occurred in Haiti 16 miles west of its capital, Port-au-Prince. An estimated three million people were affected by the earthquake. Following the earthquake many Haitians were in desperate need of food, water, and other necessary supplies however infrastructure was significantly impacted making it difficult to locate those affected.

It was known that people were creating temporary shelters using blue tarps after their homes had been destroyed by the earthquake. It was also known that these tarps would be a good indicator of the location of displaced persons. A team from the Rochester Institute of Technology flew aircraft to collect high resolution, geo-referenced imagery, however there was limited time for humans to visually inspect each photograph to identify each blue tarp.

The goal of this project is to find an algorithm that can effectively search the images in order to locate displaced persons. The location would then be communicated to rescue workers so that they can help those in need in time. In this aim, several different models were trained, compared, and evaluated.

# Exploratory Data Analysis

The data took the form of RBG pixel values with class labels. The "Blue Tarps" class represented 3.3% of observations.

<p align="center">
  <img width = 45% height = 45% src="https://github.com/johnleraas/Haiti-Disaster-Relief/blob/main/Docs/Pairwise_FullDataSet.png">
  <img width = 45% height = 45% src="https://github.com/johnleraas/Haiti-Disaster-Relief/blob/main/Docs/Pairwise_BlueTarps.png">
</p>

By exploring the relationship between different variables using the pairwise plots for both the full data set and only the “Blue Tarp” values, it can be observed that:

* There is generally (full data set) a positive relationship between Red, Blue, and Green. This intuitively can be interpreted as brightness.
* Looking only at Blue Tarp data, there remains a positive relationship, however Red and Green appear to possess a linear relationship while Blue is skewed above both Red and Green. This intuitively can be interpreted as observing the blue color, or “blueness”, as associated with a “Blue Tarp”.
* It appears beneficial to reclassify the Classes to “found” blue tarp and “not found” blue tarp (1 and 0).

Basic logistic regression was also performed as part of the EDA process. Performing basic logistic regression using the three predictors (Red, Green, Blue), it was found that complete separation occurs within the data. This indicates that Blue Tarps are well separated from other Classes, though logistic regression may have difficulty in fitting coefficients.

Additionally, the Blue feature has a positive coefficient, while Red and Green are negative. This is intuitive, as a blue observation is likely to indicate a Blue Tarp. Additionally the magnitude of the Blue coefficient is roughly the magnitude of Red plus Green.

It was previously observed that “blueness” is a strong indicator of a Blue Tarp. While transformations were considered to represent this observation, such as (i) Blue above the max of Red or Green; (ii) Blue above the average of Red and Green; and (iii) Subtracting the average and/or minimum of all three colors from each value of Red, Blue, and Green (to effectively account for brightness); the basic logistic regression fit indicates that these transformations may be unnecessary.

# Model Overview & Setup

The following models are evaluated using 10-fold cross validation:

* Logistic Regression
* LDA (Linear Discriminant Analysis)
* QDA (Quadratic Discriminant Analysis)
* KNN (K-nearest neighbors)
* Penalized Logistic Regression
* Random Forest
* Support Vector Machine

## Model Evaluation

### Evaluation Context
Given the context of the problem, identifying Blue Tarps (displaced people) is of extremely high importance. This translates to the ability to save lives and prevent suffering. The obvious objective is to maximize the true positive rate within real world constraints. As the threshold is adjusted to increase the number of true positives, the number of false positives will increase which will have some associated cost.

During the training phase, I assume that a precision of approximately 80% is acceptable, or roughly 1 in 5 predicted positives being false. A loss function was created and fit empirically to generally align with these assumptions. In practice, the validity of this assumption would depend on implementation and require additional context. If someone can quickly check an appropriate aerial photograph before sending supplies, the cost of a false positive is minimal. However, if supplies are airdropped or otherwise transported to each predicted positive, the cost of a false positive is obviously higher.

In balancing the true positive rate (predicted positives / total positive observations) to catch every blue tarp with precision (the percentage of positive predictions that are actually right), not only did I evaluate absolute values, but also where significant trade-offs are made between the two (i.e. the derivatives of each).

### Evaluation Process
Most models were evaluated with a two step evaluation process (though while a threshold was selected for each model, not every model possessed another tuning parameter). The first step involves optimizing a tuning parameter/set of tuning parameters and the second involves setting a threshold to balance the true positive rate with the precision. The first step in model evaluation focuses largely on AUROC which measures the balance of sensitivity/true positive rate with specificity/false positive rate (FPR = 1 - specificity). A high false positive rate will decrease the precision (TP / TP + FP). The rationale for focusing on AUROC is simply that if a value of 1 is obtained, the model can achieve a true positive rate of 1 and a false positive rate of 0.

The first step, tuning parameter selection, was performed using cross validation and a threshold of 0.5 (optimizing accuracy). Again, while accuracy was observed, the primary metric used for evaluation was AUROC. After tuning parameters for a model were selected, the threshold was selected using cross validation. This is partially subjective with the primary considerations being: true positive rate, precision, and the derivative of both values. A loss function was also created and used as part of the decision making process, though not the entirety of the process. This function was fit empirically to generally align with the expected objectives and constraints highlighted previously (maximizing TPR while maintaining a precision of 80%) for the better performing models (specifically omitting LDA and QDA). The loss function was defined simply as:

__Loss=FalsePositives+5∗FalseNegatives__

Additionally, once an optimal threshold value was selected, the tuning parameters were re-evaluated using this value, rather than 0.5. This did not return any significantly differing results and in the interest of brevity the results are not presented below.

### Cross Validation
10-fold cross validation was implemented to measure AUROC, Accuracy, True Positive Rate, False Positive Rate, Precision, and the previously described Loss Function. For a given tuning parameter or threshold, the out of sample predictions were aggregated for each fold and the associated calculations/measurements were performed on the single aggregated data set. This is equivalent to a weighted average (based on number of observations) applied to each measure. <ins>Notably, each model utilizes the same folds in the cross validation process.</ins>

# Model Training
Grid search processes were employed to find the optimal hyperparameters for each model.



<p align="center">
  <img width = 75% height = 75% src="https://github.com/johnleraas/Haiti-Disaster-Relief/blob/main/Docs/Model_Summary_Table.png">
</p>


<p align="center">
  <img width = 45% height = 45% src="https://github.com/johnleraas/Haiti-Disaster-Relief/blob/main/Docs/ROC_split1.png">
  <img width = 45% height = 45% src="https://github.com/johnleraas/Haiti-Disaster-Relief/blob/main/Docs/ROC_split2.png">
</p>


<p align="center">
  <img width = 45% height = 45% src="https://github.com/johnleraas/Haiti-Disaster-Relief/blob/main/Docs/HoldOut_Pairwise_FullDataSet.png">
  <img width = 45% height = 45% src="https://github.com/johnleraas/Haiti-Disaster-Relief/blob/main/Docs/HoldOut_Pairwise_BlueTarps.png">
</p>

<p align="center">
  <img width = 75% height = 75% src="https://github.com/johnleraas/Haiti-Disaster-Relief/blob/main/Docs/HoldOut_Model_Summary_Table.png">
</p>
