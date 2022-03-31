# Project Overview

This project was done as part of a class on machine learning and used actual satellite data on Haiti following the earthquake in 2010. 

The primary learning outcomes were (i) gaining familiarity with and hyperparameter tuning of different ML models, (ii) comparison of different models and understanding model capacity, and (iii) thinking through the objective function in detail. Notably, very little instruction was provided and certain assumptions and considerations are discussed below.

This project was coded in R. <ins> To see the full knit R Markdown, please download it here:</ins> [Haiti Disaster Relief Project](https://github.com/johnleraas/Haiti-Disaster-Relief/blob/main/Haiti_Disaster_Relief.html)

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
Grid search processes were employed to find the optimal hyperparameters for each model. In selecting a model, a threshold was selected as well. As noted previously, the model selection process focused on AUROC. Given the imprecise nature of the loss function within the context of this problem, the threshold was ultimately "hand chosen" with consideration given to the shape of the AUROC curves (number of additional false positives in return for one fewer false negatives).

* __Logistic Regression:__ Logistic regression is typically not associated with a tuning parameter. However, given the initial observations in the exploratory data analysis that tarps correspond to “blueness” and that the relationships may not be nonlinear, higher degree polynomials of Red, Blue, and Green were explored with the polynomial degree acting as the hyperparameter.
* __Linear Discriminant Analysis:__ Linear Discriminant Analysis does not possess any tuning parameters that require optimization. Threshold was selected in order to maximize the true positive rate while balancing precision.
* __Quadratic Discriminant Analysis:__ Similar to LDA, Quadratic Discriminant Analysis does not require the optimization of any tuning parameters. Again, threshold was explored in detail and selected to maximize the true positive rate within some reasonable precision bound.
* __K-Nearest Neighbors:__ K-Nearest Neighbors requires picking an optimal value of k (number of neighbors) using cross validation. Specifically, k is the tuning parameter within KNN. After this was completed, a threshold was selected in order to maximize the true positive rate while maintaining a reasonable precision. Of note each of the k neighbors effectively casts a vote on the prediction, so thresholds were explored corresponding to a number of votes (neighbors with a value “found”).
* __Penalized Logistic Regression:__ Penalized Logistic Regression utilizes two tuning parameters: λ and α. The value of α dictates the elastic net penalty. A value of α = 0 corresponds to Ridge Regression, a values of α = 1 corresponds to Lasso Regression, and values of 0 < α = 0 < 1 correspond to Elastic Net regression. Values of α = 0, 0.25, 0.5, 0.75, and 1 were evaluated. For each value of α, a value of tuning parameter λ is selected using cross validation. This λ value corresponds to the penalty imposed on the magnitude of coefficients and has no useful interpretable meaning. 
* __Random Forest:__ Random Forests utilize multiple tuning parameters, namely: (i) number of trees (ntree), (ii) the number of predictors to consider at each split (mtry), (iii) the sample size in each bagging sample from which to build the tree (sampsize), and (iv) tree size limit (maxnodes).
* __Support Vector Machines:__ Support vector machines (SVMs) possess two general types of tuning parameters: kernel and cost. Within the kernel parameter, we look at kernel type and an additional parameter for fit which depends on type. For polynomial kernels, this second parameter is the polynomial itself. For radial kernels, this second parameter is expressed by γ.Three kernels were explored: linear (polynomial = 1), second degree polynomial, and radial kernels.

# Summary Results

The following table summarizes the results on the selected models utilizing 10-fold cross validation (specifically using the same folds for each model). Of note, logistic regression was performed using a third degree polynomial of the predictor Blue while penalized logistic regression includes third degree polynomial terms for Red, Blue, and Green.

<p align="center">
  <img width = 75% height = 75% src="https://github.com/johnleraas/Haiti-Disaster-Relief/blob/main/Docs/Model_Summary_Table.png">
</p>

While each of the models generated a strong AUROC, LDA and QDA underperformed based upon their comparable true positive rate and precision. Neither model generated a precision of 80% and the true positive rate for both were smaller than their peers.

Logistic Regression and Penalized Logistic Regression generated particularly strong true positive rates, particularly in relation to their respective precisions. Random Forest and KNN also generated strong true positive rates, though with precisions below 90%.

<p align="center">
  <img width = 45% height = 45% src="https://github.com/johnleraas/Haiti-Disaster-Relief/blob/main/Docs/ROC_split1.png">
  <img width = 45% height = 45% src="https://github.com/johnleraas/Haiti-Disaster-Relief/blob/main/Docs/ROC_split2.png">
</p>

The ROC Curves were generated using out of sample data based upon each cross validation fold. The ROC Curves are presented in two separate plots given the overlapping nature of the curves. Logistic Regression was plotted in each for comparison.

Each of the model fits the data remarkably well based upon its ROC curve. LDA, which performed the poorest in the model training and selection process generated an AUC of 0.9889.

# Hold Out Data

For the purpose of this project, a set of data was provided after the initial modeling. A portion of this data was not "clean" and required exploration to discern usable data from unusable data. The hold-out data was obtained in a single .zip file. After unzipping, 11 files were observed of which 3 were JPEG files. These JPEG files possess unlabeled data, and are not useful for evaluating our models (though blue tarps are clearly visible to the human eye). The remaining contain labels of “Blue_Tarps”, “NOT_BLUE_Tarps”, or “NON_Blue_Tarps” in the file name. A subset of data within each file was evaluated to check for likely mislabeling. Specifically, the training data indicated that Blue Tarps typically have higher Blue values than Red or Green. Upon an initial inspection of the data, the labeling appeared correct. Additionally, it seems that columns B1, B2, B3 correspond to Red, Green, Blue (RGB) in the training data. Download the R Markdown file for more detail: [Haiti Disaster Relief Project](https://github.com/johnleraas/Haiti-Disaster-Relief/blob/main/Haiti_Disaster_Relief.html)

<p align="center">
  <img width = 45% height = 45% src="https://github.com/johnleraas/Haiti-Disaster-Relief/blob/main/Docs/HoldOut_Pairwise_FullDataSet.png">
  <img width = 45% height = 45% src="https://github.com/johnleraas/Haiti-Disaster-Relief/blob/main/Docs/HoldOut_Pairwise_BlueTarps.png">
</p>

Certain features or groups of data are observed in the hold-out data which are not observed in the training data. In particular, when looking at the full data set each pairwise comparison exhibits essentially two or three groups of data. This would warrant further inspection or discussions with those collecting the data to understand the differences. For example, these observations may correspond to different times of day or weather patterns.

Additionally, in comparing the Blue/Red comparisons between the full data set and “Blue Tarp” subset, it seems that one “group” of data possessed no Blue Tarp observations. The same is true when considering the Blue/Green pairwise comparisons.

<p align="center">
  <img width = 75% height = 75% src="https://github.com/johnleraas/Haiti-Disaster-Relief/blob/main/Docs/HoldOut_Model_Summary_Table.png">
</p>

Of note the AUROC for most models, with the exception of QDA and Penalized Logistic Regression remained strong on the hold-out data. While there is clearly a significant difference in the frequency of blue tarps observed in the training and hold-out data sets, the AUROCs and TPRs suggest that the threshold could be re-evaluated for certain models.

The highest true positive rates and precision values, and hence the “best models”, are associated with Logistic Regression and KNN. The Random Forest and Support Vector Machine models demonstrated high true positive rates and AUROCs, suggesting that threshold could be re-evaluated to improve precision, though Logistic Regression already possessed the highest true positive rate and precision. Support Vector Machines generated a relatively high false positive rate.

# Conclusions

## Discuss the best performing algorithm(s) in cross-validation and hold-out data

__Cross Validation__

Cross validation on the training data set suggested that LDA and QDA were largely inadequate in comparison to the other models based upon the associated true positive rate and precision (as well as the previously defined loss function). Of the remaining models, Support Vector Machines appeared weakest, as illustrated by the higher loss function value (suggesting a low TPR in relation to precision). Logistic Regression, KNN, Penalized Logistic Regression, and Random Forest were roughly comparable in terms of performance.

__Hold-Out Data__

Logistic Regression (utilizing up to third degree polynomial “Blue” terms) clearly performed the best, with the highest true positive rate and precision. KNN also performed well relative to the other models. LDA performed well, particularly given its relative cross-validation performance. Random Forest also generated acceptable relative performance.

Penalized Logistic Regression performed particularly poorly, though Support Vector Machines and QDA also underperformed.

Of note, the precision of each model was suboptimal. This is at least partiallly due to the difference in frequency of Blue Tarp observations between the data sets. In addition, obvious differences were observed between the Hold-Out Data EDA and the training data set. The Hold-Out data appeared to be collected under different weather/light conditions and/or with different equipment.

## Discuss or present an analysis of why these findings are compatible or reconcilable
As previously discussed, significant differences existed between the training and hold-out data sets. Specifically, the frequency of blue tarps was over four times higher in the training set than the hold-out set. Additionally, the brightness of the data sets appears to be significantly higher in the training set based upon the Red, Green, and Blue predictor value median and mean values.

Each of the models demonstrated a significant decline in precision between cross validation and the hold-out dataset, at least partially due to the lower frequency of blue tarps, resulting in a lower number of true positives relative to false positives. This issue can potentially be solved by re-evaluating the threshold (either in real-time as measurements come in, or through some high-level understanding of the process which generated the training data and hold-out data). Models such as Logistic Regression generated a strong AUROC during cross validation and as measured in the hold-out data set. These models could potentially improve hold-out results by re-evaluating the threshold.

It should be noted that while the Support Vector Machine model generated a strong AUROC, its high false positive rate relative to the other models results in a suboptimal model. Other kernels were explored (not pictured) which resulted in large swings in the TPR and FPR but generally also resulted in a suboptimal model. For example, the radial kernel with γ = 1e-4 and cost = 1e2 showed obvious signs of overfitting and experienced performance declines consistent with penalized logistic regression.

Penalized Logistic Regression demonstrated a significant decline performance, as observed by not only the true positive rate and false positive rate but also by the significant decline in AUROC. Less flexible models generally performed better on the hold-out data than more flexible models. This can be observed directly by (1) Logistic Regression generating the strongest performance, (2) LDA significantly outperforming QDA on the hold-out data set despite underperforming during cross validation, and (3) the significant decline in performance of penalized logistic regression (which considered up to third degree polynomials of each predictor). This suggests some degree of overfitting may have occurred on the training data set by the more flexible models. This overfitting was made evident by the somewhat specific data set used to train relative to the hold-out data set, which seemed to be collected under a variety of conditions.

## Present a recommendation and rationale for which algorithm to ensure for detection of blue tarps
I would recommend Logistic Regression as the algorithm to use for detecting blue tarps (specifically considering higher order polynomials for the Blue predictor). With regards to cross-validation, the performance of Logistic Regression was highly comparable to other models (as measured by AUROC, TPR, FPR, Precision, and Loss). At the same time, Logistic Regression is one of the least flexible models, and thus less subject to overfitting.

Logistic Regression clearly outperformed each of the other models on the hold-out data set, as demonstrated by AUROC, TPR, FPR, Precision, and Loss. Additionally, while the precision declined significantly between cross validation and the hold-out evaluation, this could be addressed by evaluation of real-time data and/or insight regarding the collection of training vs. hold-out data (e.g. the training data was collected on a sunny day or very close to an urban center). The importance of precision is subject to the actual circumstances of the situation. Specifically, if in a real-world implementation someone were able to “check” the aerial photographs before sending supplies the lower precision may be completely acceptable. Conversely, budgets, supply inventories, and an inability to quickly mitigate false positives may determine the threshold derived through cross-validation to be inadequate for Logistic Regression and all models.

## Discuss the relevance of the metrics calculated in the table to this application context
* __AUROC:__ measures the predictive ability of a model across a variety of thresholds. This is an extremely useful metric, particularly in optimizing tuning parameters. Generating the highest AUROC suggests that the model can generate a higher tradeoff between true positive rate and false positive rate (which inherently affects precision).

* __Accuracy:__ has limited relevance to this particular application. Accuracy could arguable useful to support decisions based upon AUROC calculations, however it is largely unnecessary. In this situation the penalty for a false negative and false positive vary widely, as a false negative may result in the suffering and/or death of a human being. Within this context, a higher TPR and FPR associated with a lower accuracy are desirable within real-world constraints.

* __TPR:__ true positive rate essentially indicates the percentage of people that will receive resources. TPR is extremely important given the context of this problem given that it is synonymous with lives saved and the prevention of human suffering.

* __FPR:__ is important within the context of resource constraints. The importance of this metric is largely dependent upon the real-world application. If someone is able to quickly check an aerial photograph to double check the algorithm, there is a somewhat minimal cost to false positives. Conversely, if supplies have to be hiked in, significant budget constraints exist, or there are significant limits to supply inventories the false positive rate becomes more important. Additionally, the frequency of “blue tarp” observations relative to “no blue tarp” observations should be considered alongside the false positive rate. The false positive rate has a direct effect on cost (human time, monetary, supplies, etc.).

* __Precision:__ can be interpreted as the percentage of predicted positives that are accurate. Precision is essentially a balance of positive benefit (lives saved/human suffering prevented) to cost. Given the context of this problem, precision is extremely important given that any humanitarian effort has some set of real-world constraints.

## How effective do you think your work here could actually be in terms of helping to save human life?
The results of this exercise are compelling, though some additional real-world insight may be required for successful implementation. Specifically, each of the models generated a precision of less than 50% on the hold-out data. Though limited information is presented about the real-world implementation, this seems problematic.

This issue may be resolved with a re-evaluation of the threshold of certain models. As previously discussed, a re-evaluation of real-time data or some high-level insight may be beneficial. For example, human evaluation of a subset of aerial photos from each data set may indicate obvious differences and suggest using a higher threshold. Logistic Regression generated a TPR of .994 and a FPR of only .013, suggesting that it is a good candidate for this re-optimization.

Even with the somewhat lower precision observed in the hold-out data, if these models were able to identify focus areas for further human inspection, they may still save a significant amount of time during an urgent situation.

## Were there multiple adequately performing methods, or just one clear best method? What is your level of confidence in the results?
Multiple adequate models were observed in this work. In particular, Logistic Regression, KNN, and Random Forest generated reasonable performance as measured by both cross validation and using the hold-out data set (assuming a low level of precision is acceptable and/or a re-evaluation of threshold is possible).

Of note, significant differences existed between the training data set and the hold-out data set, including the frequency of blue tarp observations and brightness. While the results of the above-mentioned models are positive, given these differences, it does raise significant questions regarding the conditions of training data collection relative to the actual implementation. For example, if the training data was collected on a sunny day at noon near an urban area (bright with a high expected number of blue tarps), similar performance may not be expected on cloudy days during early morning in rural areas.The pairwise comparisons of the hold-out data set suggested that data was collected under multiple lighting conditions and/or with multiple sets of equipment.

The limited information regarding the data collection process and conditions provides some uncertainty. However, I view this as a potential question to explore and consideration refining the model selection and optimization, rather than uncertainty in the process or potential for a model to save significant time and resources.

# References

Plotly

https://plotly.com/r/3d-scatter-plots/

Logistic Regressin with Class Separation

https://medium.com/analytics-vidhya/how-to-use-linear-discriminant-analysis-for-classification-8e46e0ceb3ef

AUC / ROC

https://www.rdocumentation.org/packages/pROC/versions/1.17.0.1/topics/auc

https://www.youtube.com/watch?v=qcvAqAH60Yw

LDA ROC

https://stackoverflow.com/questions/41533811/roc-curve-in-linear-discriminant-analysis-with-r

Confusion Matrix

https://classeval.wordpress.com/introduction/basic-evaluation-measures/

glmnet

https://glmnet.stanford.edu/articles/glmnet.html

https://cran.r-project.org/web/packages/glmnet/glmnet.pdf

Penalized Logistic Regression

http://www.sthda.com/english/articles/36-classification-methods-essentials/149-penalized-logistic-regression-essentials-in-r-ridge-lasso-and-elastic-net/

Random Forests

https://cran.r-project.org/web/packages/randomForest/randomForest.pdf

Support Vector Machine Probabilities

https://www.rdocumentation.org/packages/e1071/versions/1.7-6/topics/predict.svm

Haiti Earthquake

https://en.wikipedia.org/wiki/2010_Haiti_earthquake

ggplot

http://www.sthda.com/english/wiki/ggplot2-scatter-plots-quick-start-guide-r-software-and-data-visualization

https://ggplot2.tidyverse.org/reference/

R Markdown

https://people.ok.ubc.ca/jpither/modules/Symbols_markdown.html

https://rpruim.github.io/s341/S19/from-class/MathinRmd.html

Kable Extra

https://cran.r-project.org/web/packages/kableExtra/vignettes/awesome_table_in_html.html#Column__Row_Specification

https://cran.r-project.org/web/packages/kableExtra/kableExtra.pdf
