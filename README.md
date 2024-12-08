### SyriaTel Churn Modeling: Predicting Customer Retention








<img src="sl-blog-2021-04-churnandretention-2.jpg" width="1200" height="100" />









#### Business Overview
SyriaTel is a leading mobile communications service provider headquartered in Damascus, Syria. As the top-performing mobile operator in the country, SyriaTel operates a 2G and 3G network and partners with over 200 international providers across 121 countries (Tracxn, 2024). The company offers a wide range of services, including data, voice, messaging, news, and roaming.

Like many telecom companies, SyriaTel faces the challenge of customer churn — the loss of subscribers to competitors. Identifying potentially dissatisfied customers before they leave is critical for reducing churn and improving profitability. Understanding the rate at which customers leave is essential for effective churn analysis, which helps the company maintain a competitive edge by retaining valuable customers. This proactive approach can significantly increase customer retention and profitability.

SyriaTel can leverage machine learning algorithms to analyze customer behavior and churn-influencing factors, enabling data-driven decisions to improve customer retention strategies.

#### Problem Statement
SyriaTel currently lacks a reliable method for identifying customers at risk of churn before they leave. This absence of predictive tools hampers the company's ability to target customers with retention strategies effectively, leading to increased turnover rates and higher customer acquisition costs.

#### Approach Methodology
To tackle this problem, we perform a detailed exploration of the available data to understand the relationships between various factors influencing customer churn. The process begins with:

Correlation Analysis: Identifying strong relationships between features to determine the most influential factors for churn.
Univariate, Bivariate, and Multivariate Analysis: Further analysis reveals patterns and the interaction of features with churn.
Feature Selection: Based on these analyses, we select relevant features for building predictive models.
Once the data is pre-processed and cleaned, we train models to predict customer churn and evaluate their performance using key metrics.

#### Metrics of Success
The success of the churn prediction model is evaluated using the following metrics:

Model Accuracy: > 80%
Precision: > 70%
Recall: > 60%
F1-Score: > 60%
ROC-AUC: > 70%
Data Understanding
We used a customer churn dataset sourced from Kaggle for this analysis. The dataset, containing 21 columns and 3333 rows, includes a mix of categorical, numerical, and boolean data. Initial exploration of the dataset with Pandas (via .info() and .describe()) showed a balance of different data types, providing an opportunity to explore customer behaviors in detail.

### Data Preparation

#### Data Cleaning
Data cleaning is a crucial step to ensure the dataset is ready for analysis. We used the following methods to detect any issues:

Null Values: The .isna() method helped identify missing values, but the dataset was free of null values.
Duplicates: We used .duplicated() to identify and remove any duplicate rows.
Some columns, such as area codes, were numerical but treated as discrete categorical variables since the difference between values was not meaningful.

#### Exploratory Data Analysis (EDA)
EDA forms the foundation of understanding the dataset. Through visualizations (scatter plots, histograms, and box plots), we uncovered key patterns, distributions, and trends in the data. Additionally, we performed:

Univariate Analysis: Analyzing individual features.
Bivariate and Multivariate Analysis: Exploring relationships between multiple features and churn.
Feature Correlations
Understanding the correlation between features helps identify which variables influence churn. Positive correlations occur when two features change in the same direction, while negative correlations indicate an inverse relationship.

#### Modeling
We developed predictive models to identify customer churn, starting with Logistic Regression. To ensure the model performs well, we took several steps:

Data Preprocessing: The dataset was preprocessed to ensure it's suitable for modeling.
Handling Class Imbalance: We used the Synthetic Minority Over-sampling Technique (SMOTE) to balance the classes and address any data imbalance.
Model Tuning: Hyperparameters were tuned for regularization to avoid overfitting.
We compared the performance of two models:

#### Logistic Regression
Decision Tree Classifier
The best-performing model was selected based on accuracy, precision, recall, F1-score, and AUC.

#### Model Evaluation
We evaluated the models against the success metrics outlined above. The final selected model was chosen based on its ability to effectively predict churn, helping SyriaTel take action on retaining customers.

#### Conclusion
This churn modeling project provides SyriaTel with a reliable method for identifying customers at risk of leaving. By implementing predictive models, the company can proactively intervene, targeting at-risk customers with tailored retention strategies that could ultimately boost long-term profitability.




