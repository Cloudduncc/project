# project
Cloud Computing UNCC Project

## TOPIC

**PRODUCT CLASSIFICATION BASED ON SENTIMENTAL ANALYSIS ON LARGE SCALE AMAZON PRODUCT REVIEWS**

**TEAM MEMBERS**

*AKASH ASHOK (800991236)  
PRABHAKAR TEJA SEEDA (800)  
KARTHICK SELVARAJ (800)*

**ABSTRACT**

*This project aims at classifying online products based on analysis of customers’ sentiments on a large-scale amazon product review dataset. This is achieved by using traditional Information Retrieval techniques and Machine Learning approach. Logistic Regression and Random Forest techniques are used for classification and regression. Alternating Least Square method is made use of for collaborative filtering for recommender systems. The dataset being made use of is the Amazon Reviews dataset.*

**DATA SETS**

The dataset we are using is the Amazon Reviews dataset. It spans over a period of 18 years, containing about ~35 million reviews up to March 2013. Reviews include product and user information, ratings and a plaintext review. The dataset was downloaded from Stanford.edu and was compiled by J. McAuley and J. Leskovec.

![](Images/Dataset.png)

The dataset is close to 11 GB and is formatted in JSON. The files are also split by product categories. It has huge volume to leverage for running on cluster platforms. 

**TEAM RESPONSIBILITIES**

*Needs to be filled*

**DATA PREPROCESSING**

*Needs to be filled*

**ALGORITHMS IMPLEMENTED**

***LOGISTIC REGRESSION***  
Logistic Regression is a predictive analysis. It is used in describing data and relationship between one dependent binary variable with one or more independent variables. Using Logistic Regression, we can find the best fitting model to describe the relationship between these variables. It is more robust, in the sense that it does not require the independent variables to be normally distributed and because we are dealing with a lot of data, we can make use of Logistic Regression. This approach is evaluated using the ROC curve analysis. 
Logistic Regression with TF-IDF: Term Frequency-Inverse Document Frequency is a widely known technique for text processing. Here, each term in the document is assigned a weight. Words which appear more, that is with more frequency are assigned higher weights. Also, if a word appears frequently in all the documents of the document corpus, it is assigned lower weights.
Term Frequency TF is the number of times a term occurs within a document. A term in different documents have different TF values. Document Frequency is the number of documents having this term. There may be commonly occurring words like “a” “is” in the documents. Although such words have a high term frequency, they do not convey important information. With Inverse Document Frequency, we can down-weight such background noise terms. TF-IDF is the product of the term’s TF and IDF scores. Each term has a different TF-IDF score in every document in the corpus. The search engine accepts a set of user-specified keywords and ranks all documents against the user’s set of keywords. The rank of the document is a sum of TF_IDF weights for all the user’s keywords within this documents.
The formula used to calculate IDF that is included in Spark MLib is: 
`IDF(t,D) = log[ (|D| + 1) / (DF(t,D) + 1) ]`  
where,



