import pandas as pd
import numpy as np
import seaborn as sns
import operator
import warnings
warnings.filterwarnings(action="ignore")

import matplotlib.pyplot as plt
# %matplotlib inline

import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from bs4 import BeautifulSoup

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

fake = pd.read_csv('Fake.csv')
fake.info()

true = pd.read_csv('True.csv')
true.info()

true['is_fake'] = 0
fake['is_fake'] = 1

news = pd.concat([true, fake], axis=0)

news.head(10)

news.tail(10)

# Rows and columns of fake news dataset
print(f"News dataset has: {news.shape[0]} rows")
print(f"News dataset has: {news.shape[1]} columns")

#checking for null values
news.isna().sum()

news.duplicated().sum()

# remove duplicates
news = news.drop_duplicates(keep='first')
news.duplicated().sum()

import plotly.express as px
fig = px.pie(news, names = "is_fake", title = "Are news fake?",
             width = 1000, height = 500, color_discrete_sequence = px.colors.sequential.Sunset_r)

fig.update_traces(textposition = "inside", textinfo = "percent+label",
                  marker = dict(line = dict(width = 1.2, color = "#000000")))

import plotly.express as px

fig = px.pie(news, names = "subject", title = "News Subject", hole = 0.5,
            width = 1000, height = 500, color_discrete_sequence = px.colors.sequential.Sunset_r)

fig.update_traces(textposition = "inside", textinfo = "percent+label",
                  marker = dict(line = dict(width = 1.2, color = "#000000")))

# news['news'] = news['title'] + ' ' + news['subject']
# news.head(10)

news.rename(columns = {'title':'news'}, inplace = True)
news.drop(['text', 'subject', 'date'], axis=1, inplace=True)
news.head(10)

# news.drop(['title', 'text'], axis=1, inplace=True)
# news.head(10)

news.info()

news = news.sample(frac = 1)
news.head(10)

#library that contains punctuation
import string
string.punctuation

#defining the function to remove punctuation
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

news.to_csv('start.csv', encoding='utf-8', index=False)

#storing the puntuation free text
news['clean_news']= news['news'].apply(lambda x:remove_punctuation(x))
news.head(10)

news['news_lower']= news['clean_news'].apply(lambda x: x.lower())
news.head(10)

#applying function to the column
news['news_tokenied']= news['news_lower'].apply(lambda x: nltk.word_tokenize(x))
news.head(10)

#importing nlp library
import nltk

#Stop words present in the library
stopwords = nltk.corpus.stopwords.words('english')
stopwords[0:10]

#defining the function to remove stopwords from tokenized text
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

#applying the function
news['no_stopwords']= news['news_tokenied'].apply(lambda x:remove_stopwords(x))
news.head(10)

#importing the Stemming function from nltk library
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

#defining a function for stemming
def stemming(text):
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text

news['news_stemmed']=news['no_stopwords'].apply(lambda x: stemming(x))
news.head(10)

from nltk.stem import WordNetLemmatizer
#defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()

#defining the function for lemmatization
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text

news['news_lemmatized'] = news['news_stemmed'].apply(lambda x:lemmatizer(x))
news.head(10)

news['news_lemmatized'] = news['news_lemmatized'].apply(lambda x: ' '.join(x))
news.head(10)

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
tfidf = TfidfVectorizer()

X = tfidf.fit_transform(news['news_lemmatized'])

print(X)

X.shape

y = news['is_fake'].values

y.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

X_train.shape, X_test.shape

"""### Naive Bayes"""

model = MultinomialNB()
model.fit(X_train, y_train)
ans = model.score(X_test, y_test)
y_pred = model.predict(X_test)

print("Multinomial Naive Bayes\nAccuracy score: ", ans)

confusion = pd.DataFrame(confusion_matrix(y_test, y_pred))
confusion = confusion.div(confusion.sum())
confusion.columns = ["Predicted Negative", "Predicted Positive"]
confusion.index = ["Actual Negative", "Actual Positive"]
ax = sns.heatmap(confusion, vmin=0, vmax=1, annot=True, fmt=".0%")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.collections[0].colorbar.set_ticks((0, .25, .5, .75, 1))
ax.collections[0].colorbar.set_ticklabels(("0%", "25%", "50%", "75%", "100%"))

from sklearn.metrics import classification_report, accuracy_score,  precision_score, recall_score, log_loss

print(f'Accuracy: {round(accuracy_score(y_test, y_pred), 3)}')
print(f"Log Loss:  {round(log_loss(y_test, y_pred), 3)}\n")
print(f"F1 Score: {round(f1_score(y_test, y_pred), 3)}")
print(f"Precision: {round(precision_score(y_test,y_pred), 3)}")
print(f"Recall: {round(recall_score(y_test, y_pred), 3)}\n")
print(classification_report(y_test, y_pred))

"""### Support Vector Machine"""

from sklearn.svm import SVC, LinearSVC

svm_model =  SVC(kernel='linear', probability=True) # SVC(kernel = 'rbf', random_state = 0)
svm_model.fit(X_train, y_train)
ans1 = svm_model.score(X_test, y_test)
y_pred1 = svm_model.predict(X_test)

print("Support Vector Machine\nAccuracy score: ", ans1)

confusion = pd.DataFrame(confusion_matrix(y_test, y_pred1))
confusion = confusion.div(confusion.sum())
confusion.columns = ["Predicted Negative", "Predicted Positive"]
confusion.index = ["Actual Negative", "Actual Positive"]
ax = sns.heatmap(confusion, vmin=0, vmax=1, annot=True, fmt=".0%")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.collections[0].colorbar.set_ticks((0, .25, .5, .75, 1))
ax.collections[0].colorbar.set_ticklabels(("0%", "25%", "50%", "75%", "100%"))

from sklearn.metrics import classification_report, accuracy_score,  precision_score, recall_score

print(f'Accuracy: {round(accuracy_score(y_test, y_pred1), 3)}')
print(f"Log Loss:  {round(log_loss(y_test, y_pred1), 3)}\n")
print(f"F1 Score: {round(f1_score(y_test, y_pred1), 3)}")
print(f"Precision: {round(precision_score(y_test,y_pred1), 3)}")
print(f"Recall: {round(recall_score(y_test, y_pred1), 3)}\n")
print(classification_report(y_test, y_pred1))

"""### Logistic Regression"""

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(solver='lbfgs', max_iter=1000)
lr_model.fit(X_train, y_train)
ans2 = lr_model.score(X_test, y_test)
y_pred2 = lr_model.predict(X_test)

print("Logistic Regression\nAccuracy score: ", ans2)

confusion = pd.DataFrame(confusion_matrix(y_test, y_pred2))
confusion = confusion.div(confusion.sum())
confusion.columns = ["Predicted Negative", "Predicted Positive"]
confusion.index = ["Actual Negative", "Actual Positive"]
ax = sns.heatmap(confusion, vmin=0, vmax=1, annot=True, fmt=".0%")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.collections[0].colorbar.set_ticks((0, .25, .5, .75, 1))
ax.collections[0].colorbar.set_ticklabels(("0%", "25%", "50%", "75%", "100%"))

from sklearn.metrics import classification_report, accuracy_score,  precision_score, recall_score

print(f'Accuracy: {round(accuracy_score(y_test, y_pred2), 3)}')
print(f"Log Loss:  {round(log_loss(y_test, y_pred2), 3)}\n")
print(f"F1 Score: {round(f1_score(y_test, y_pred2), 3)}")
print(f"Precision: {round(precision_score(y_test,y_pred2), 3)}")
print(f"Recall: {round(recall_score(y_test, y_pred2), 3)}\n")
print(classification_report(y_test, y_pred2))

"""### Desicion Tree"""

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
ans3 = dt_model.score(X_test, y_test)
y_pred3 = dt_model.predict(X_test)

print("Desicion Tree\nAccuracy score: ", ans3)

confusion = pd.DataFrame(confusion_matrix(y_test, y_pred3))
confusion = confusion.div(confusion.sum())
confusion.columns = ["Predicted Negative", "Predicted Positive"]
confusion.index = ["Actual Negative", "Actual Positive"]
ax = sns.heatmap(confusion, vmin=0, vmax=1, annot=True, fmt=".0%")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.collections[0].colorbar.set_ticks((0, .25, .5, .75, 1))
ax.collections[0].colorbar.set_ticklabels(("0%", "25%", "50%", "75%", "100%"))

from sklearn.metrics import classification_report, accuracy_score, log_loss, precision_score, recall_score

print(f'Accuracy: {round(accuracy_score(y_test, y_pred3), 3)}')
print(f"Log Loss:  {round(log_loss(y_test, y_pred3), 3)}\n")
print(f"F1 Score: {round(f1_score(y_test, y_pred3), 3)}")
print(f"Precision: {round(precision_score(y_test,y_pred3), 3)}")
print(f"Recall: {round(recall_score(y_test, y_pred3), 3)}\n")
print(classification_report(y_test, y_pred3))

"""### Voting Classifier (hard)"""

from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators=[('NB', model),
                                          ('SVC', svm_model),
                                          ('LogReg', lr_model),
                                          ('DTree', dt_model)],
                              voting='hard')

voting_clf.fit(X_train, y_train)
preds = voting_clf.predict(X_test)

confusion = pd.DataFrame(confusion_matrix(y_test, preds))
confusion = confusion.div(confusion.sum())
confusion.columns = ["Predicted Negative", "Predicted Positive"]
confusion.index = ["Actual Negative", "Actual Positive"]
ax = sns.heatmap(confusion, vmin=0, vmax=1, annot=True, fmt=".0%")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.collections[0].colorbar.set_ticks((0, .25, .5, .75, 1))
ax.collections[0].colorbar.set_ticklabels(("0%", "25%", "50%", "75%", "100%"))

from sklearn.metrics import classification_report, accuracy_score, log_loss, f1_score, precision_score, recall_score

print(f'Accuracy: {round(accuracy_score(y_test, preds), 3)}')
print(f"Log Loss:  {round(log_loss(y_test, preds), 3)}\n")
print(f"F1 Score: {round(f1_score(y_test, preds), 3)}")
print(f"Precision: {round(precision_score(y_test, preds), 3)}")
print(f"Recall: {round(recall_score(y_test, preds), 3)}\n")

print(classification_report(y_test, preds))

"""### Voting Classifier (soft)"""

from sklearn.ensemble import VotingClassifier

voting_clf1 = VotingClassifier(estimators=[('NB', model),
                                          ('SVC', svm_model),
                                          ('LogReg', lr_model),
                                          ('DTree', dt_model)],
                              voting='soft')

voting_clf1.fit(X_train, y_train)
preds1 = voting_clf1.predict(X_test)

confusion = pd.DataFrame(confusion_matrix(y_test, preds1))
confusion = confusion.div(confusion.sum())
confusion.columns = ["Predicted Negative", "Predicted Positive"]
confusion.index = ["Actual Negative", "Actual Positive"]
ax = sns.heatmap(confusion, vmin=0, vmax=1, annot=True, fmt=".0%")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.collections[0].colorbar.set_ticks((0, .25, .5, .75, 1))
ax.collections[0].colorbar.set_ticklabels(("0%", "25%", "50%", "75%", "100%"))

from sklearn.metrics import classification_report, accuracy_score, log_loss, f1_score, precision_score, recall_score

print(f'Accuracy: {round(accuracy_score(y_test, preds1), 3)}')
print(f"Log Loss:  {round(log_loss(y_test, preds1), 3)}\n")
print(f"F1 Score: {round(f1_score(y_test, preds1), 3)}")
print(f"Precision: {round(precision_score(y_test, preds1), 3)}")
print(f"Recall: {round(recall_score(y_test, preds1), 3)}\n")

print(classification_report(y_test, preds1))

"""### XGBoost"""

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic', silent=True, nthread=1)

# A parameter grid for XGBoost
params_1 = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'max_depth': [5, 6, 7]
        }

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# folds = 3  # number of folds to be used
# 
# skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1)  # define a stratified K-Fold to preserve percentage of each target class
# 
# random_search_1 = RandomizedSearchCV(xgb, param_distributions=params_1, n_iter=4, scoring=['roc_auc','accuracy','recall','precision'],
#                                    n_jobs=-1, cv=skf.split(X_train,y_train), verbose=2, random_state=1001, refit='roc_auc')
# 
# random_search_1.fit(X_train, y_train)
# random_search_1.best_params_

def results_summary(classifier):
    roc_auc_results = classifier.cv_results_['mean_test_roc_auc']
    loc = np.where(roc_auc_results == np.amax(roc_auc_results))[0][0]

    rs_roc_auc = classifier.cv_results_['mean_test_roc_auc'][loc]
    rs_prec = classifier.cv_results_['mean_test_precision'][loc]
    rs_recall = classifier.cv_results_['mean_test_recall'][loc]
    rs_accur = classifier.cv_results_['mean_test_accuracy'][loc]

    print("ROC_AUC = {:.3f}".format(rs_roc_auc))
    print("Precision = {:.3f}".format(rs_prec))
    print("Recall = {:.3f}".format(rs_recall))
    print("Accuracy = {:.3f}".format(rs_accur))

    return [rs_roc_auc,rs_prec,rs_recall,rs_accur] # return array for the final summary

xgb_results = results_summary(random_search_1)

xgb_best = random_search_1.best_estimator_

from xgboost import plot_importance
plot_importance(xgb_best, max_num_features=15) # top 15 most important features
plt.show()

""" ### Ada Boost Classifier"""

from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier()

params_2 = {
        'n_estimators': np.arange(100,1200,100),
        'learning_rate': np.arange(0.1,1.1,0.2),
        }

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# random_search_2 = RandomizedSearchCV(ada, param_distributions=params_2, n_iter=4, scoring=['roc_auc','accuracy','recall','precision'],
#                                    n_jobs=-1, cv=skf.split(X_train,y_train), verbose=3, random_state=1001, refit='roc_auc')
# random_search_2.fit(X_train, y_train)
# 
# ada_best = random_search_2.best_estimator_
# ada_results = results_summary(random_search_2)

ada_best.feature_importances_[:10]

"""### Random Forest Classifier"""

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_jobs=-1)

params_3 = {
        'n_estimators': np.arange(100,1000,100),
        'max_depth': np.arange(30,110,10),
        'bootstrap': [True, False]
        }

random_search_3 = RandomizedSearchCV(rfc, param_distributions=params_3, n_iter=8, scoring=['roc_auc','accuracy','recall','precision'],cv=skf.split(X_train,y_train), verbose=3, random_state=1001, refit='roc_auc',n_jobs=-1)
random_search_3.fit(X_train, y_train)

rfc_best = random_search_3.best_estimator_
rfc_results = results_summary(random_search_3)

"""### Voting Classifier (hard)"""

from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators=[("XGB", xgb_best),
                                          ("ADA", ada_best),
                                          ('RFC', rfc_best)],
                              voting='hard')

voting_clf.fit(X_train, y_train)
preds = voting_clf.predict(X_test)

confusion = pd.DataFrame(confusion_matrix(y_test, preds))
confusion = confusion.div(confusion.sum())
confusion.columns = ["Predicted Negative", "Predicted Positive"]
confusion.index = ["Actual Negative", "Actual Positive"]
ax = sns.heatmap(confusion, vmin=0, vmax=1, annot=True, fmt=".0%")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.collections[0].colorbar.set_ticks((0, .25, .5, .75, 1))
ax.collections[0].colorbar.set_ticklabels(("0%", "25%", "50%", "75%", "100%"))

from sklearn.metrics import classification_report, accuracy_score, log_loss, f1_score, precision_score, recall_score

print(f'Accuracy: {round(accuracy_score(y_test, preds), 3)}')
print(f"Log Loss:  {round(log_loss(y_test, preds), 3)}\n")
print(f"F1 Score: {round(f1_score(y_test, preds), 3)}")
print(f"Precision: {round(precision_score(y_test, preds), 3)}")
print(f"Recall: {round(recall_score(y_test, preds), 3)}\n")

print(classification_report(y_test, preds))

"""### Voting Classifier (soft)"""

from sklearn.ensemble import VotingClassifier

voting_clf1 = VotingClassifier(estimators=[("XGB", xgb_best),
                                          ("ADA", ada_best),
                                          ('RFC', rfc_best)],
                              voting='soft')

voting_clf1.fit(X_train, y_train)
preds1 = voting_clf1.predict(X_test)

confusion = pd.DataFrame(confusion_matrix(y_test, preds1))
confusion = confusion.div(confusion.sum())
confusion.columns = ["Predicted Negative", "Predicted Positive"]
confusion.index = ["Actual Negative", "Actual Positive"]
ax = sns.heatmap(confusion, vmin=0, vmax=1, annot=True, fmt=".0%")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.collections[0].colorbar.set_ticks((0, .25, .5, .75, 1))
ax.collections[0].colorbar.set_ticklabels(("0%", "25%", "50%", "75%", "100%"))

from sklearn.metrics import classification_report, accuracy_score, log_loss, f1_score, precision_score, recall_score

print(f'Accuracy: {round(accuracy_score(y_test, preds1), 3)}')
print(f"Log Loss:  {round(log_loss(y_test, preds1), 3)}\n")
print(f"F1 Score: {round(f1_score(y_test, preds1), 3)}")
print(f"Precision: {round(precision_score(y_test, preds1), 3)}")
print(f"Recall: {round(recall_score(y_test, preds1), 3)}\n")

print(classification_report(y_test, preds1))





