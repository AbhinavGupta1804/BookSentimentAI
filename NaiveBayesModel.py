import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import pickle

"""
BNB Accuracy(with Bow):    0.7858333333333334
BNB CM(with Bow):    [[ 401   84]
                     [ 430 1485]]

BNB Accuracy(with Tfidf): 0.7858333333333334
BNB CM(with Tfidf): [[ 401   84]
                    [ 430 1485]]

MNB Accuracy(with Bow):  0.8208333333333333
MNB CM(with Bow):  [[ 571  170]
                   [ 260 1399]]

MNB Accuracy(with Tfidf): 0.6808333333333333
MNB CM(with Tfidf): [[  68    3]
                    [ 763 1566]]

GNB Accuracy(with Bow):    0.5858333333333333
GNB CM(with Bow):   [[552 715]
                    [279 854]]
                    
GNB Accuracy(with Tfidf): 0.5875
GNB CM(with Tfidf): [[538 697]
                    [293 872]]
"""
def PreparingData(train_df):
    train_df=train_df.drop(columns=[i for i in train_df.columns if i not in ["reviewText","rating"]])
    # for binary classification:
    train_df["rating"]=train_df["rating"].apply(lambda x: 0 if x<3 else 1)
    # for multiclass classification:
    # train_df["rating"]=train_df["rating"]
    train_df['reviewText']=train_df['reviewText'].str.lower()

    ### Cleaning Up
    ## Removing special characters (idk)
    train_df['reviewText']=train_df['reviewText'].apply(lambda x:re.sub('[^a-z A-z 0-9-]+', '',x))
    ## Remove the stopswords
    train_df['reviewText']=train_df['reviewText'].apply(lambda x: " ".join([y for y in x.split() if y not in stopwords.words('english')]))
    ## Remove url (idk)
    train_df['reviewText']=train_df['reviewText'].apply(lambda x: re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , str(x)))
    ## Remove html tags (idk)
    # from bs4 import BeautifulSoup
    # train_df['reviewText']=train_df['reviewText'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())
    ## Remove any additional spaces
    train_df['reviewText']=train_df['reviewText'].apply(lambda x: " ".join(x.split()))

    ### Lemmatizing (root wording basically)
    from nltk.stem import WordNetLemmatizer
    lemmatizer=WordNetLemmatizer()
    def lemmatize_words(text):
        return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    train_df['reviewText']=train_df['reviewText'].apply(lemmatize_words)

    ### Train Test Split
    X_train,X_test,y_train,y_test=train_test_split(train_df['reviewText'],train_df['rating'],
                                                test_size=0.20,random_state=69)
    with open("./ARTIFACTS/X_train.pkl","wb") as file:
        pickle.dump(X_train,file)
    with open("./ARTIFACTS/X_test.pkl","wb") as file:
        pickle.dump(X_test,file)
    with open("./ARTIFACTS/y_train.pkl","wb") as file:
        pickle.dump(y_train,file)
    with open("./ARTIFACTS/y_test.pkl","wb") as file:
        pickle.dump(y_test,file)

    return (X_train,X_test,y_train,y_test)

def FeatureExtractionByBOW(X_train,X_test,y_train,y_test):
    #BOW is just enhanced OHE.
    from sklearn.feature_extraction.text import CountVectorizer
    bow=CountVectorizer()
    #this is not training, this is comverting sents in xtrain into bow'ed arrays
    X_train_Bow=bow.fit_transform(X_train).toarray()
    with open("./ARTIFACTS/Bow.pkl","wb") as file:
        pickle.dump(bow,file)
    X_test_Bow=bow.transform(X_test).toarray()
    with open("./ARTIFACTS/X_train_Bow.pkl","wb") as file:
        pickle.dump(X_train_Bow,file)
    with open("./ARTIFACTS/X_test_Bow.pkl","wb") as file:
        pickle.dump(X_test_Bow,file)

    return (X_train_Bow,X_test_Bow)

def FeatureExtractionByTFIDF(X_train,X_test,y_train,y_test):
    #BOW me word count ki jagah word importance (basically enhanced BOW)
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf=TfidfVectorizer()
    X_train_Tfidf=tfidf.fit_transform(X_train).toarray()
    with open("./ARTIFACTS/Tfidf.pkl","wb") as file:
        pickle.dump(tfidf,file)
    X_test_Tfidf=tfidf.transform(X_test).toarray()
    with open("./ARTIFACTS/X_train_Tfidf.pkl","wb") as file:
        pickle.dump(X_train_Tfidf,file)
    with open("./ARTIFACTS/X_test_Tfidf.pkl","wb") as file:
        pickle.dump(X_test_Tfidf,file)

    return (X_train_Tfidf,X_test_Tfidf)

def ClassificationBy_BernoulliNB(X_train,y_train):
    from sklearn.naive_bayes import BernoulliNB
    #this is TRAINING
    BNB_Model=BernoulliNB().fit(
        X_train,y_train
        )
    with open("./NaiveBayes_MODELS/BNB_Model.pkl","wb") as file:
        pickle.dump(BNB_Model,file)

    return BNB_Model

def ClassificationBy_MultinomialNB(X_train,y_train):
    from sklearn.naive_bayes import MultinomialNB
    #This is TRAINING
    MNB_Model=MultinomialNB().fit(X_train,y_train)
    with open("./NaiveBayes_MODELS/MNB_Model.pkl","wb") as file:
        pickle.dump(MNB_Model,file)

    return MNB_Model

def ClassificationBy_GaussianNB(X_train,y_train):
    from sklearn.naive_bayes import GaussianNB
    #This is TRAINING
    GNB_Model=GaussianNB().fit(X_train,y_train)
    with open("./NaiveBayes_MODELS/GNB_Model.pkl","wb") as file:
        pickle.dump(GNB_Model,file)

    return GNB_Model

def PredictionAndAccuracyTest(to_test,test_labels, model):
    pred=model.predict(to_test)
    from sklearn.metrics import confusion_matrix,accuracy_score
    accuracyScore=accuracy_score(pred,test_labels)
    confusionMatrix=confusion_matrix(pred,test_labels)
    return (pred,accuracyScore,confusionMatrix)

if __name__=="__main__":
    train_df=pd.read_csv("./DATASET/kindleReviewDataset.csv",index_col=0)

    X_train,X_test,y_train,y_test=PreparingData(train_df)

    X_train_Bow, X_test_Bow=FeatureExtractionByBOW(X_train,X_test,y_train,y_test)
    X_train_Tfidf, X_test_Tfidf=FeatureExtractionByTFIDF(X_train,X_test,y_train,y_test)

    BNB_Model_With_Bow=ClassificationBy_BernoulliNB(X_train_Bow,y_train)
    with open("./NaiveBayes_MODELS/BNB_Model_With_Bow.pkl","wb") as file:
        pickle.dump(BNB_Model_With_Bow,file)
    pred_BNB_Bow,  accuracyScore_BNB_Bow,  confusionMatrix_BNB_Bow=PredictionAndAccuracyTest(X_test_Bow,y_test, BNB_Model_With_Bow)
    print("BNB Accuracy(with Bow):   ",accuracyScore_BNB_Bow)
    print("BNB CM(with Bow):   ",confusionMatrix_BNB_Bow)

    BNB_Model_With_Tfidf=ClassificationBy_BernoulliNB(X_train_Tfidf,y_train)
    with open("./NaiveBayes_MODELS/BNB_Model_With_Tfidf.pkl","wb") as file:
        pickle.dump(BNB_Model_With_Tfidf,file)
    pred_BNB_Tfidf,accuracyScore_BNB_Tfidf,confusionMatrix_BNB_Tfidf=PredictionAndAccuracyTest(X_test_Tfidf,y_test, BNB_Model_With_Tfidf)
    print("BNB Accuracy(with Tfidf): \n",accuracyScore_BNB_Tfidf)
    print("BNB CM(with Tfidf): \n",confusionMatrix_BNB_Tfidf)



    MNB_Model_With_Bow=ClassificationBy_MultinomialNB(X_train_Bow,y_train)
    with open("./NaiveBayes_MODELS/MNB_Model_With_Bow.pkl","wb") as file:
        pickle.dump(MNB_Model_With_Bow,file)
    pred_MNB_Bow,  accuracyScore_MNB_Bow,  confusionMatrix_MNB_Bow=PredictionAndAccuracyTest(X_test_Bow,y_test, MNB_Model_With_Bow)
    print("MNB Accuracy(with Bow):   ",accuracyScore_MNB_Bow)
    print("MNB CM(with Bow):   ",confusionMatrix_MNB_Bow)

    MNB_Model_With_Tfidf=ClassificationBy_MultinomialNB(X_train_Tfidf,y_train)
    with open("./NaiveBayes_MODELS/MNB_Model_With_Tfidf.pkl","wb") as file:
        pickle.dump(MNB_Model_With_Tfidf,file)
    pred_MNB_Tfidf,accuracyScore_MNB_Tfidf,confusionMatrix_MNB_Tfidf=PredictionAndAccuracyTest(X_test_Tfidf,y_test, MNB_Model_With_Tfidf)
    print("MNB Accuracy(with Tfidf): \n",accuracyScore_MNB_Tfidf)
    print("MNB CM(with Tfidf): \n",confusionMatrix_MNB_Tfidf)



    GNB_Model_With_Bow=ClassificationBy_GaussianNB(X_train_Bow,y_train)
    with open("./NaiveBayes_MODELS/GNB_Model_With_Bow.pkl","wb") as file:
        pickle.dump(GNB_Model_With_Bow,file)
    pred_GNB_Bow,  accuracyScore_GNB_Bow,  confusionMatrix_GNB_Bow=PredictionAndAccuracyTest(X_test_Bow,y_test, GNB_Model_With_Bow)
    print("GNB Accuracy(with Bow):   ",accuracyScore_GNB_Bow)
    print("GNB CM(with Bow):   ",confusionMatrix_GNB_Bow)

    GNB_Model_With_Tfidf=ClassificationBy_GaussianNB(X_train_Tfidf,y_train)
    with open("./NaiveBayes_MODELS/GNB_Model_With_Tfidf.pkl","wb") as file:
        pickle.dump(GNB_Model_With_Tfidf,file)
    pred_GNB_Tfidf,accuracyScore_GNB_Tfidf,confusionMatrix_GNB_Tfidf=PredictionAndAccuracyTest(X_test_Tfidf,y_test, GNB_Model_With_Tfidf)
    print("GNB Accuracy(with Tfidf): \n",accuracyScore_GNB_Tfidf)
    print("GNB CM(with Tfidf): \n",confusionMatrix_GNB_Tfidf)