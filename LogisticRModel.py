# Logistic Regression Classification

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
import pickle
from sklearn.linear_model import LogisticRegression

"""
#Accuracy with BoW: 0.8316666666666667
#CM with BoW: [[ 597  234]
              [ 170 1399]]

#Accuracy with TF-IDF: 0.8341666666666666
#CM with TF-IDF: [[ 549  282]
                 [ 116 1453]]
"""

def loadTestAndTrainPickle():
    with open("./ARTIFACTS/X_train_Bow.pkl", "rb") as file:
        X_train_Bow=pickle.load(file)
    with open("./ARTIFACTS/X_train_Tfidf.pkl", "rb") as file:
        X_train_Tfidf=pickle.load(file)

    with open("./ARTIFACTS/X_test_Bow.pkl","rb") as file:
        X_test_Bow=pickle.load(file)
    with open("./ARTIFACTS/X_test_Tfidf.pkl","rb") as file:
        X_test_Tfidf=pickle.load(file)

    with open("./ARTIFACTS/y_train.pkl","rb") as file:
        y_train=pickle.load(file)
    with open("./ARTIFACTS/y_test.pkl","rb") as file:
        y_test=pickle.load(file)
    return X_train_Bow,X_train_Tfidf,X_test_Bow,X_test_Tfidf,y_train,y_test

def BuildModel(X_train, y_train):
    logR= LogisticRegression(
        max_iter=1000,#maximum no of times we let it converge
        solver="lbfgs",#limited memory bfgs(works best for small to medium sized data)
        verbose=1
    )
    logR.fit(
        X_train,
        y_train
    )
    return logR

if __name__=="__main__":
    X_train_Bow,X_train_Tfidf,X_test_Bow,X_test_Tfidf,y_train,y_test=loadTestAndTrainPickle()
    # model_Bow=BuildModel(X_train_Bow,y_train)
    # with open("./ClassicalML_MODELS/LogR_Bow.pkl", "wb") as file:
    #     pickle.dump(model_Bow, file)

    # model_Tfidf=BuildModel(X_train_Tfidf, y_train)
    # with open("./ClassicalML_MODELS/LogR_Tfidf.pkl", "wb") as file:
    #     pickle.dump(model_Tfidf, file)

    #Load models
    with open("./ClassicalML_MODELS/LogR_Bow.pkl", "rb") as file:
        model_Bow=pickle.load(file)
    with open("./ClassicalML_MODELS/LogR_Tfidf.pkl", "rb") as file:
        model_Tfidf =pickle.load(file)

    # Accuracy and confudion matrix
    pred_Bow=model_Bow.predict(X_test_Bow)
    accuracy_Bow=accuracy_score(y_test, pred_Bow)
    cm_Bow=confusion_matrix(y_test,pred_Bow)

    pred_Tfidf=model_Tfidf.predict(X_test_Tfidf)
    accuracy_Tfidf=accuracy_score(y_test, pred_Tfidf)
    cm_Tfidf=confusion_matrix(y_test,pred_Tfidf)

    print(f"\nAccuracy with BoW: {accuracy_Bow}")
    print(f"\nCM with BoW: {cm_Bow}")
    print(f"Accuracy with TF-IDF: {accuracy_Tfidf}")
    print(f"\nCM with TF-IDF: {cm_Tfidf}")

    #Custom Prediction
    # with open("./ARTIFACTS/Bow.pkl","rb") as file:
    #     Bow=pickle.load(file)
    # with open("./ARTIFACTS/Tfidf.pkl","rb") as file:
    #     Tfidf=pickle.load(file)

    # custom_test=["are we serious, its bad but good","not good bad very bad very very bad bad"]
    # custom_test_Bow=Bow.transform(custom_test).toarray()
    # custom_test_Tfidf=Tfidf.transform(custom_test).toarray()

    # pred_custom_Bow=model_Bow.predict(custom_test_Bow)
    # print(pred_custom_Bow)
    # pred_custom_Tfidf=model_Tfidf.predict(custom_test_Tfidf)
    # print(pred_custom_Tfidf)
