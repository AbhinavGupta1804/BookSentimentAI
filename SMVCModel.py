# Support Vector Machine Classification

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
import pickle

"""
SVMC Accuracy with BOW: 0.7895833333333333
SVMC CM with BOW: [[ 535  296]
                  [ 209 1360]]
                  
SVMC Accuracy with Tfidf: 0.8220833333333334
SVMC CM with Tfidf: [[ 554  277]
                    [ 150 1419]]
"""
def loadTestAndTrainDataPickle():
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


# print(X_train.shape)       #(9600,)
# print(X_test.shape)        #(2400,)
# print(X_train_Bow.shape)   #(9600, 35811)
#9600 vectors, of size 35811 each
# print(X_train_Tfidf.shape) #(9600, 35811)
# print(X_test_Bow.shape)    #(2400, 35811)
# print(X_test_Tfidf.shape)  #(2400, 35811)
# print(y_train.shape)       #(9600,)
# print(y_test.shape)        #(2400,)

def PCA(X_train,X_test):
    #since we have too many features(35811), SVM stuck
    #   so use PCA to reduce features
    from sklearn.decomposition import PCA
    #300 still toooo much tho, 300 words cant be important
    #   forgive me father for i have made an unneccessarily expensive model
    pca=PCA(n_components=300) #reduces to 300(use eigen vectors find out how)
    X_train_PCA=pca.fit_transform(X_train) 
    X_test_PCA=pca.transform(X_test)
    with open("./ARTIFACTS/PCAforSVC.pkl","wb") as file:
        pickle.dump(pca,file)

    return (X_train_PCA,X_test_PCA)

def BuildModel(X_train, y_train):
    SVM_Model=SVC(kernel="linear", C=1.0, verbose=True)
    #c is like intensity of classification, more and more c will lead to overfitting, 1.0 is fine
    SVM_Model.fit(
        X_train,
        y_train
    )
    return SVM_Model

def PredictionAndMetrix(to_pred,y_test,model):
    pred=model.predict(to_pred)
    cm=confusion_matrix(y_test,pred)
    acc_score=accuracy_score(y_test,pred)
    return pred,cm,acc_score


if __name__=="__main__":
    X_train_Bow,X_train_Tfidf,X_test_Bow,X_test_Tfidf,y_train,y_test=loadTestAndTrainDataPickle()
    X_train_Bow_PCAed,X_test_Bow_PCAed=PCA(X_train_Bow,X_test_Bow)
    X_train_Tfidf_PCAed,X_test_Tfidf_PCAed=PCA(X_train_Tfidf,X_test_Tfidf)

    ##build models
    ## with open("./SVM_MODELS/SVC_Bow.pkl","wb") as file:
    ##     pickle.dump(BuildModel(X_train_Bow_PCAed,y_train),file)
    ## with open("./SVM_MODELS/SVC_Tfidf.pkl","wb") as file:
    ##     pickle.dump(BuildModel(X_train_Tfidf_PCAed,y_train),file)

    with open("./SVM_MODELS/SVC_Bow.pkl","rb") as file:
        SVC_Bow_Model=pickle.load(file)
    with open("./SVM_MODELS/SVC_Tfidf.pkl","rb") as file:
        SVC_Tfidf_Model=pickle.load(file)

    pred_Bow,cm_Bow,Accuracy_Bow=PredictionAndMetrix(X_test_Bow_PCAed,y_test,SVC_Bow_Model)
    pred_Tfidf,cm_Tfidf,Accuracy_Tfidf=PredictionAndMetrix(X_test_Tfidf_PCAed,y_test,SVC_Tfidf_Model)

    # Accuracy_Bow=accuracy_score(y_test,pred_Bow)
    # Accuracy_Tfidf=accuracy_score(y_test,pred_Tfidf)
    print("SVMC Accuracy with BOW:", Accuracy_Bow)
    print("SVMC CM with BOW:", cm_Bow)
    print("SVMC Accuracy with Tfidf:", Accuracy_Tfidf)
    print("SVMC CM with Tfidf:", cm_Tfidf)
    ## SVMC Accuracy with BOW: 0.8
    ## SVMC Accuracy with Tfidf: 0.8

    #Custom Prediction
    # with open("./ARTIFACTS/Bow.pkl", "rb") as file:
    #     Bow=pickle.load(file)
    # with open("./ARTIFACTS/Tfidf.pkl", "rb") as file:
    #     Tfidf=pickle.load(file)

    # custom_test=["book was awesome, I loved the book. It was just so awesome", "book was bad, like very bad"]

    # custom_test_Bow=Bow.transform(custom_test).toarray()
    # _,custom_test_Bow_PCAed=PCA(X_train_Bow,custom_test_Bow)

    # custom_test_Tfidf=Tfidf.transform(custom_test).toarray()
    # _,custom_test_Tfidf_PCAed=PCA(X_train_Tfidf,custom_test_Tfidf)

    # pred_custom_Bow = SVC_Bow_Model.predict(custom_test_Bow_PCAed)
    # print("Custom Predictions By SVC_Bow_Model:", pred_custom_Bow)
    # pred_custom_Tfidf = SVC_Tfidf_Model.predict(custom_test_Tfidf_PCAed)
    # print("Custom Predictions By SVC_Tfidf_Model:", pred_custom_Tfidf)
    # # Custom Predictions By SVC_Bow_Model: [1 0]
    # # Custom Predictions By SVC_Tfidf_Model: [1 0]