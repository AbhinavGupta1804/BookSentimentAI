
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score,confusion_matrix
import pickle
from tensorflow import keras
import numpy as np
"""
# Accuracy By Bow ANN:  0.8166666666666667
# CM By Bow ANN:  [[ 553  278]
                  [ 162 1407]]
# Accuracy By Tfidf ANN:  0.8020833333333334
# CM By Tfidf ANN:  [[ 575  256]
                    [ 219 1350]]
"""

def loadTrainAndTestDataPickle():
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

    with open("./ARTIFACTS/X_train.pkl", "rb") as file:
        X_train=pickle.load(file)
    with open("./ARTIFACTS/X_test.pkl", "rb") as file:
        X_test=pickle.load(file)
    return X_train_Bow,X_train_Tfidf,X_test_Bow,X_test_Tfidf,y_train,y_test,X_train,X_test

# print(X_train.shape)       #(9600,)
# print(X_test.shape)        #(2400,)
# print(X_train_Bow.shape)   #(9600, 35811)
#9600 vectors, of size 35811 each
# print(X_train_Tfidf.shape) #(9600, 35811)
# print(X_test_Bow.shape)    #(2400, 35811)
# print(X_test_Tfidf.shape)  #(2400, 35811)
# print(y_train.shape)       #(9600,)
# print(y_test.shape)        #(2400,)

def buildANNModel(X_train,X_test,y_train,y_test):
    model=Sequential()
    model.add(keras.Input(shape=X_train.shape[1:]))
    model.add(Dense(128,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64,activation="relu"))
    model.add(Dense(1,activation="sigmoid"))
    #if rating (1 to 5):
        #if OHE,    Dense(5,activation="softmax")
                    #loss="categorical_crossentropy"

        #if Not OHE, Dense(1,activation="softmax")
                    #loss="sparse_categorical_crossentropy"

    #if rating (0 to 1):
        #if OHE,     Dense(2,activation="softmax")
                    #loss="categorical_crossentropy"

        #if Not OHE, Dense(1,activation="sigmoid")
                    #loss="binary_crossentropy"
    #since hume ek number output hi chahiye ratings(1_to_5 ya 0_to_1 jaise liya ho pkl banate wakt)
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    trainHistory=model.fit(
        X_train,
        y_train,
        epochs=20,
        validation_data=(X_test,y_test)
    )
    return (trainHistory,model)

def Prediction(X_test,model):
    return model.predict(X_test)

def readycustomPrediction(to_test,vectorizer):
    to_test=[to_test]
    #Custom Prediction
    if (vectorizer.lower()=="bow"):
        model=load_model("./NeuralNetwork_MODELS/ANN_Bow.keras")
        with open("./ARTIFACTS/Bow.pkl","rb") as file:
            Bow=pickle.load(file)
            to_test=Bow.transform(to_test).toarray()
    else :
        model=load_model("./NeuralNetwork_MODELS/ANN_Tfidf.keras")
        with open("./ARTIFACTS/Tfidf.pkl","rb") as file:
            Tfidf=pickle.load(file)
            to_test=Tfidf.transform(to_test).toarray()


    pred_custom=Prediction(to_test,model)
    print(pred_custom)
    floor=lambda x: 0 if x<0.5 else 1
    print([floor(i) for i in pred_custom.flatten()])
    #working like a charm
    return pred_custom.flatten()


if __name__=="__main__":
    # X_train_Bow,X_train_Tfidf,X_test_Bow,X_test_Tfidf,y_train,y_test,X_train,X_test=loadTrainAndTestDataPickle()
    # # trainHistory,model=buildANNModel(X_train_Bow,X_test_Bow,y_train,y_test)
    # # model.save("./NeuralNetworks_MODELS/ANN_Bow.keras")
    # # trainHistory,model=buildANNModel(X_train_Tfidf,X_test_Tfidf,y_train,y_test)
    # # model.save("./NeuralNetwork_MODELS/ANN_Tfidf.keras")

    # #load_models
    # model_Bow=load_model("./NeuralNetwork_MODELS/ANN_Bow.keras")
    # model_Tfidf=load_model("./NeuralNetwork_MODELS/ANN_Tfidf.keras")
    # pred_Bow=Prediction(X_test_Bow,model_Bow)
    # pred_Tfidf=Prediction(X_test_Tfidf,model_Tfidf)
    # #pred will automatically be a numpy array

    # #modifying array in place
    # pred_Bow[pred_Bow<0.5]=0
    # pred_Bow[pred_Bow>=0.5]=1
    # pred_Tfidf[pred_Tfidf<0.5]=0
    # pred_Tfidf[pred_Tfidf>=0.5]=1

    # #since pred=[[],[],..] and y_test=[,,..],
    # #   we can flatten pred to [,,..]

    # #confusion matrix:
    # cm_Bow=confusion_matrix(y_test,pred_Bow)
    # cm_Tfidf=confusion_matrix(y_test,pred_Tfidf)

    # #accuracy
    # accuracy_Bow=np.mean(pred_Bow.flatten()==y_test)
    # accuracy_Tfidf=np.mean(pred_Tfidf.flatten()==y_test)

    # print("Accuracy By Bow ANN: ",accuracy_Bow)
    # print("CM By Bow ANN: ",cm_Bow)
    # print("Accuracy By Tfidf ANN: ",accuracy_Tfidf)
    # print("CM By Tfidf ANN: ",cm_Tfidf)

    # Custom Prediction
    readycustomPrediction(" bad","tfidf",)


        
