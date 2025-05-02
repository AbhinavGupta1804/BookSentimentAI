import streamlit as st
import pickle
from keras.models import load_model
import os

st.set_page_config(
    page_title="BookSentimentAI",
    page_icon="😊",
    layout="wide"
)

# Load all models and vectorizers
@st.cache_resource
def load_models_and_vectorizers():
    models = {}
    
    # Load Vectorizers
    with open("./ARTIFACTS/Bow.pkl", "rb") as file:
        models["BOW_Vectorizer"] = pickle.load(file)
    with open("./ARTIFACTS/Tfidf.pkl", "rb") as file:
        models["TFIDF_Vectorizer"] = pickle.load(file)

    # Load Logistic Regression Models
    with open("./ClassicalML_MODELS/LogR_Bow.pkl", "rb") as file:
        models["LogR_BOW_Model"] = pickle.load(file)
    with open("./ClassicalML_MODELS/LogR_Tfidf.pkl", "rb") as file:
        models["LogR_TFIDF_Model"] = pickle.load(file)

    # Load Naive Bayes Models
    with open("./NaiveBayes_MODELS/BNB_Model_With_Bow.pkl", "rb") as file:
        models["BNB_Model_With_Bow"] = pickle.load(file)
    with open("./NaiveBayes_MODELS/BNB_Model_With_Tfidf.pkl", "rb") as file:
        models["BNB_Model_With_Tfidf"] = pickle.load(file)

    with open("./NaiveBayes_MODELS/MNB_Model_With_Bow.pkl", "rb") as file:
        models["MNB_Model_With_Bow"] = pickle.load(file)
    with open("./NaiveBayes_MODELS/MNB_Model_With_Tfidf.pkl", "rb") as file:
        models["MNB_Model_With_Tfidf"] = pickle.load(file)

    with open("./NaiveBayes_MODELS/GNB_Model_With_Bow.pkl", "rb") as file:
        models["GNB_Model_With_Bow"] = pickle.load(file)
    with open("./NaiveBayes_MODELS/GNB_Model_With_Tfidf.pkl", "rb") as file:
        models["GNB_Model_With_Tfidf"] = pickle.load(file)

    # Load SVM Models
    with open("./ARTIFACTS/PCAforSVC.pkl", "rb") as file:
        models["PCAforSVC"] = pickle.load(file)
    with open("./SVM_MODELS/SVC_Bow.pkl", "rb") as file:
        models["SVC_BOW_Model"] = pickle.load(file)
    with open("./SVM_MODELS/SVC_Tfidf.pkl", "rb") as file:
        models["SVC_TFIDF_Model"] = pickle.load(file)

    # Load Neural Network Models
    models["ANN_BOW_Model"] = load_model("./NeuralNetwork_MODELS/ANN_Bow.keras")
    models["ANN_TFIDF_Model"] = load_model("./NeuralNetwork_MODELS/ANN_Tfidf.keras")
    
    return models

def predict_sentiment(text, vectorizer_type, model_type, models):
    if vectorizer_type == "bow":
        if model_type == "ann":
            to_test = models["BOW_Vectorizer"].transform([text]).toarray()
            pred = models["ANN_BOW_Model"].predict(to_test)
            pred = pred[0][0]
        elif model_type == "svm":
            to_test = models["BOW_Vectorizer"].transform([text])
            to_test = models["PCAforSVC"].transform(to_test)
            pred = models["SVC_BOW_Model"].predict(to_test)
            pred = pred[0]
        elif model_type == "lr":
            to_test = models["BOW_Vectorizer"].transform([text])
            pred = models["LogR_BOW_Model"].predict(to_test)
            pred = pred[0]
        elif model_type == "nb_bern":
            to_test = models["BOW_Vectorizer"].transform([text])
            pred = models["BNB_Model_With_Bow"].predict(to_test)
            pred = pred[0]
        elif model_type == "nb_multi":
            to_test = models["BOW_Vectorizer"].transform([text])
            pred = models["MNB_Model_With_Bow"].predict(to_test)
            pred = pred[0]
        elif model_type == "nb_guass":
            to_test = models["BOW_Vectorizer"].transform([text])
            pred = models["GNB_Model_With_Bow"].predict(to_test.toarray())
            pred = pred[0]
    elif vectorizer_type == "tfidf":
        if model_type == "ann":
            to_test = models["TFIDF_Vectorizer"].transform([text]).toarray()
            pred = models["ANN_TFIDF_Model"].predict(to_test)
            pred = pred[0][0]
        elif model_type == "svm":
            to_test = models["TFIDF_Vectorizer"].transform([text])
            to_test = models["PCAforSVC"].transform(to_test)
            pred = models["SVC_TFIDF_Model"].predict(to_test)
            pred = pred[0]
        elif model_type == "lr":
            to_test = models["TFIDF_Vectorizer"].transform([text])
            pred = models["LogR_TFIDF_Model"].predict(to_test)
            pred = pred[0]
        elif model_type == "nb_bern":
            to_test = models["TFIDF_Vectorizer"].transform([text])
            pred = models["BNB_Model_With_Tfidf"].predict(to_test)
            pred = pred[0]
        elif model_type == "nb_multi":
            to_test = models["TFIDF_Vectorizer"].transform([text])
            pred = models["MNB_Model_With_Tfidf"].predict(to_test)
            pred = pred[0]
        elif model_type == "nb_guass":
            to_test = models["TFIDF_Vectorizer"].transform([text])
            pred = models["GNB_Model_With_Tfidf"].predict(to_test.toarray())
            pred = pred[0]
    
    return {
        "confidence": str(pred),
        "sentiment": "Positive" if pred > 0.5 else "Negative"
    }

def main():
    st.title("BookSentimentAI")
    st.write("Enter text to analyze its sentiment using different models and vectorizers")
    
    # Load all models and vectorizers
    models = load_models_and_vectorizers()
    
    # Create input area
    text_input = st.text_area("Enter text for sentiment analysis:", height=150)
    
    # Create sidebar for model selection
    st.sidebar.header("Model Settings")
    
    vectorizer = st.sidebar.selectbox(
        "Select Vectorizer",
        options=["bow", "tfidf"],
        format_func=lambda x: "Bag of Words" if x == "bow" else "TF-IDF"
    )
    
    model = st.sidebar.selectbox(
        "Select Model",
        options=["ann", "svm", "lr", "nb_bern", "nb_multi", "nb_guass"],
        format_func=lambda x: {
            "ann": "Artificial Neural Network",
            "svm": "Support Vector Machine",
            "lr": "Logistic Regression",
            "nb_bern": "Bernoulli Naive Bayes",
            "nb_multi": "Multinomial Naive Bayes",
            "nb_guass": "Gaussian Naive Bayes"
        }[x]
    )
    
    # Create analyze button
    if st.button("Analyze Sentiment"):
        if text_input:
            with st.spinner("Analyzing sentiment..."):
                result = predict_sentiment(text_input, vectorizer, model, models)
                
                # Display results
                st.subheader("Analysis Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    sentiment_icon = "😃" if result["sentiment"] == "Positive" else "😔"
                    st.markdown(f"### Sentiment: {result['sentiment']} {sentiment_icon}")
                
                with col2:
                    confidence = float(result["confidence"])
                    st.markdown(f"### Confidence: {confidence:.4f}")
                    
                    # Create a progress bar for confidence
                    st.progress(confidence if result["sentiment"] == "Positive" else 1 - confidence)
                
                # Display model info
                st.subheader("Model Information")
                st.info(f"Using {vectorizer.upper()} vectorizer with {model.upper()} model")
        else:
            st.error("Please enter some text to analyze.")
            
if __name__ == "__main__":
    main()