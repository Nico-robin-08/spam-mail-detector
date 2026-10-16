import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📧 Spam Mail / SMS Detector (NTCC Project)")

# File uploader (expects CSV with text and label columns)
uploaded_file = st.file_uploader("Upload your dataset (CSV format with message and label columns)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')
    
    # Clean up standard columns if using the Kaggle SMS Spam dataset
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Tabs for the workflow
    tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🛠 Training", "📈 Evaluation", "🔍 Live Test"])

    with tab1:
        st.header("Exploratory Data Analysis")
        st.write(f"Total Rows: {df.shape[0]}")
        st.write("Label Counts:")
        st.write(df['label'].value_counts())
        
        st.subheader("Spam vs Ham Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='label', data=df, ax=ax, palette="Set2")
        st.pyplot(fig)

    with tab2:
        st.header("Model Training")
        
        # Encode labels (ham: 0, spam: 1)
        df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
        df = df.dropna(subset=['message', 'label_num'])

        X = df['message']
        y = df['label_num']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # TF-IDF Vectorizer (Crucial for text data)
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Train Models
        nb_model = MultinomialNB()
        nb_model.fit(X_train_vec, y_train)

        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train_vec, y_train)

        st.success("✅ Models trained successfully using TF-IDF features!")
        
        # Save objects in session state for other tabs
        st.session_state['vectorizer'] = vectorizer
        st.session_state['nb_model'] = nb_model
        st.session_state['lr_model'] = lr_model
        st.session_state['X_test_vec'] = X_test_vec
        st.session_state['y_test'] = y_test

    with tab3:
        st.header("Model Evaluation")
        if 'nb_model' not in st.session_state:
            st.warning("Please train the models first in the Training tab.")
        else:
            y_test = st.session_state['y_test']
            X_test_vec = st.session_state['X_test_vec']
            
            nb_model = st.session_state['nb_model']
            lr_model = st.session_state['lr_model']

            # Multinomial Naive Bayes Evaluation
            st.subheader("1. Multinomial Naive Bayes")
            nb_pred = nb_model.predict(X_test_vec)
            st.write(f"**Accuracy:** {accuracy_score(y_test, nb_pred):.4f}")
            st.text(classification_report(y_test, nb_pred))

            # Logistic Regression Evaluation
            st.subheader("2. Logistic Regression")
            lr_pred = lr_model.predict(X_test_vec)
            st.write(f"**Accuracy:** {accuracy_score(y_test, lr_pred):.4f}")
            st.text(classification_report(y_test, lr_pred))

    with tab4:
        st.header("Test with Your Own Email / SMS")
        if 'nb_model' not in st.session_state:
            st.warning("Please train the models first.")
        else:
            user_input = st.text_area("Enter email or message text:")
            selected_model = st.selectbox("Choose Model", ["Multinomial Naive Bayes", "Logistic Regression"])
            
            if st.button("Predict"):
                vec_input = st.session_state['vectorizer'].transform([user_input])
                model = st.session_state['nb_model'] if selected_model == "Multinomial Naive Bayes" else st.session_state['lr_model']
                
                prediction = model.predict(vec_input)[0]
                result = "🚨 SPAM" if prediction == 1 else "✅ HAM (Legitimate)"
                
                st.subheader(f"Prediction: {result}")