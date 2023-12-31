import streamlit as st
import pickle
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.tokenize import word_tokenize
import nltk
import itertools

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Loading the medicine dataset
medicines_dict = pickle.load(open("medicines.pkl", "rb"))
medicines = pd.DataFrame(medicines_dict)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess_tags(tags):
    """
    Function to preprocess the tags
    Args:
    tags : str : input string to be preprocessed

    Returns:
    str : preprocessed string
    """
    tokens = word_tokenize(tags)
    filtered_tokens = [
        lemmatizer.lemmatize(token) for token in tokens if token not in stop_words
    ]
    return " ".join(filtered_tokens)


# Preprocessing the tags of the medicines
medicines["processed_tags"] = medicines["tags"].apply(preprocess_tags)

# Initializing the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(medicines["processed_tags"])


def search_medicine(search_phrase):
    """
    Function to search for medicine based on a search phrase
    Args:
    search_phrase : str : input search phrase

    Returns:
    DataFrame : top 10 most similar medicines
    """
    search_phrase = preprocess_tags(search_phrase)
    search_vector = tfidf_vectorizer.transform([search_phrase])
    cosine_similarities = linear_kernel(search_vector, tfidf_matrix).flatten()
    related_medicines_indices = cosine_similarities.argsort()[:-50:-1]
    matching_medicines = medicines.iloc[related_medicines_indices]
    top_medicines = matching_medicines[
        [
            "Medicine Name",
            "Composition",
            "Manufacturer",
            "Image URL",
            "Average Review %",
            "processed_tags",
        ]
    ]
    top_medicines["processed_tags"] = top_medicines["processed_tags"].apply(
        lambda x: x.split()
    )
    top_medicines = top_medicines.rename(
        columns={
            "Medicine Name": "MedicineName",
            "Composition": "Composition",
            "Manufacturer": "Manufacturer",
            "Image URL": "imageurl",
            "Average Review %": "AverageReview",
            "processed_tags": "ProcessedTags",
        }
    )
    return top_medicines


# Setting up the Streamlit interface
st.set_page_config(
    page_title="Medicine Recommender - Your Go-to Medi Assist!",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.linkedin.com/in/riyarakshit/",
        "Report a bug": "https://www.linkedin.com/in/riyarakshit/",
        "About": "This is a *beginner-friendly* project developed as a part of learning NLP!",
    },
)
st.title("Your Go-To Medi 💊 Assist!")
symptoms = st.text_input("Enter your symptoms 👇")


def recommend(text):
    """
    Function to recommend medicines based on text
    Args:
    text : str : input text

    Returns:
    DataFrame : recommended medicines
    """
    result = search_medicine(text)
    return result


if st.button("Discover Medicines 👀"):
    medicines_recommended = recommend(symptoms)
    st.write(
        "Here is a list of recommendations, but please consult a physician 👨‍⚕️👩‍⚕️ before consuming these:"
    )
    col1, col2, col3 = st.columns(3)
    cols = itertools.cycle([col1, col2, col3])
    for row in medicines_recommended.itertuples():
        col = next(cols)
        col.markdown(
            f"""
                <div style="
                    background-color: #F1F3F8;
                    border-radius: 5px;
                    padding: 10px;
                    margin: 10px;
                ">
                    <img src={row.imageurl} style="max-width:200px; max-height:200px; padding:10px;" />
                    <div>
                        <h3 style="color: #393B44;padding:0;">{row.MedicineName}</h3>
                        <h5 style="color: #8D93AB;padding:0;">By: {row.Manufacturer}</h5>
                        <p style="color: #393B44;">Mainly contains: {row.Composition}%</p>
                        <p style="color: #393B44;">Average Review: {row.AverageReview}%</p>
                        <p> 
                            {' '.join([f'<span style="background-color: #D6E0F0; border-radius: 5px; padding: 4px; color:#393B44;">{tag}</span>' for tag in row.ProcessedTags])}
                        </p>
                    </div>
                </div>
                """,
            unsafe_allow_html=True,
        )
