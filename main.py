import streamlit as st
import pickle
import numpy as np
from difflib import get_close_matches

# ----------------------
# Load Data
# ----------------------
@st.cache_data
def load_data():
    popular_df = pickle.load(open('popular.pkl', 'rb'))
    pt = pickle.load(open('pt.pkl', 'rb'))
    books = pickle.load(open('books.pkl', 'rb'))
    similarity_scores = pickle.load(open('similarity_scores.pkl', 'rb'))
    return popular_df, pt, books, similarity_scores

popular_df, pt, books, similarity_scores = load_data()

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="📚 Book Recommender", page_icon="📚")
st.title("📚 Book Recommender System")

# Sidebar: show top 5 popular books
st.sidebar.header("Popular Books")
for i in range(min(5, len(popular_df))):
    st.sidebar.text(f"{popular_df['Book-Title'].values[i]} by {popular_df['Book-Author'].values[i]}")
    st.sidebar.image(popular_df['Image-URL-M'].values[i], width=100)
    st.sidebar.write(f"⭐ {popular_df['avg_rating'].values[i]} ({popular_df['num_ratings'].values[i]} votes)")

# Dropdown for exact match or text input
book_input = st.selectbox("Select a book", pt.index)

# ----------------------
# Recommendation logic
# ----------------------
def recommend(book_name):
    if book_name not in pt.index:
        # Fuzzy match suggestion
        matches = get_close_matches(book_name, pt.index, n=3, cutoff=0.5)
        if matches:
            return None, f"Book not found. Did you mean: {', '.join(matches)}?"
        else:
            return None, "Book not found. Try another title."

    index = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(
        list(enumerate(similarity_scores[index])),
        key=lambda x: x[1],
        reverse=True
    )[1:6]  # Top 5 recommendations

    recommendations = []
    for i in similar_items:
        temp_df = books[books['Book-Title'] == pt.index[i[0]]].drop_duplicates('Book-Title')
        item = {
            "title": temp_df['Book-Title'].values[0],
            "author": temp_df['Book-Author'].values[0],
            "image": temp_df['Image-URL-M'].values[0] if temp_df['Image-URL-M'].values[0] else "https://via.placeholder.com/150x200?text=No+Image"
        }
        recommendations.append(item)
    return recommendations, None

# ----------------------
# Show recommendations
# ----------------------
if st.button("Get Recommendations"):
    recommendations, error = recommend(book_input)
    if error:
        st.error(error)
    else:
        st.subheader("Recommended Books:")
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            if idx < len(recommendations):
                col.image(recommendations[idx]["image"], width=150)
                col.write(recommendations[idx]["title"])
                col.write(recommendations[idx]["author"])
