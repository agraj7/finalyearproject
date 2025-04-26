import streamlit as st
import pandas as pd
import re
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import altair as alt
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load data
@st.cache_data
def load_data():
    reviews = pd.read_csv("product_reviews.csv")
    reviews.rename(columns={
        'pid': 'product_id',
        'categories': 'category',
        'name': 'name',
        'reviews': 'reviews'
    }, inplace=True)
    products = reviews[['product_id', 'name', 'category']].drop_duplicates()
    orders = pd.DataFrame({'order_id': [1001, 1002, 1003], 'status': ['Shipped', 'Delivered', 'Processing']})  # mock data
    return products, orders, reviews

def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return 'positive'
    elif polarity > 0.05:
        return 'neutral'
    else:
        return 'negative'

def train_sentiment_model(reviews_df):
    reviews_df['reviews'] = reviews_df['reviews'].fillna('')
    reviews_df['sentiment'] = reviews_df['reviews'].apply(analyze_sentiment)
    reviews_df['clean'] = reviews_df['reviews'].str.lower()
    X_train, X_test, y_train, y_test = train_test_split(reviews_df['clean'], reviews_df['sentiment'], test_size=0.3, random_state=42)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier()
    }

    performance_metrics = {}

    for model_name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        performance_metrics[model_name] = [acc, prec, rec, f1]

    # Return logistic regression model by default
    return models["Logistic Regression"], vectorizer, performance_metrics

def chatbot_response(user_input, products, orders, reviews, model, vectorizer, performance_metrics):
    user_input = user_input.lower()

    if "search" in user_input or "find" in user_input:
        search_words = re.findall(r'\b\w+\b', user_input)
        found = pd.DataFrame()
        for word in search_words:
            found = pd.concat([found, products[products['name'].str.lower().str.contains(word)]]).drop_duplicates()
        if not found.empty:
            st.subheader("Search Results")
            st.table(found[["name"]])
            return "Here are some products I found."
        else:
            return "Sorry, no products found."

    elif "order" in user_input or "status" in user_input:
        order_ids = re.findall(r'\d+', user_input)
        if order_ids:
            order_id = int(order_ids[0])
            match = orders[orders['order_id'] == order_id]
            if not match.empty:
                st.subheader(f"Order {order_id} Status")
                st.write(f"Order {order_id} status: {match.iloc[0]['status']}")
                return f"Order {order_id} status: {match.iloc[0]['status']}"
            else:
                return "Order not found."
        return "Please provide your order ID."

    elif "recommend" in user_input or "suggest" in user_input:
        sample = products.sample(3)
        st.subheader("Product Recommendations")
        st.table(sample[["name"]])
        return "Here are some product recommendations."

    elif "reviews" in user_input:
        product_name = re.findall(r'"([^"]*)"', user_input)
        if product_name:
            product_reviews = get_product_reviews(product_name[0], products, reviews)
            return product_reviews
        else:
            return "Please provide the product name in quotes to see reviews."

    elif "category" in user_input and "top" not in user_input:
        category_name = re.findall(r'"([^"]*)"', user_input)
        if category_name:
            category_products = get_products_in_category(category_name[0], products)
            return category_products
        else:
            return "Please provide the category name in quotes."

    elif "top" in user_input and "category" in user_input:
        category_name = re.findall(r'"([^"]*)"', user_input)
        num_products = re.findall(r'\d+', user_input)
        if category_name and num_products:
            top_products = get_top_n_products_by_sentiment(category_name[0], int(num_products[0]), products, reviews, model, vectorizer)
            return top_products
        else:
            return "Please provide the category name in quotes and the number of products."

    else:
        return "I can help with product search, order tracking, recommendations, reviews, category search, or top category products."

def get_product_reviews(product_name, products, reviews):
    product = products[products['name'].str.lower() == product_name.lower()]
    if product.empty:
        return "Product not found."
    product_id = product.iloc[0]['product_id']
    product_reviews = reviews[reviews['product_id'] == product_id]
    if product_reviews.empty:
        return "No reviews found for this product."
    review_text = "Product Reviews:\n"
    for _, review in product_reviews.iterrows():
        review_text += f"- Review: {review['reviews']}\n"
    return review_text

def get_products_in_category(category_name, products):
    category_products = products[products['category'].str.lower() == category_name.lower()]
    if category_products.empty:
        return "Category not found."
    st.subheader(f"Products in {category_name}")
    st.table(category_products[["name"]])
    return f"Products in {category_name}."

def get_top_n_products_by_sentiment(category_name, n, products, reviews, model, vectorizer):
    category_products = products[products['category'].str.lower() == category_name.lower()]
    if category_products.empty:
        return "Category not found."
    product_sentiments = []
    for _, product in category_products.iterrows():
        product_reviews = reviews[reviews['product_id'] == product['product_id']]
        if not product_reviews.empty:
            reviews_text = product_reviews['reviews'].fillna('').tolist()
            reviews_vec = vectorizer.transform(reviews_text)
            predicted_sentiments = model.predict(reviews_vec)
            positive_count = sum(1 for sentiment in predicted_sentiments if sentiment == 'positive')
            total_reviews = len(predicted_sentiments)
            positive_ratio = positive_count / total_reviews if total_reviews > 0 else 0
            product_sentiments.append({
                "Product Name": product['name'],
                "Positive Sentiment Ratio": round(positive_ratio, 2)
            })
    sorted_products = sorted(product_sentiments, key=lambda x: x["Positive Sentiment Ratio"], reverse=True)[:n]
    if not sorted_products:
        return "No reviews found for products in this category."
    df = pd.DataFrame(sorted_products)
    st.subheader(f"Top {n} Products in {category_name}")
    st.table(df)
    return f"Showing top {n} products in {category_name} based on sentiment ratings."

# Streamlit App Layout
st.title("\U0001F6D2 E-commerce Chatbot")

products, orders, reviews = load_data()
model, vectorizer, performance_metrics = train_sentiment_model(reviews)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show Model Comparison (Table and Graph)
if st.button("Show Model Comparison Results"):
    st.subheader("Sentiment Analysis - Model Performance (Table)")

    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    perf_df = pd.DataFrame(performance_metrics, index=metric_names).T.reset_index().rename(columns={"index": "Model"})
    st.table(perf_df)

    st.subheader("Sentiment Analysis - Model Performance (Line Chart)")
    perf_melted = perf_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    chart = alt.Chart(perf_melted).mark_line(point=True).encode(
        x='Metric:N',
        y='Score:Q',
        color='Model:N'
    ).properties(width=700, height=400)
    st.altair_chart(chart, use_container_width=True)

# Chat UI
user_input = st.text_input("You:", key="chat_input")
st.markdown("### Sample queries:")
st.markdown("- `top 5 category \"Amazon Devices\"`")
st.markdown("- `search Kindle Keyboard`")
st.markdown("- `track order 1001`")
st.markdown("- `recommend products`")
st.markdown("- `show reviews \"Kindle Paperwhite\"`")

if user_input:
    response = chatbot_response(user_input, products, orders, reviews, model, vectorizer, performance_metrics)
    st.session_state.chat_history.insert(0, ("Bot", response))
    st.session_state.chat_history.insert(0, ("User", user_input))

for sender, message in reversed(st.session_state.chat_history):
    if sender == "User":
        st.markdown(f"*You:* {message}")
    else:
        st.markdown(f"\U0001F9E0 *Bot:* {message}")


# Product Comparison in Sidebar
with st.sidebar:
    st.header("Product Comparison")
    product1_name = st.selectbox("Select Product 1", products['name'].tolist())
    product2_name = st.selectbox("Select Product 2", products['name'].tolist())

    if st.button("Compare Products"):
        # Get sentiment scores for both products
        product1 = products[products['name'] == product1_name]
        product2 = products[products['name'] == product2_name]
        
        if product1.empty or product2.empty:
            st.write("One or both products not found.")
        else:
            product1_reviews = reviews[reviews['product_id'] == product1.iloc[0]['product_id']]['reviews'].fillna('').tolist()
            product2_reviews = reviews[reviews['product_id'] == product2.iloc[0]['product_id']]['reviews'].fillna('').tolist()

            # Vectorize the reviews and predict sentiments
            product1_reviews_vec = vectorizer.transform(product1_reviews)
            product2_reviews_vec = vectorizer.transform(product2_reviews)

            product1_pred_sentiments = model.predict(product1_reviews_vec)
            product2_pred_sentiments = model.predict(product2_reviews_vec)

            # Calculate positive sentiment ratio for both products
            product1_positive_ratio = sum(1 for sentiment in product1_pred_sentiments if sentiment == 'positive') / len(product1_pred_sentiments) if len(product1_pred_sentiments) > 0 else 0
            product2_positive_ratio = sum(1 for sentiment in product2_pred_sentiments if sentiment == 'positive') / len(product2_pred_sentiments) if len(product2_pred_sentiments) > 0 else 0

            # Compare and recommend the better product
            if product1_positive_ratio > product2_positive_ratio:
                recommended_product = product1_name
                st.write(f"**{recommended_product}** is better based on sentiment analysis.")
            elif product1_positive_ratio < product2_positive_ratio:
                recommended_product = product2_name
                st.write(f"**{recommended_product}** is better based on sentiment analysis.")
            else:
                st.write("Both products are equally rated based on sentiment analysis.")
