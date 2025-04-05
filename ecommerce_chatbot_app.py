import streamlit as st
import pandas as pd
import re
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
@st.cache_data
def load_data():
    products = pd.read_csv("products.csv")
    orders = pd.read_csv("orders.csv")
    reviews = pd.read_csv("product_reviews.csv")
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
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    return model, vectorizer, (acc, prec, rec, f1)

def chatbot_response(user_input, products, orders, reviews, model, vectorizer, performance_metrics):
    user_input = user_input.lower()

    if "search" in user_input or "find" in user_input:
        search_words = re.findall(r'\b\w+\b', user_input)
        found = pd.DataFrame()
        for word in search_words:
            found = pd.concat([found, products[products['name'].str.lower().str.contains(word)]])
        found = found.drop_duplicates()

        if not found.empty:
            st.subheader("Search Results")
            st.table(found[["name", "price"]])
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
        st.table(sample[["name", "price"]])
        return "Here are some product recommendations."

    elif "performance" in user_input or "metric" in user_input:
        acc, prec, rec, f1 = performance_metrics
        st.subheader("Model Performance")
        perf_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Value": [acc, prec, rec, f1]
        })
        st.table(perf_df)
        return f"Model Performance:\nAccuracy: {acc:.2f}\nPrecision: {prec:.2f}\nRecall: {rec:.2f}\nF1 Score: {f1:.2f}"

    elif "reviews" in user_input:
        product_name = re.findall(r'"([^"]*)"', user_input)
        if product_name:
            product_reviews = get_product_reviews(product_name[0], products, reviews)
            return product_reviews
        else:
            return "Please provide the product name in quotes to see reviews."

    elif "category" in user_input:
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
            return "Please provide the category name in quotes and the number of products (e.g., top 5 category \"Electronics\")."

    else:
        return "I can help with product search, order tracking, recommendations, reviews, category search, or top category products. Try typing: top 5 category \"Electronics\""

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
        review_text += f"Rating: {review['rating']}, Review: {review['reviews']}\n"
    return review_text

def get_products_in_category(category_name, products):
    category_products = products[products['category'].str.lower() == category_name.lower()]
    if category_products.empty:
        return "Category not found."

    st.subheader(f"Products in {category_name}")
    st.table(category_products[["name", "price"]])
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
            if total_reviews > 0:
                positive_ratio = positive_count / total_reviews
            else:
                positive_ratio = 0
            product_sentiments.append({
                "Product Name": product['name'],
                "Positive Sentiment Ratio": round(positive_ratio, 2),
                "Price ($)": product['price']
            })

    sorted_products = sorted(product_sentiments, key=lambda x: x["Positive Sentiment Ratio"], reverse=True)[:n]

    if not sorted_products:
        return "No reviews found for products in this category."

    df = pd.DataFrame(sorted_products)
    st.subheader(f"Top {n} Products in {category_name}")
    st.table(df)

    return f"Showing top {n} products in {category_name} based on sentiment ratings."

# App Layout
st.title("\U0001F6D2 E-commerce Chatbot")

products, orders, reviews = load_data()
model, vectorizer, performance_metrics = train_sentiment_model(reviews)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Product Comparison Feature
st.sidebar.subheader("Product Comparison")
product_names = products['name'].tolist()
selected_product1 = st.sidebar.selectbox("Select Product 1", product_names)
selected_product2 = st.sidebar.selectbox("Select Product 2", product_names, index=1)

if st.sidebar.button("Compare Products"):
    product1 = products[products['name'] == selected_product1].iloc[0]
    product2 = products[products['name'] == selected_product2].iloc[0]

    product1_reviews = reviews[reviews['product_id'] == product1['product_id']]['reviews'].fillna('').tolist()
    product2_reviews = reviews[reviews['product_id'] == product2['product_id']]['reviews'].fillna('').tolist()

    if product1_reviews:
        product1_vec = vectorizer.transform(product1_reviews)
        product1_sentiments = model.predict(product1_vec)
        product1_positive_ratio = sum(1 for sentiment in product1_sentiments if sentiment == 'positive') / len(product1_sentiments)
    else:
        product1_positive_ratio = 0

    if product2_reviews:
        product2_vec = vectorizer.transform(product2_reviews)
        product2_sentiments = model.predict(product2_vec)
        product2_positive_ratio = sum(1 for sentiment in product2_sentiments if sentiment == 'positive') / len(product2_sentiments)
    else:
        product2_positive_ratio = 0

    comparison_df = pd.DataFrame({
        "Feature": ["Name", "Category", "Price", "Positive Sentiment Ratio"],
        selected_product1: [product1['name'], product1['category'], product1['price'], round(product1_positive_ratio, 2)],
        selected_product2: [product2['name'], product2['category'], product2['price'], round(product2_positive_ratio, 2)]
    })
    st.subheader("Product Comparison")
    st.table(comparison_df)

    if product1_positive_ratio > product2_positive_ratio:
        st.write(f"Based on sentiment, {selected_product1} is preferred.")
    elif product2_positive_ratio > product1_positive_ratio:
        st.write(f"Based on sentiment, {selected_product2} is preferred.")
    else:
        st.write("Both products have similar sentiment scores.")

user_input = st.text_input("You:", key="chat_input")
st.markdown("### Try these sample queries:")
st.markdown("- `top 5 category \"Electronics\"` - Get top 5 products in Electronics")
st.markdown("- `search \"Portable Cream \"` - Find products related to Portable Cream")
st.markdown("- `track order 1001` - Check the status of order 1001")
st.markdown("- `recommend products` - Get product recommendations")
st.markdown("- `show reviews \"Portable Cream\"` - Get reviews for Portable Cream")

if user_input:
    if "top" in user_input.lower() and "category" in user_input.lower():
        category_name = re.findall(r'"([^"]*)"', user_input)
        num_products = re.findall(r'\d+', user_input)
        if category_name and num_products:
            table_message = get_top_n_products_by_sentiment(category_name[0], int(num_products[0]), products, reviews, model, vectorizer)
            response = table_message
        else:
            response = "Please provide the category name in quotes and the number of products (e.g., top 5 category \"Electronics\")."
    else:
        response = chatbot_response(user_input, products, orders, reviews, model, vectorizer, performance_metrics)

    st.session_state.chat_history.insert(0, ("Bot", response))
    st.session_state.chat_history.insert(0, ("User", user_input))

for sender, message in reversed(st.session_state.chat_history):
    if sender == "User":
        st.markdown(f"*You:* {message}")
    else:
        st.markdown(f"\U0001F9E0 *Bot:* {message}")