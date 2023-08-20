import os
import numpy as np
import pandas as pd
import nltk
import webbrowser
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from flask import Flask, render_template, request
import ast
from flask import jsonify

feedback_data = []
feedbacks = []

def store_feedback(user_id, outfit_id, feedback):
    feedback_entry = {
        "user_id": user_id,
        "outfit_id": outfit_id,
        "feedback": feedback
    }
    feedback_data.append(feedback_entry)

app = Flask(__name__)

# Model Persistence
def save_model(model, filename):
    model.save(filename)

def load_genai_model(filename):
    return load_model(filename)

# Error Handling for File Reading
def safe_read_csv(filename):
    if os.path.exists(filename):
        try:
            return pd.read_csv(filename)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return None
    else:
        print(f"{filename} does not exist.")
        return None

# Downloading necessary datasets for nltk
nltk.download('stopwords')
nltk.download('wordnet')

@app.route('/form')
def index():
    return render_template('form.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data['user_id']
    user_input = data['user_input']
    occasion = data['occasion']
    weather = data['weather']
    budget = int(data['budget'])
    
    # Load the user data and social media data with error handling
    user_data = safe_read_csv("synthetic_user_data.csv")
    social_media_data = safe_read_csv("dataset.csv")

    # Preprocess the user data
    user_data = preprocess_user_data(user_data)

    # Train the GenAI model or load a pre-trained model
    model, word2vec_model, social_media_data, label_encoder = train_genai_model(user_data, social_media_data)

    # Generate the outfit recommendations
    matching_outfits = generate_outfit(model, word2vec_model, social_media_data, label_encoder, user_input, occasion, weather, budget)

    # Convert numpy arrays to lists for JSON serialization
    for column in matching_outfits.columns:
        if matching_outfits[column].apply(type).eq(np.ndarray).any():
            matching_outfits.loc[:, column] = matching_outfits[column].apply(list)

    # Convert the recommendations to a list of dictionaries for easy JSON serialization
    outfits_list = matching_outfits.to_dict(orient='records')
    
    return jsonify(outfits_list)

# function to save the feedback data to a CSV file
def save_feedback_to_csv():
    df = pd.DataFrame(feedback_data)
    df.to_csv('feedback_data.csv', index=False)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    user_id = data.get('user_id')
    outfit_id = data.get('outfit_id')
    feedback_text = data.get('feedback')

    # Store the feedback in the feedback_data list
    store_feedback(user_id, outfit_id, feedback_text)

    # Save feedback data to CSV
    save_feedback_to_csv()

    return jsonify({"message": "Feedback submitted successfully!"})

# 1. Data Preprocessing
def preprocess_user_data(user_data):
    user_data["past_purchase_history"] = user_data["past_purchase_history"].apply(ast.literal_eval)
    
    stop_words = set(nltk.corpus.stopwords.words("english"))
    user_data["past_purchase_history"] = user_data["past_purchase_history"].apply(
        lambda x: [word for word in x if word not in stop_words])

    stemmer = PorterStemmer()
    user_data["past_purchase_history"] = user_data["past_purchase_history"].apply(
        lambda x: [stemmer.stem(word) for word in x])

    lemmatizer = nltk.stem.WordNetLemmatizer()
    user_data["past_purchase_history"] = user_data["past_purchase_history"].apply(
        lambda x: [lemmatizer.lemmatize(word) for word in x])

    return user_data

# 2. Model Training
def train_genai_model(user_data, social_media_data):
    hashtags = ["#ootd"]
    hashtags_series = pd.Series(hashtags)
    user_data["past_purchase_history"] = pd.concat([user_data["past_purchase_history"], hashtags_series], ignore_index=True)

    word2vec_model = Word2Vec(social_media_data["style"].apply(lambda x: [x]), vector_size=100, window=5, min_count=1, workers=4)

    epochs = 10
    word2vec_model.train(user_data["past_purchase_history"].apply(lambda x: str(x)), total_examples=word2vec_model.corpus_count, epochs=epochs)

    def get_vector_or_default(word, model):
        if word in model.wv:
            return model.get_vector(word)
        else:
            return np.zeros(model.vector_size)

    social_media_data["feature_vector"] = social_media_data["hashtag"].apply(
        lambda x: get_vector_or_default(x, word2vec_model))

    label_encoder = LabelEncoder()
    social_media_data["style"] = label_encoder.fit_transform(social_media_data["style"])

    X = np.stack(social_media_data["feature_vector"].to_numpy())

    num_classes = len(np.unique(social_media_data["style"]))
    y = to_categorical(social_media_data["style"], num_classes=num_classes)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Dense(128, activation="relu", input_shape=(100,)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[EarlyStopping(patience=5)])

    return model, word2vec_model, social_media_data, label_encoder

# 3. Outfit Generation
def generate_outfit(model, word2vec_model, social_media_data, label_encoder, user_input, occasion, weather, budget):
    if user_input in word2vec_model.wv:
        predicted_probs = model.predict(np.array([word2vec_model.wv[user_input]]))
        predicted_style_indices = np.argsort(predicted_probs, axis=-1)[0]
        predicted_style = predicted_style_indices[-1]
    else:
        print(f"The word '{user_input}' is not recognized. Please try a different input.")
        return pd.DataFrame()

    matching_outfits = social_media_data[
        (social_media_data["style"] == predicted_style) &
        (social_media_data["occasion"] == occasion) &
        (social_media_data["weather"] == weather) &
        (social_media_data["budget"] <= budget)
    ]
    
    decoded_styles = label_encoder.inverse_transform(matching_outfits["style"])

    words = []
    for style in decoded_styles:
        words.append(style)
    return matching_outfits

# 4. Chatbot Integration
class FashionChatbot:
    def __init__(self):
        self.greetings = ["Hello!", "Hi there!", "Greetings!"]
    
    def greet_user(self):
        return np.random.choice(self.greetings)

# 5. Social Media Sharing
def share_on_social_media(user_id, outfit_id, platform="Instagram"):
    print(f"Sharing outfit {outfit_id} for user {user_id} on {platform}...")

# 6. User Reviews and Community Integration
def leave_review(user_id, outfit_id, review_text, rating):
    print(f"User {user_id} left a review for outfit {outfit_id}: {review_text} with a rating of {rating}/5")

# 8. Personalized User Profiles
def create_user_profile(user_id, preferences):
    print(f"Creating profile for user {user_id} with preferences: {preferences}")

# 9. Trend Analysis
def analyze_trends(social_media_data):
    popular_styles = social_media_data['style'].value_counts().idxmax()
    popular_colors = social_media_data['color'].value_counts().idxmax()
    popular_brands = social_media_data['brand'].value_counts().idxmax()

    return popular_styles, popular_colors, popular_brands

# 10. Seasonal Recommendations
def seasonal_recommendations(season, user_data):
    # Recommend outfits based on the season and user's past purchase history
    seasonal_outfits = user_data[user_data['season'] == season]  
    return seasonal_outfits

# 11. Occasion-based Recommendations
def occasion_based_recommendations(occasion, user_data):
    # Recommend outfits based on the occasion and user's past purchase history
    occasion_outfits = user_data[user_data['occasion'] == occasion]
    return occasion_outfits

# 12. Feedback-driven Model Retraining
def retrain_model_on_feedback(feedback_data, model):
    # Logic to retrain the model based on the feedback data.
    # This can involve adjusting weights, adding more layers, or even changing the model architecture.
    pass

def interact_with_user(user_id, user_data, social_media_data, model, word2vec_model, label_encoder):
    user_input = input("What kind of outfit are you looking for?")
    occasion = input("What is the occasion?")
    weather = input("What is the weather?")
    budget = int(input("What is your budget?"))

    # Generate the outfit.
    matching_outfits = generate_outfit(model, word2vec_model, social_media_data, label_encoder, user_input, occasion, weather, budget)

    # If generate_outfit returns None or an empty list/DataFrame
    if matching_outfits.empty:
        print("Sorry, I couldn't find any matching outfits for you.")
        return

    display_outfit_recommendations(matching_outfits)
    
    # Collect feedback
    feedback = input("Do you like the recommended outfit? (yes/no): ")
    if feedback.lower() == "no":
        detailed_feedback = collect_detailed_feedback(matching_outfits.iloc[0])
        store_feedback(user_id, matching_outfits.iloc[0]['outfit_id'], detailed_feedback)  # Assuming 'outfit_id' is a column in your dataset
        user_data = adjust_recommendations_based_on_feedback(detailed_feedback, user_data)
        
        # Generate a new set of recommendations based on feedback
        matching_outfits = generate_outfit(model, word2vec_model, social_media_data, label_encoder, user_input, occasion, weather, budget)
        
        # Return the results to the frontend
        return render_template('results.html', outfits=matching_outfits)

        # Display the new recommendations
        if matching_outfits.empty:
            print("Sorry, I couldn't find any matching outfits based on your feedback.")
            return

        display_outfit_recommendations(matching_outfits)
    else:
        print("Thank you for choosing GenAI for your outfit recommendation!")

        # Ask the user if they want to try again
        try_again = input("Would you like to try again? (yes/no): ")
        if try_again.lower() == "yes":
            return interact_with_user(user_id, user_data, social_media_data, model, word2vec_model, label_encoder)
        else:
            print("Thank you for using the outfit generator!")
            return

    # If the user likes the outfit, ask them if they want to share it on social media or leave a review.
    if feedback.lower() == "yes":
        share_on_social_media(user_id, matching_outfits.iloc[0]['outfit_id'])
        leave_review(user_id, matching_outfits.iloc[0]['outfit_id'], "This outfit is amazing!", 5)

        print("Thank you for using the outfit generator! I hope you enjoy your outfit.")
        return

# 14. Regional Preferences
def regional_recommendations(region, user_data):
    # Recommend outfits based on the user's region
    regional_outfits = user_data[user_data['region'] == region]  # Assuming there's a 'region' column in the dataset
    return regional_outfits

# 15. Age-based Recommendations
def age_based_recommendations(age, user_data):
    # Recommend outfits based on the user's age
    if age < 25:
        age_group = 'young'
    elif 25 <= age < 50:
        age_group = 'middle-aged'
    else:
        age_group = 'senior'
    age_outfits = user_data[user_data['age_group'] == age_group]  # Assuming there's an 'age_group' column in the dataset
    return age_outfits

# 16. Body Type Recommendations
def body_type_recommendations(body_type, user_data):
    # Recommend outfits based on the user's body type
    body_type_outfits = user_data[user_data['body_type'] == body_type]  # Assuming there's a 'body_type' column in the dataset
    return body_type_outfits

# 17. Accessory Recommendations
def accessory_recommendations(outfit_id, user_data):
    # Recommend accessories that go well with a given outfit
    outfit = user_data[user_data['outfit_id'] == outfit_id].iloc[0]  # Assuming there's an 'outfit_id' column in the dataset
    accessory_type = outfit['accessory_type']  # Assuming there's an 'accessory_type' column in the dataset
    matching_accessories = user_data[user_data['accessory_type'] == accessory_type]
    return matching_accessories

# 18. Footwear Recommendations
def footwear_recommendations(outfit_id, user_data):
    # Recommend footwear that goes well with a given outfit
    outfit = user_data[user_data['outfit_id'] == outfit_id].iloc[0]
    footwear_type = outfit['footwear_type']  
    matching_footwear = user_data[user_data['footwear_type'] == footwear_type]
    return matching_footwear

# 19. Brand Preferences
def brand_based_recommendations(brand, user_data):
    # Recommend outfits based on the user's favorite brand
    brand_outfits = user_data[user_data['brand'] == brand]  
    return brand_outfits

# 20. Color Preferences
def color_based_recommendations(color, user_data):
    # Recommend outfits based on the user's favorite color
    color_outfits = user_data[user_data['color'] == color]  
    return color_outfits

# 21. Style Preferences
def style_based_recommendations(style, user_data):
    # Recommend outfits based on the user's favorite style
    style_outfits = user_data[user_data['style'] == style]  
    return style_outfits

# 22. Feedback-driven Outfit Tweaking
def tweak_outfit(outfit_id, feedback, user_data):
    # Modify an outfit based on user feedback
    # This is a placeholder and will need actual implementation based on your data and feedback structure.
    print(f"Modifying outfit {outfit_id} based on feedback: {feedback}")
    return user_data

# 23. Social Media Trend Integration
def integrate_social_media_trends(social_media_data, user_data):
    # Integrate social media trends into the recommendation system
    # This is a placeholder and will need actual implementation based on your data and social media trends.
    print("Integrating social media trends...")
    return user_data

# 24. User Browsing Data Integration
def integrate_browsing_data(browsing_data, user_data):
    # Integrate user's browsing data into the recommendation system
    # This is a placeholder and will need actual implementation based on your data and browsing history.
    print("Integrating user browsing data...")
    return user_data

# 25. Personalized User Dashboard
def display_user_dashboard(user_id, user_data):
    # Display a dashboard with personalized outfit recommendations for the user
    # This is a placeholder and will need actual implementation based on your data and user preferences.
    print(f"Displaying dashboard for user {user_id}...")

    # Collect feedback
    feedback = input("Do you have any feedback on the recommendations?")
    collect_feedback(user_id="sample_user_id", outfit_id="sample_outfit_id", feedback=feedback)  # Adjust user_id and outfit_id as needed.

def tweak_outfit_recommendation(feedback, recommended_outfit):
    # Placeholder function to tweak outfit based on feedback
    # This needs to be implemented based on your dataset and logic
    return recommended_outfit

def display_outfit_recommendations(matching_outfits):
    """Displays the outfit recommendations and provides options to the user."""
    index = 0
    while index < len(matching_outfits):
        outfit = matching_outfits.iloc[index]
        
        # Display the outfit description
        print(outfit["hashtag"])  # Assuming there's a 'description' column in the dataset
        
        # Check if the image path is valid and display the image
        image_path = outfit["Image Link"]
        if isinstance(image_path, str) and os.path.exists(image_path):
            plt.imshow(plt.imread(image_path))
            plt.axis('off')
            plt.show()
        else:
            print("Image not available.")
        
        # Provide options to the user
        print(f"Buy Now: {outfit['Purchase Link']}")
        action = input("Options: (buy/next/quit): ").lower()
        
        if action == "buy":
            webbrowser.open(outfit['Purchase Link'])
            break
        elif action == "next":
            index += 1
            if index == len(matching_outfits):
                print("Sorry, there are no more recommendations.")
                break
        elif action == "quit":
            print("Thank you for using Fashion GenAI")
            break
        else:
            print("Invalid input. Please enter 'buy' or 'next'.")

# Enhance the Feedback Collection:
def collect_detailed_feedback(outfit):
    feedback_categories = {
        "style": "Do you like the style of the outfit?",
        "color": "Do you like the color of the outfit?",
        "brand": "Do you like the brand of the outfit?",
        # We can add more categories as needed
    }
    
    feedback_responses = {}
    for category, question in feedback_categories.items():
        response = input(f"{question} (yes/no): ").lower()
        feedback_responses[category] = response
    
    return feedback_responses

# Adjust Recommendations Based on Feedback:
def adjust_recommendations_based_on_feedback(feedback, user_data):
    for category, response in feedback.items():
        if response == "no":
            # Logic to adjust user_data based on feedback
            # For example, if user doesn't like a particular style:
            user_data = user_data[user_data[category] != feedback[category]]
    return user_data

def store_feedback(user_id, outfit_id, feedback):
    feedback_entry = {
        "user_id": user_id,
        "outfit_id": outfit_id,
        "feedback": feedback
    }
    feedback_data.append(feedback_entry)
    # Optionally, save to a database or file for long-term storage

def apply_stored_feedback(user_id, user_data):
    user_feedbacks = [f for f in feedback_data if f["user_id"] == user_id]
    for feedback in user_feedbacks:
        # Logic to adjust user_data based on stored feedback
        # For example, if a user consistently doesn't like a particular style:
        user_data = user_data[user_data["style"] != feedback["feedback"]["style"]]
    return user_data


    # Prompt the user for their user ID at the beginning
    user_id = input("Please enter your user ID: ")
    # We have plans to allow users to login through flipkart accont so that this model can recommend according to their style. 

    # Load the user data and social media data with error handling
    user_data = safe_read_csv("synthetic_user_data.csv")
    social_media_data = safe_read_csv("dataset.csv")

    # Data Validation (simple check for null values)
    if user_data.isnull().any().any() or social_media_data.isnull().any().any():
        print("Data contains missing values. Please clean the data before proceeding.")
        exit()
    
    # Apply stored feedback for the user
    user_data = apply_stored_feedback(user_id, user_data)

    if user_data is None or social_media_data is None:
        print("Error loading data. Exiting.")
        exit()

    # Check for expected columns in dataframes
    expected_columns_user = ['past_purchase_history', 'region', 'age', 'body_type', 'brand', 'color', 'style']
    expected_columns_social = ['style', 'Image Link', 'Purchase Link']
    for col in expected_columns_user:
        if col not in user_data.columns:
            print(f"Expected column {col} not found in user data. Exiting.")
            exit()
    for col in expected_columns_social:
        if col not in social_media_data.columns:
            print(f"Expected column {col} not found in social media data. Exiting.")
            exit()

    # Preprocess the user data.
    user_data = preprocess_user_data(user_data)

    # Train the GenAI model.
    model, word2vec_model, social_media_data, label_encoder = train_genai_model(user_data, social_media_data)

    # Interact with the user
    matching_outfits = interact_with_user(user_id, user_data, social_media_data, model, word2vec_model, label_encoder)
    
    #We have added few sections for future work, We can add these features asap if needed, We are ready with the rough codes, If neended than we can work with datasets and get through these.

    # Utilizing more functions:
    # Assuming you have a 'region' and 'age' column in your user_data
    region = user_data['region'].iloc[0]
    age = user_data['age'].iloc[0]
    
    # Regional Recommendations
    recommended_regional_outfits = regional_recommendations(region, user_data)
    print("Recommended outfits based on your region:", recommended_regional_outfits)

    # Age-based Recommendations
    recommended_age_outfits = age_based_recommendations(age, user_data)
    print("Recommended outfits based on your age:", recommended_age_outfits)

    # Body Type Recommendations (assuming you have a 'body_type' column in your user_data)
    body_type = user_data['body_type'].iloc[0]
    recommended_body_type_outfits = body_type_recommendations(body_type, user_data)
    print("Recommended outfits based on your body type:", recommended_body_type_outfits)

    # Accessory Recommendations (assuming you have an 'outfit_id' column in your user_data)
    outfit_id = user_data['outfit_id'].iloc[0]
    recommended_accessories = accessory_recommendations(outfit_id, user_data)
    print("Recommended accessories for your outfit:", recommended_accessories)

    # Footwear Recommendations
    recommended_footwear = footwear_recommendations(outfit_id, user_data)
    print("Recommended footwear for your outfit:", recommended_footwear)

    # Brand Preferences (assuming you have a 'brand' column in your user_data)
    brand = user_data['brand'].iloc[0]
    recommended_brand_outfits = brand_based_recommendations(brand, user_data)
    print("Recommended outfits based on your favorite brand:", recommended_brand_outfits)

    # Color Preferences (assuming you have a 'color' column in your user_data)
    color = user_data['color'].iloc[0]
    recommended_color_outfits = color_based_recommendations(color, user_data)
    print("Recommended outfits based on your favorite color:", recommended_color_outfits)

    # Style Preferences (assuming you have a 'style' column in your user_data)
    style = user_data['style'].iloc[0]
    recommended_style_outfits = style_based_recommendations(style, user_data)
    print("Recommended outfits based on your favorite style:", recommended_style_outfits)

    # Social Media Trend Integration
    updated_user_data = integrate_social_media_trends(social_media_data, user_data)
    print("User data after integrating social media trends:", updated_user_data)

    # User Browsing Data Integration (assuming you have a 'browsing_data' column in your user_data)
    browsing_data = user_data['browsing_data'].iloc[0]
    updated_user_data = integrate_browsing_data(browsing_data, user_data)
    print("User data after integrating browsing data:", updated_user_data)

# Main Execution
if __name__ == "__main__":
    app.run(debug=True)