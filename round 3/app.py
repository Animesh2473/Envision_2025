from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__, template_folder=os.path.join(os.getcwd(), 'templates'))
# Load ML models
recommendation_model = pickle.load(open("random_forest_model_1.pkl", "rb"))

# Load dataset
data = pd.read_csv("cleaned_datathon.csv")

# Home route
@app.route("/")
def home():
    return render_template(r'''C:\Users\shtryash\OneDrive\Desktop\python programing code\Flask\new\templates\index.html''')

# Recommendation API
@app.route("/recommend", methods=["POST"])
def recommend():
    common_name = request.form.get("common_name")
    if not common_name:
        return jsonify({"error": "common_name is required"}), 400

    filtered_data = data[data["common_name"] == common_name]
    if filtered_data.empty:
        return jsonify({"message": "No recommendations found"}), 404

    recommendations = filtered_data[["product_name", "Rate", "product_price"]].head(5)
    return recommendations.to_json(orient="records")

# Pricing Prediction API
@app.route("/predict_price", methods=["POST"])
def predict_price():
    try:
        rate = float(request.form.get("rate"))
        sentiment_code = int(request.form.get("sentiment_code"))
        review = request.form.get("review", "")
        common_name = request.form.get("common_name")

        if not common_name:
            return jsonify({"error": "common_name is required"}), 400

        # Format input for the model
        input_data = [[rate, sentiment_code, len(review), common_name]]
        predicted_price = pricing_model.predict(input_data)
        return jsonify({"predicted_price": predicted_price[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Popularity Prediction API
@app.route("/predict_popularity", methods=["POST"])
def predict_popularity():
    try:
        rate = float(request.form.get("rate"))
        text = request.form.get("text", "")
        common_name = request.form.get("common_name")
        product_price = float(request.form.get("product_price"))

        if not common_name:
            return jsonify({"error": "common_name is required"}), 400

        # Format input for the model
        input_data = [[rate, len(text), common_name, product_price]]
        predicted_popularity = popularity_model.predict(input_data)
        return jsonify({"predicted_popularity": predicted_popularity[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Fraud Detection API
@app.route("/detect_review", methods=["POST"])
def detect_review():
    try:
        text = request.form.get("text", "")
        rate = float(request.form.get("rate"))
        sentiment_code = int(request.form.get("sentiment_code"))

        # Format input for the model
        input_data = [[len(text), rate, sentiment_code]]
        is_fraud = review_detection_model.predict(input_data)
        return jsonify({"is_fraud": bool(is_fraud[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":  # Corrected the comparison
    app.run(debug=True)
