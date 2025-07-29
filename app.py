from flask import Flask, render_template, request, jsonify
import requests
from textblob import TextBlob
import speech_recognition as sr
import csv
import json
import numpy as np
import datetime
import os
from dotenv import load_dotenv
api_key = os.getenv("API_KEY")

app = Flask(__name__)

# Load CRM data
def load_crm_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

CRM_DATA = load_crm_data('crm_data.json')

# Query LLaMA LLM API
def query_llama_llm(user_input, customer_context):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": "Bearer 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",
        "Content-Type": "application/json"
    }

    negotiation_prompt = (
        f"Your name is alexa, an Intelligent AI-driven Negotiation Coach for the Automotive Industry. "
        f"You are assisting a customer named {customer_context['name']}."
        f"{' The customer prefers ' + customer_context['preferences'] if customer_context.get('preferences') else ''}."
        f"{' They have previously purchased ' + ', '.join(customer_context['purchase_history']) if customer_context.get('purchase_history') else ''}."
        f"{' During the last interaction, the customer showed interest in ' + customer_context['last_interaction'] if customer_context.get('last_interaction') else ''}."
        " Analyze the language, sentiment, and tone to provide tailored recommendations and strategies to negotiate prices. "
        "Respond professionally to maximize customer satisfaction while protecting profitability. "
        "You're in India, so provide prices in INR format without using the ₹ symbol - use 'INR' instead."
    )

    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": negotiation_prompt},
            {"role": "user", "content": user_input}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get("choices")[0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error in API call: {str(e)}"

# Sentiment analysis
def analyze_sentiment(user_input):
    analysis = TextBlob(user_input)
    polarity = analysis.sentiment.polarity
    sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
    return sentiment, polarity

# Logging to CSV
def log_to_csv(file_name, data):
    column_names = ['Customer Name', 'User Input', 'Sentiment', 'Sentiment Score', 'Response', 'Timestamp']
    formatted_data = [str(item).encode('utf-8').decode('utf-8') if item is not None else '' for item in data]

    try:
        file_exists = os.path.isfile(file_name)
        with open(file_name, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(column_names)
            writer.writerow(formatted_data)
    except Exception as e:
        print(f"Error logging to CSV: {str(e)}")

# Speech recognition
def recognize_speech():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Listening...")
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Could not request results"
    except Exception as e:
        print(f"Speech error: {str(e)}")
        return "Speech recognition error"

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_listening', methods=['POST'])
def start_listening():
    text = recognize_speech()
    return jsonify({"text": text})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if request.is_json:
            data = request.get_json()
            user_input = data.get('user_input', '')
            customer_name = data.get('customer_name', '')
        else:
            user_input = request.form.get('user_input', '')
            customer_name = request.form.get('customer_name', '')

        default_context = {
            "name": customer_name,
            "preferences": "Not specified",
            "purchase_history": [],
            "last_interaction": "None"
        }

        customer_context = CRM_DATA.get(customer_name.lower(), default_context)

        sentiment, sentiment_score = analyze_sentiment(user_input)
        response = query_llama_llm(user_input, customer_context)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_to_csv("sales_data2.csv", [
            customer_context['name'],
            user_input,
            sentiment,
            sentiment_score,
            response,
            timestamp
        ])

        if request.is_json:
            return jsonify({
                "sentiment": {"sentiment": sentiment, "polarity": sentiment_score},
                "llama_response": response
            })
        else:
            return render_template('result.html',
                                   user_input=user_input,
                                   sentiment=sentiment,
                                   sentiment_score=sentiment_score,
                                   response=response)

    except Exception as e:
        print(f"Analyze error: {str(e)}")
        return jsonify({"error": "An error occurred"}), 500

@app.route('/sentiment_data', methods=['GET'])
def sentiment_data():
    sentiments = []
    try:
        with open("sales_data2.csv", mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    sentiments.append({
                        "name": row["Customer Name"],
                        "sentiment": row["Sentiment"],
                        "score": float(row["Sentiment Score"])
                    })
                except Exception:
                    continue
    except FileNotFoundError:
        return jsonify({"error": "No data found"}), 404

    return jsonify(sentiments)

@app.route('/sentiment_graph')
def sentiment_graph():
    return render_template('sentiment_graph.html')

if __name__ == '__main__':
    app.run(debug=True)
