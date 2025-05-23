from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import random
import logging
from datetime import datetime

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Initialize model and tokenizer
MODEL_NAME = "facebook/blenderbot-400M-distill"
try:
    app.logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    MODEL_LOADED = True
    app.logger.info("Model loaded successfully!")
except Exception as e:
    app.logger.error(f"Error loading model: {e}")
    MODEL_LOADED = False

# Enhanced empathetic responses
EMPATHETIC_RESPONSES = [
    "I hear you. Would you like to share more about that?",
    "That sounds important. Tell me more.",
    "I'm listening. How does that make you feel?",
    "You're being very open. I appreciate that.",
    "Let's explore that together. What else comes to mind?"
]

TOPIC_RESPONSES = {
    "anxiety": [
        "Anxiety can feel overwhelming. Let's break this down together.",
        "What you're feeling is valid. Would you like to explore coping strategies?"
    ],
    "depression": [
        "I'm sorry you're feeling this way. You're not alone in this.",
        "Depression can make things feel heavy. What small thing might help today?"
    ],
    "stress": [
        "Stress can really build up. What's one thing that might relieve some pressure?",
        "Let's identify what's within your control right now."
    ]
}

CRISIS_KEYWORDS = ["suicide", "kill myself", "end it all", "want to die"]

def detect_topic(text):
    text = text.lower()
    if any(word in text for word in ["anxious", "anxiety", "nervous", "panic"]):
        return "anxiety"
    elif any(word in text for word in ["depress", "sad", "hopeless", "empty"]):
        return "depression"
    elif any(word in text for word in ["stress", "overwhelm", "pressure", "burnout"]):
        return "stress"
    return None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message", "").strip()
        app.logger.info(f"Received message: {user_input}")
        
        if not user_input:
            return jsonify({
                "response": "I notice you're being quiet. Would you like to share?",
                "suggestions": [
                    "I'm feeling...",
                    "What's on my mind is...",
                    "I need help with..."
                ]
            })
        
        # Crisis detection
        if any(keyword in user_input.lower() for keyword in CRISIS_KEYWORDS):
            return jsonify({
                "response": "I'm deeply concerned. Please contact a crisis hotline immediately. "
                           "In the US: call/text 988. International help: https://www.opencounseling.com/suicide-hotlines",
                "urgent": True
            })
        
        # Check for specific topics
        topic = detect_topic(user_input)
        if topic and random.random() < 0.7:  # 70% chance to use topic-specific response
            return jsonify({
                "response": random.choice(TOPIC_RESPONSES[topic]),
                "suggestions": [
                    f"More about my {topic}",
                    "How this affects my daily life",
                    "Coping strategies"
                ]
            })
        
        # Generate response
        if MODEL_LOADED:
            try:
                inputs = tokenizer([user_input], return_tensors="pt", truncation=True)
                
                reply_ids = model.generate(
                    **inputs,
                    max_length=150,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    num_beams=3
                )
                
                response = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
                response = response.replace("<s>", "").replace("</s>", "").strip()
                
                # Make response more conversational
                if not response.endswith(('?', '!', '.')):
                    response += "."
                
            except Exception as e:
                app.logger.error(f"Generation error: {e}")
                response = random.choice(EMPATHETIC_RESPONSES)
        else:
            response = random.choice(EMPATHETIC_RESPONSES)
        
        # Add follow-up suggestions
        suggestions = [
            "Tell me more about that",
            "How does this make you feel?",
            "What else is on your mind?"
        ]
        
        return jsonify({
            "response": response,
            "suggestions": suggestions
        })
    
    except Exception as e:
        app.logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            "response": "I'm having trouble understanding. Could you rephrase that?",
            "error": "processing_error"
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)