import os

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from groq import Groq, AuthenticationError, RateLimitError, APIError

# Load settings from our local .env file.
load_dotenv()

app = Flask(__name__)

# Read the key without putting it directly in our code.
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Check your .env file."
    )

client = Groq(
    api_key=api_key,
    timeout=30.0,
    max_retries=0
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Please send a JSON object."}), 400

    message = data.get("message")

    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "Please type a message."}), 400

    message = message.strip()

    if len(message) > 1000:
        return jsonify({
            "error": "Please use 1,000 characters or fewer."
        }), 400

    history = data.get("history", [])

    if not isinstance(history, list) or len(history) > 10:
        return jsonify({"error": "Invalid conversation history."}), 400

    clean_history = []

    for index, item in enumerate(history):
        if not isinstance(item, dict):
            return jsonify({"error": "Invalid history item."}), 400

        # History must alternate: user, assistant, user, assistant.
        expected_role = "user" if index % 2 == 0 else "assistant"
        content = item.get("content")

        if (
            item.get("role") != expected_role
            or not isinstance(content, str)
            or not content.strip()
            or len(content) > 12000
        ):
            return jsonify({"error": "Invalid history message."}), 400

        clean_history.append({
            "role": expected_role,
            "content": content
        })

    if len(clean_history) % 2 != 0:
        return jsonify({"error": "Incomplete conversation history."}), 400
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
                    messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Ilm Dost, a friendly learning assistant. "
                        "Explain things clearly for school students. "
                        "Use the language the student asks for. "
                        "Keep answers short unless more detail is requested. "
                        "Admit uncertainty and do not invent sources."
                    )
                }
            ] + clean_history + [
                {
                    "role": "user",
                    "content": message
                }
            ],
            max_completion_tokens=1024
        )

        reply = response.choices[0].message.content

        if not reply or not reply.strip():
            return jsonify({
                "error": "The AI returned no answer. Please try again."
            }), 502

        return jsonify({"reply": reply})

    except AuthenticationError:
        return jsonify({
            "error": "The API key was rejected. Check your .env file."
        }), 503

    except RateLimitError:
        return jsonify({
            "error": "The AI usage limit was reached. Please try later."
        }), 429

    except APIError:
        return jsonify({
            "error": "Could not reach the AI service. Please try again."
        }), 502


if __name__ == "__main__":
    app.run(debug=True)