import sys
import ollama
import cv2
import base64
import wave
import os
import uuid
import winsound
import requests
import re
import json
import threading
from datetime import datetime
from piper import PiperVoice
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import time

# Force UTF-8 on the Windows console so Unicode output never crashes the CLI
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# --- Configuration ---
MODEL = "llava:7b"
VOICE_MODEL = "en_US-danny-low.onnx"
VOICE_CONFIG = "en_US-danny-low.onnx.json"
LAT = -8.6751
LON = 115.2217

# --- File-based Memory ---
MEMORY_FILE = "vespa_memory.json"
MEMORY_LIMIT = 20  # How many past messages to inject as context

# Lock that ensures only one TTS call plays at a time,
# preventing the race condition where two threads write to the same wav file.
_tts_lock = threading.Lock()
_memory_lock = threading.Lock()

SYSTEM_INSTRUCTION = (
    "You are VESPA, Agas's witty and intelligent personal assistant. "
    "You are an expert in math, physics, and programming. "
    "Always respond in plain text only — no markdown, no asterisks, no bullet symbols. "
    "If an image has been provided, you MUST describe what you actually see before answering. "
    "Never guess or assume what is in an image — only describe what is visually present."
)

# --- Initialization ---
print("Initializing VESPA...")
client = ollama.Client()
voice = PiperVoice.load(VOICE_MODEL, config_path=VOICE_CONFIG)
print("VESPA is ready.\n")


# ══════════════════════════════════════════════════════
#  FILE-BASED MEMORY
# ══════════════════════════════════════════════════════

def memory_save(role: str, content: str):
    """Append a single message to the local JSON memory file."""
    with _memory_lock:
        try:
            if os.path.exists(MEMORY_FILE):
                with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                    history = json.load(f)
            else:
                history = []

            history.append({"role": role, "content": content})

            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[VESPA: Memory write failed — {e}]")


def memory_load() -> list[dict]:
    """
    Read the last MEMORY_LIMIT messages from the local JSON file,
    returned in oldest-first order.
    """
    with _memory_lock:
        try:
            if not os.path.exists(MEMORY_FILE):
                return []
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
            return history[-MEMORY_LIMIT:]
        except Exception as e:
            print(f"[VESPA: Memory read failed — {e}]")
            return []


def memory_format_context(history: list[dict]) -> str:
    """
    Formats the memory list into a readable context block that the model
    can understand, injected just before the current user message.
    """
    if not history:
        return ""
    lines = ["[Previous conversation memory — use this to maintain context:]"]
    for msg in history:
        prefix = "Agas" if msg["role"] == "user" else "VESPA"
        lines.append(f"  {prefix}: {msg['content']}")
    lines.append("[End of memory]")
    return "\n" + "\n".join(lines)


# ══════════════════════════════════════════════════════
#  FLASK APP
# ══════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app)   # Allow the browser to call localhost:5000 from any origin


@app.route('/check-intent', methods=['POST'])
def check_intent_endpoint():
    """
    The browser calls this first to decide whether to open its camera.
    Returns: { needs_camera: bool, needs_weather: bool }
    """
    data  = request.get_json(force=True)
    query = data.get('query', '')
    return jsonify({
        'needs_camera':  needs_camera(query),
        'needs_weather': needs_weather(query),
    })


@app.route('/ask', methods=['POST'])
def ask_endpoint():
    """
    SSE streaming endpoint.
    Body: { query: str, image: str|null }  (image is base64 JPEG from browser camera)

    Emits Server-Sent Events:
      data: {"type": "log",   "message": "..."}
      data: {"type": "token", "text":    "..."}
      data: {"type": "done"}
      data: {"type": "error", "message": "..."}
    """
    data      = request.get_json(force=True)
    query     = data.get('query', '')
    image_b64 = data.get('image')   # base64 from browser, or null

    def generate():
        try:
            yield sse({'type': 'log', 'message': 'NEURAL_LINK: Building context...'})

            extra_context = ''

            # Memory
            history = memory_load()
            if history:
                extra_context += memory_format_context(history)
                yield sse({'type': 'log', 'message': f'MEM_SUBSYS: {len(history)} memories loaded'})

            # Weather
            if needs_weather(query):
                weather_info = get_weather_info()
                extra_context += f'\n[System weather data: {weather_info}]'
                yield sse({'type': 'log', 'message': 'METEO_SUBSYS: Weather data injected'})

            # Vision (image captured by the browser and sent here)
            if image_b64:
                extra_context += (
                    '\n[System: A live image from the user\'s webcam is attached. '
                    'You MUST look at it carefully and describe exactly what you see '
                    'before answering. Do not guess — base your answer solely on the image.]'
                )
                yield sse({'type': 'log', 'message': 'OPTICS_SUBSYS: Image received from browser'})

            # Build message
            full_prompt = f"{SYSTEM_INSTRUCTION}{extra_context}\n\nUser: {query}"

            if image_b64:
                messages = [{
                    'role': 'user',
                    'content': full_prompt,
                    'images': [image_b64],   # Correct ollama/llava format
                }]
            else:
                messages = [{'role': 'user', 'content': full_prompt}]

            yield sse({'type': 'log', 'message': 'NEURAL_TX: Streaming from model...'})

            # Stream tokens
            full_response = ''
            response = client.chat(model=MODEL, messages=messages, stream=True)

            for chunk in response:
                token = chunk['message']['content']   # chat() key — NOT chunk['response']
                if token:
                    clean = clean_text(token)
                    full_response += clean
                    yield sse({'type': 'token', 'text': clean})

            # TTS on server side after streaming finishes
            if full_response.strip():
                threading.Thread(
                    target=speak_text, args=(full_response,), daemon=True
                ).start()
                # Save this exchange to memory (non-blocking)
                threading.Thread(
                    target=lambda: (memory_save("user", query), memory_save("assistant", full_response)),
                    daemon=True
                ).start()
                yield sse({'type': 'log', 'message': 'MEM_SUBSYS: Exchange saved to memory'})

            yield sse({'type': 'done'})

        except Exception as e:
            yield sse({'type': 'error', 'message': str(e)})

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control':     'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection':        'keep-alive',
        }
    )


def sse(payload: dict) -> str:
    """Formats a dict as a Server-Sent Event string."""
    return f"data: {json.dumps(payload)}\n\n"


# ══════════════════════════════════════════════════════
#  WEATHER
# ══════════════════════════════════════════════════════

def get_weather_info():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":  LAT,
        "longitude": LON,
        "hourly":    ["temperature_2m", "weather_code", "relative_humidity_2m"],
        "timezone":  "auto",
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        target_str = (
            datetime.now()
            .replace(minute=0, second=0, microsecond=0)
            .strftime("%Y-%m-%dT%H:00")
        )
        idx  = data["hourly"]["time"].index(target_str)
        temp = data["hourly"]["temperature_2m"][idx]
        hum  = data["hourly"]["relative_humidity_2m"][idx]
        code = data["hourly"]["weather_code"][idx]
        condition_map = {
            0: "clear skies", 1: "mainly clear skies", 2: "partly cloudy skies",
            3: "overcast skies", 45: "foggy conditions", 61: "light rain",
            63: "moderate rain", 65: "heavy rain", 80: "rain showers", 95: "thunderstorms",
        }
        condition = condition_map.get(code, "fair weather")
        return (
            f"Currently at your location there are {condition}, "
            f"with a temperature of {temp} degrees Celsius "
            f"and {hum} percent humidity."
        )
    except Exception as e:
        print(f"[VESPA: Weather fetch failed — {e}]")
        return "I am currently unable to retrieve the weather data."


# ══════════════════════════════════════════════════════
#  TEXT UTILITIES
# ══════════════════════════════════════════════════════

def clean_text(text):
    return re.sub(r'[*#_`~]', '', text)


def speak_text(text: str):
    """
    Synthesise speech and play it.
    Uses a per-call unique filename + a global lock so concurrent calls
    (web thread and CLI thread) never overwrite each other's wav file.
    """
    text = clean_text(text).strip()
    if not text:
        return
    # Unique file so two threads never collide on disk
    tmp = f"vespa_tts_{uuid.uuid4().hex}.wav"
    with _tts_lock:
        try:
            with wave.open(tmp, "wb") as wav_file:
                voice.synthesize_wav(text, wav_file)
            winsound.PlaySound(tmp, winsound.SND_FILENAME)
        except Exception as e:
            print(f"[VESPA: TTS error — {e}]")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


# ══════════════════════════════════════════════════════
#  INTENT DETECTION
# ══════════════════════════════════════════════════════

def needs_camera(user_input: str) -> bool:
    """
    Two-stage camera intent detection.

    Stage 1 — Rule-based pre-filter (fast, no AI call):
      • If the query matches a definite-NO pattern → return False immediately.
      • If the query matches a definite-YES pattern → return True immediately.

    Stage 2 — AI arbiter (only for genuinely ambiguous queries):
      • A tightly constrained prompt asks the AI to decide.
      • The AI is only reached when Stage 1 produces no match.
    """
    lowered = user_input.lower().strip()

    # ── Stage 1a: Definite NO patterns ─────────────────────────────────────
    _no_starters = (
        "what is ", "what are ", "what was ", "what were ", "what does ",
        "what do ", "what did ", "what will ", "what would ", "what should ",
        "how do ", "how does ", "how did ", "how can ", "how to ",
        "how is ", "how are ", "how was ", "how were ",
        "why is ", "why are ", "why do ", "why does ", "why did ", "why was ",
        "who is ", "who are ", "who was ", "who were ", "who did ",
        "when is ", "when are ", "when was ", "when did ", "when does ",
        "where is ", "where are ", "where was ", "where did ",
        "which is ", "which are ", "which was ", "which were ",
        "can you explain", "can you tell me", "can you help",
        "can you write", "can you code", "can you create",
        "can you calculate", "can you solve", "can you give me",
        "explain ", "define ", "describe the concept", "tell me about",
        "write a ", "write me ", "write an ",
        "code a ", "code me ", "create a ", "generate a ",
        "calculate ", "solve ", "compute ",
        "i put ", "i left ", "i placed ", "i stored ", "i kept ",
        "remember that ", "remember my ", "note that ", "keep in mind",
        "my name is", "i am ", "i'm ",
    )
    if any(lowered.startswith(s) for s in _no_starters):
        return False

    _no_phrases = (
        "remember", "don't forget", "make a note", "keep track",
        "what is a ", "what is an ", "what is the ",
        "what are the ", "what are some ",
        "how do i ", "how do you ", "how does a ", "how does the ",
        "write me a poem", "tell me a joke", "give me a recipe",
    )
    if any(p in lowered for p in _no_phrases):
        return False

    # ── Stage 1b: Definite YES patterns ────────────────────────────────────
    _yes_phrases = (
        "look at this", "look at that", "what is this",
        "what am i holding", "what am i wearing", "what's on my",
        "what is on my", "read this", "read what", "scan this",
        "what do you see", "can you see", "tell me what you see",
        "what's in front of me", "what is in front of me",
        "how many fingers", "is this broken", "does this look",
        "what color is my", "what colour is my",
        "identify this", "recognise this", "recognize this",
    )
    if any(p in lowered for p in _yes_phrases):
        return True

    # ── Stage 2: AI arbiter for genuinely ambiguous queries ─────────────────
    prompt = (
        f"Query: \"{user_input}\"\n\n"
        "Does answering this query require seeing a live image from a webcam?\n"
        "The camera is ONLY needed if the user is explicitly asking you to look "
        "at a physical object or scene that is right in front of them right now.\n"
        "The camera is NOT needed for any knowledge question, math, code, "
        "definitions, explanations, memory requests, or statements.\n"
        "Reply with one word only: YES or NO."
    )
    try:
        result = client.generate(model=MODEL, prompt=prompt, stream=False)
        answer = result['response'].strip().upper()
        return answer == "YES"
    except Exception:
        return False


def needs_weather(user_input: str) -> bool:
    """
    Uses the AI to decide whether real-time weather data is needed.
    """
    prompt = f"""You are a decision-making module for an AI assistant called VESPA.
Your ONLY job right now is to decide whether answering the query below requires
fetching the current real-time weather conditions at the user's location.

=== QUERY ===
{user_input}
=============

=== YOUR DECISION RULES ===

Answer YES if the query:
- Explicitly asks about the current weather, temperature, humidity, rain, wind,
  or forecast. Examples: "What's the weather like?", "Is it going to rain?",
  "How hot is it outside?", "Should I bring an umbrella today?", "What's the
  temperature right now?"
- Asks whether conditions are suitable for an outdoor activity today or right now.

Answer NO if the query:
- Mentions weather only in a hypothetical, historical, or general science context.
- Is about anything other than the current real-world weather conditions.
- Does not mention weather, temperature, rain, or outdoor conditions at all.

=== CRITICAL ANTI-HALLUCINATION RULES ===
- Do NOT say YES just because the query mentions the words "hot", "cold", or
  "temperature" in a non-weather context.
- Default to NO when in doubt.
- Do NOT answer the actual question. Your entire output must be one word only.

=== YOUR RESPONSE ===
Reply with ONLY the single word YES or NO. No explanation, no punctuation,
no other text whatsoever. A single word."""

    try:
        result = client.generate(model=MODEL, prompt=prompt, stream=False)
        answer = result['response'].strip().upper()
        return answer.startswith("YES") and len(answer) < 6
    except Exception:
        return False


# ══════════════════════════════════════════════════════
#  CLI (still works in parallel with the web server)
# ══════════════════════════════════════════════════════

def capture_vision_cv2():
    """OpenCV camera capture — used only by the CLI path."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "ERROR: Camera could not be opened."
    time.sleep(1.5)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return "ERROR: Could not read frame."
    cv2.imshow("VESPA's Eyes", frame)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')


def cli_loop():
    """Original command-line interface — runs in a background thread."""
    print("CLI also available. Type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("You (CLI): ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                speak_text("Goodbye, Agas.")
                print("Goodbye.")
                os._exit(0)

            extra_context = ''
            image_b64 = None

            if needs_camera(user_input):
                print("[VESPA: Camera required. Opening eyes...]")
                result = capture_vision_cv2()
                if "ERROR" in result:
                    extra_context += "\n[System: Camera error. Tell the user you cannot see.]"
                else:
                    image_b64 = result
                    extra_context += "\n[System: Image attached. Describe exactly what you see.]"

            if needs_weather(user_input):
                extra_context += f"\n[System weather data: {get_weather_info()}]"

            history = memory_load()
            if history:
                extra_context += memory_format_context(history)

            full_prompt = f"{SYSTEM_INSTRUCTION}{extra_context}\n\nUser: {user_input}"
            messages = (
                [{'role': 'user', 'content': full_prompt, 'images': [image_b64]}]
                if image_b64 else
                [{'role': 'user', 'content': full_prompt}]
            )

            print("\nAssistant: ", end='', flush=True)
            response = client.chat(model=MODEL, messages=messages, stream=True)
            sentence_buffer = ''
            full_response_cli = ''
            for chunk in response:
                token = chunk['message']['content']
                clean = clean_text(token)
                print(clean, end='', flush=True)
                sentence_buffer += clean
                full_response_cli += clean
                if any(p in token for p in (".", "!", "?", "\n")):
                    if sentence_buffer.strip():
                        speak_text(sentence_buffer)
                    sentence_buffer = ''
            if sentence_buffer.strip():
                speak_text(sentence_buffer)
            memory_save("user", user_input)
            memory_save("assistant", full_response_cli)
            print("\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[Glitch]: {e}\n")


# ══════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Starting VESPA web server on http://localhost:5000 ...")

    # CLI runs in a background daemon thread so Flask stays alive on the main thread
    cli_thread = threading.Thread(target=cli_loop, daemon=True)
    cli_thread.start()

    # Flask blocks the main thread
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)