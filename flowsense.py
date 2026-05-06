import os
import re
import ssl
import json
import uuid
import base64
import asyncio
import subprocess
import requests

import numpy as np
import sounddevice as sd
import websockets
from flask import Flask, request, jsonify, send_from_directory
from websockets.exceptions import ConnectionClosedOK
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

GRADIUM_API_KEY = os.getenv("GRADIUM_API_KEY")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")

GRADIUM_ASR_URL = "wss://api.gradium.ai/api/speech/asr"
MINIMAX_LLM_URL = "https://api.minimax.io/v1/chat/completions"
MINIMAX_LLM_MODEL = "MiniMax-M2.7"

SAMPLE_RATE = 24000
DURATION = 3
CHUNK_SAMPLES = 1920
CHUNK_BYTES = CHUNK_SAMPLES * 2

AUDIO_DIR = "generated_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

THINKING_KEYWORDS = ["think", "mean", "remember", "hmm", "um", "uh"]
ANGER_KEYWORDS = ["unacceptable", "frustrating", "annoying", "stop"]


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/audio/<filename>")
def audio(filename):
    return send_from_directory(AUDIO_DIR, filename)


# --------------------
# Gradium STT
# --------------------

def record_pcm():
    print("Recording...", flush=True)

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    sd.wait()

    print("Recording done.", flush=True)

    audio = audio.reshape(-1)
    print("Audio max amplitude:", int(np.max(np.abs(audio))), flush=True)

    return audio.tobytes()


def insecure_ssl_context():
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    return ssl_context


async def transcribe_pcm(pcm_bytes):
    if not GRADIUM_API_KEY:
        raise RuntimeError("GRADIUM_API_KEY is missing")

    headers = {"x-api-key": GRADIUM_API_KEY}
    texts = []

    async with websockets.connect(
        GRADIUM_ASR_URL,
        additional_headers=headers,
        ssl=insecure_ssl_context(),
        max_size=20 * 1024 * 1024,
    ) as ws:
        await ws.send(json.dumps({
            "type": "setup",
            "model_name": "default",
            "input_format": "pcm",
        }))

        ready = json.loads(await ws.recv())
        print("Gradium ready:", ready, flush=True)

        if ready.get("type") == "error":
            raise RuntimeError(ready)

        for i in range(0, len(pcm_bytes), CHUNK_BYTES):
            chunk = pcm_bytes[i:i + CHUNK_BYTES]
            if not chunk:
                continue

            encoded = base64.b64encode(chunk).decode("utf-8")

            await ws.send(json.dumps({
                "type": "audio",
                "audio": encoded,
            }))

            await asyncio.sleep(0.08)

        await ws.send(json.dumps({
            "type": "flush",
            "flush_id": 1,
        }))

        await ws.send(json.dumps({
            "type": "end_of_stream",
        }))

        try:
            async for raw in ws:
                msg = json.loads(raw)
                print("Gradium:", msg, flush=True)

                msg_type = msg.get("type")

                if msg_type == "text":
                    texts.append(msg.get("text", ""))

                elif msg_type == "error":
                    raise RuntimeError(msg)

                elif msg_type == "end_of_stream":
                    break

        except ConnectionClosedOK:
            print("Gradium connection closed normally.", flush=True)

    return " ".join(texts).strip()


# --------------------
# FlowSense rules
# --------------------

def detect_scene(user_text: str):
    text = user_text.lower()

    matched_thinking = [k for k in THINKING_KEYWORDS if k in text]
    matched_anger = [k for k in ANGER_KEYWORDS if k in text]

    if matched_anger:
        return {
            "detected_scene": "anger",
            "matched_keywords": matched_anger,
            "rule": "anger_keywords",
        }

    if matched_thinking:
        return {
            "detected_scene": "thinking",
            "matched_keywords": matched_thinking,
            "rule": "thinking_keywords",
        }

    return {
        "detected_scene": "normal",
        "matched_keywords": [],
        "rule": "no_keyword_match",
    }


def next_turn_control(turn_stage: str):
    if turn_stage in ["first_pause", "user_continue"]:
        return {
            "next_action": "resume_listening",
            "next_turn_stage": "user_continue",
        }

    return {
        "next_action": "idle",
        "next_turn_stage": "first_pause",
    }


def flowsense_policy(user_text: str, turn_stage: str = "first_pause"):
    detection = detect_scene(user_text)

    scene = detection["detected_scene"]
    matched_keywords = detection["matched_keywords"]
    rule = detection["rule"]

    control = next_turn_control(turn_stage)

    base = {
        "detected_scene": scene,
        "matched_keywords": matched_keywords,
        "rule": rule,
        **control,
    }

    if scene == "thinking":
        if turn_stage == "first_pause":
            return {
                **base,
                "scene": "thinking_silence",
                "state": "THINKING_WAIT",
                "decision": "Give low-pressure support",
                "required_agent_intent": "Say only: Take your time.",
                "max_words": 8,
            }

        if turn_stage == "user_continue":
            return {
                **base,
                "scene": "thinking_continue",
                "state": "ACTIVE_LISTENING",
                "decision": "Acknowledge briefly",
                "required_agent_intent": "Say a short active-listening phrase.",
                "max_words": 10,
            }

        return {
            **base,
            "scene": "thinking_solve",
            "state": "PROBLEM_SOLVING",
            "decision": "Help solve the problem",
            "required_agent_intent": "Summarize briefly and offer a first concrete step.",
            "max_words": 28,
        }

    if scene == "anger":
        if turn_stage == "first_pause":
            return {
                **base,
                "scene": "anger_silence",
                "state": "DISSATISFACTION_WAIT",
                "decision": "Align emotionally before solving",
                "required_agent_intent": "Say only: Yeah, totally.",
                "max_words": 8,
            }

        if turn_stage == "user_continue":
            return {
                **base,
                "scene": "anger_continue",
                "state": "EMOTIONAL_ALIGNMENT",
                "decision": "Validate the user's frustration",
                "required_agent_intent": "Say only: I understand.",
                "max_words": 8,
            }

        return {
            **base,
            "scene": "anger_solve",
            "state": "PROBLEM_SOLVING",
            "decision": "Help solve after emotional alignment",
            "required_agent_intent": "Validate once, then offer a concrete next step.",
            "max_words": 30,
        }

    return {
        **base,
        "scene": "normal",
        "state": "READY_TO_RESPOND",
        "decision": "Respond normally",
        "required_agent_intent": "Give a brief helpful response.",
        "max_words": 25,
    }


# --------------------
# MiniMax LLM brain
# --------------------

def clean_minimax_text(text: str):
    if not text:
        return ""

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def fallback_agent_text(policy: dict):
    fallback_map = {
        "thinking_silence": "Take your time.",
        "thinking_continue": "Mm-hmm, I’m listening.",
        "thinking_solve": "Got it. Let’s start with what feels unclear.",
        "anger_silence": "Yeah, totally.",
        "anger_continue": "I understand.",
        "anger_solve": "I understand. Let’s see what I can do right now.",
        "normal": "I understand. Please go ahead.",
    }

    return fallback_map.get(
        policy.get("scene", ""),
        "I understand. Please go ahead."
    )


def minimax_llm_brain(user_text: str, policy: dict):
    if not MINIMAX_API_KEY:
        raise RuntimeError("MINIMAX_API_KEY is missing")

    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }

    system_prompt = f"""
You are the LLM brain of a real-time voice agent.

You MUST follow the FlowSense interaction policy exactly.

FlowSense keyword detection:
detected scene: {policy["detected_scene"]}
matched keywords: {policy["matched_keywords"]}
rule used: {policy["rule"]}

FlowSense interaction policy:
scene: {policy["scene"]}
state: {policy["state"]}
decision: {policy["decision"]}
required agent intent: {policy["required_agent_intent"]}
maximum words: {policy["max_words"]}

Conversation control:
next_action: {policy["next_action"]}
next_turn_stage: {policy["next_turn_stage"]}

Rules:
- Generate ONLY the agent's spoken response.
- Do NOT explain your reasoning.
- Do NOT include <think> tags.
- Do NOT mention FlowSense.
- Keep it natural for speech.
- Keep it within {policy["max_words"]} words.
- If the required intent says "Say only", say only that phrase or a very close equivalent.
- If this is a micro-response, do not solve the problem yet.
"""

    payload = {
        "model": MINIMAX_LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "max_tokens": 100,
        "temperature": 0.4,
    }

    response = requests.post(
        MINIMAX_LLM_URL,
        headers=headers,
        json=payload,
        timeout=30,
    )

    print("MiniMax LLM status:", response.status_code, flush=True)
    print("MiniMax LLM raw:", response.text, flush=True)

    response.raise_for_status()

    data = response.json()
    raw_text = data["choices"][0]["message"].get("content", "")
    agent_text = clean_minimax_text(raw_text)

    if not agent_text:
        agent_text = fallback_agent_text(policy)

    return agent_text


# --------------------
# MiniMax TTS
# --------------------

def minimax_tts(text: str, output_path: str):
    if not text.strip():
        raise RuntimeError("TTS text is empty")

    subprocess.run(
        [
            "mmx",
            "speech",
            "synthesize",
            "--text",
            text,
            "--out",
            output_path,
        ],
        check=True,
    )


def build_agent_response(user_text: str, turn_stage: str):
    policy = flowsense_policy(user_text, turn_stage)
    agent_text = minimax_llm_brain(user_text, policy)

    if not agent_text:
        agent_text = fallback_agent_text(policy)

    filename = f"{uuid.uuid4().hex}.mp3"
    output_path = os.path.join(AUDIO_DIR, filename)

    print("User text:", user_text, flush=True)
    print("Turn stage:", turn_stage, flush=True)
    print("Policy:", policy, flush=True)
    print("Agent text:", agent_text, flush=True)

    minimax_tts(agent_text, output_path)

    return {
        "user_text": user_text,
        "turn_stage": turn_stage,
        "policy": policy,
        "agent_text": agent_text,
        "audio_url": f"/audio/{filename}",
        "next_action": policy.get("next_action", "idle"),
        "next_turn_stage": policy.get("next_turn_stage", "first_pause"),
    }


# --------------------
# Endpoints
# --------------------

@app.route("/agent_turn", methods=["POST"])
def agent_turn():
    try:
        data = request.json or {}

        user_text = data.get("user_text", "").strip()
        turn_stage = data.get("turn_stage", "first_pause").strip()

        if not user_text:
            return jsonify({"error": "No user_text provided"}), 400

        if turn_stage not in ["first_pause", "user_continue", "solve"]:
            return jsonify({
                "error": "Invalid turn_stage. Use first_pause, user_continue, or solve."
            }), 400

        result = build_agent_response(user_text, turn_stage)
        return jsonify(result)

    except Exception as e:
        print("agent_turn error:", e, flush=True)
        return jsonify({"error": str(e)}), 500


@app.route("/record_turn", methods=["POST"])
def record_turn():
    try:
        print("\n========== /record_turn called ==========", flush=True)

        data = request.json or {}
        turn_stage = data.get("turn_stage", "first_pause").strip()

        print("Turn stage:", turn_stage, flush=True)

        if turn_stage not in ["first_pause", "user_continue", "solve"]:
            return jsonify({
                "error": "Invalid turn_stage. Use first_pause, user_continue, or solve."
            }), 400

        print("Step 1: recording audio...", flush=True)
        pcm = record_pcm()
        print("Step 1 done: audio recorded", flush=True)

        print("Step 2: sending audio to Gradium STT...", flush=True)
        user_text = asyncio.run(transcribe_pcm(pcm))
        print("Step 2 done: transcript =", user_text, flush=True)

        if not user_text:
            return jsonify({"error": "No speech detected"}), 400

        print("Step 3: building FlowSense + MiniMax response...", flush=True)
        result = build_agent_response(user_text, turn_stage)
        print("Step 3 done: response ready", flush=True)

        print("========== /record_turn finished ==========\n", flush=True)

        return jsonify(result)

    except Exception as e:
        print("record_turn error:", e, flush=True)
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "gradium_key_loaded": bool(GRADIUM_API_KEY),
        "minimax_key_loaded": bool(MINIMAX_API_KEY),
        "audio_dir": AUDIO_DIR,
        "thinking_keywords": THINKING_KEYWORDS,
        "anger_keywords": ANGER_KEYWORDS,
    })


if __name__ == "__main__":
    print("GRADIUM_API_KEY loaded:", bool(GRADIUM_API_KEY))
    print("MINIMAX_API_KEY loaded:", bool(MINIMAX_API_KEY))
    app.run(debug=True, port=5050)