import json
import re
from typing import Dict, Optional

import google.generativeai as genai

# ======================================
# CONFIG
# ======================================
genai.configure(api_key="gemini_api_key")

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="""
You ALWAYS return a single JSON object with the following format:

{
  "action": "...",
  "target": "...",
  "message": "...",
  "value": number,
  "step": number,
  "query": "..."
}

Rules:
- No text before or after JSON.
- No backticks.
- No comments.
- If unsure, return: {"action": "none"}.

Supported actions:
- send_telegram
- send_telegram_group
- read_telegram
- reply_telegram
- brightness_set
- brightness_increase
- brightness_decrease
- open_youtube
- system
- none
"""
)

# ======================================
# LLM PARSER – bulletproof
# ======================================
def call_llm_parse(text: str) -> Optional[Dict]:
    try:
        response = model.generate_content(text)

        if not hasattr(response, "text"):
            return None

        raw = response.text.strip()

        # Remove markdown like ```json etc.
        raw = re.sub(r"```.*?```", "", raw, flags=re.DOTALL).strip()

        # Extract first JSON block safely
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return None

        json_block = m.group(0)

        try:
            return json.loads(json_block)

        except json.JSONDecodeError:
            # fix trailing commas
            cleaned = re.sub(r",\s*}", "}", json_block)
            cleaned = re.sub(r",\s*]", "]", cleaned)
            return json.loads(cleaned)

    except Exception as e:
        print("LLM parse error:", e)
        return None


# ======================================
# RULE-BASED PARSER – improved
# ======================================
def rule_based_parse(text: str) -> Dict:
    t = text.lower().strip()

    # ----------------------------------------
    # Brightness
    # ----------------------------------------
    m = re.search(r"brightness.*?(\d{1,3})", t)
    if m:
        v = max(0, min(100, int(m.group(1))))
        return {"action": "brightness_set", "value": v}

    if "increase brightness" in t or "brightness up" in t:
        return {"action": "brightness_increase", "step": 10}

    if "decrease brightness" in t or "brightness down" in t:
        return {"action": "brightness_decrease", "step": 10}

    # ----------------------------------------
    # Read message
    # ----------------------------------------
    m = re.search(r"(?:read|check).*(?:message|messages).*from\s+([a-zA-Z0-9\s]+)", t)
    if m:
        return {"action": "read_telegram", "target": m.group(1).strip()}

    # ----------------------------------------
    # Send message
    # ----------------------------------------
    # send message to Ashu hello bro
    m = re.search(r"(?:send|message|text).*to\s+([a-zA-Z0-9\s]+)\s+(.*)", t)
    if m:
        return {
            "action": "send_telegram",
            "target": m.group(1).strip(),
            "message": m.group(2).strip()
        }

    # send to Ashu hi
    m = re.search(r"send\s+to\s+([a-zA-Z0-9\s]+)\s+(.*)", t)
    if m:
        return {
            "action": "send_telegram",
            "target": m.group(1).strip(),
            "message": m.group(2).strip()
        }

    # ----------------------------------------
    # Reply
    # ----------------------------------------
    # reply to Ashu hi
    m = re.search(r"reply\s+to\s+([a-zA-Z0-9\s]+)\s+(.*)", t)
    if m:
        return {
            "action": "reply_telegram",
            "target": m.group(1).strip(),
            "message": m.group(2).strip()
        }

    # reply hi (auto last contact)
    m = re.search(r"reply\s+(.*)", t)
    if m:
        return {
            "action": "reply_telegram",
            "target": None,
            "message": m.group(1).strip()
        }

    # ----------------------------------------
    # YouTube
    # ----------------------------------------
    if "youtube" in t or "play" in t:
        q = t.replace("play", "").replace("on youtube", "").strip()
        return {"action": "open_youtube", "query": q}

    # ----------------------------------------
    # Default
    # ----------------------------------------
    return {"action": "none"}


# ======================================
# PUBLIC PARSE FUNCTION
# ======================================
def parse(text: str) -> Dict:
    if not text or not text.strip():
        return {"action": "none"}

    llm = call_llm_parse(text)
    if isinstance(llm, dict) and "action" in llm:
        return llm

    return rule_based_parse(text)
