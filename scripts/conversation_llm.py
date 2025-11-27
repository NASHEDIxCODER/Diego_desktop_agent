# import os
# import re
# import json
# from typing import Optional, Dict
# import openai
import google.generativeai as genai
from numba.typed.listobject import ListModel

genai.configure(api_key= "gemini_api_key")
ListModel


SYSTEM_PROMPT = """
You are leo — a friendly AI assistant. 
Give short answers (1–2 sentences).
Speak casually like a helpful friend.
Do NOT mention that you are an AI.
Do NOT use emoji or sticker.
Do NOT respond in hindi language.
You are created by yeshu.
When the user asks to sing, sing the lyrics "mera yeshuu yeshuu mera yeshuu yeshuu" 2 times.



"""

model = genai.GenerativeModel(

    model_name="gemini-2.0-flash",
    system_instruction=SYSTEM_PROMPT
)

# --------------------------
# CHAT FUNCTION
# --------------------------
def chat(text: str) -> str:
    try:
        response = model.generate_content(
            {
                "role": "user",
                "parts": [
                    {"text": text}
                ]
            }
        )

        if hasattr(response, "text") and response.text:
            return response.text.strip()

        return "I didn't get that."

    except Exception as e:
        print("Chat error:", e)
        return "I'm having trouble thinking right now."


# --------------------------
# TEST RUN
# --------------------------
if __name__ == "__main__":
    while True:
        msg = input("You: ")
        reply = chat(msg)
        print("Leo:", reply)