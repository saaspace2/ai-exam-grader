"""Diagnose what the vision model actually returns for an image.

Usage:
    python diagnose_vision.py path/to/image.jpeg

This calls OpenRouter directly with your real key and prints the RAW response,
so you can see exactly why a read failed (rate limit, model support, etc).
"""
import sys, json, base64, mimetypes
from pathlib import Path
from grader.config import settings

if len(sys.argv) < 2:
    print("Usage: python diagnose_vision.py <image_path>"); sys.exit()
img = Path(sys.argv[1])
if not img.exists():
    print("File not found:", img); sys.exit()

print("Key configured:", settings.has_api_key())
print("Model:", settings.OPENROUTER_MODEL)
print("Reading:", img.name)

import requests
mime = mimetypes.guess_type(img.name)[0] or "image/jpeg"
b64 = base64.b64encode(img.read_bytes()).decode()
data_uri = f"data:{mime};base64,{b64}"

resp = requests.post(
    settings.OPENROUTER_BASE_URL,
    headers={"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
             "Content-Type": "application/json"},
    data=json.dumps({
        "model": settings.OPENROUTER_MODEL,
        "messages": [{"role":"user","content":[
            {"type":"text","text":"What is written on this page?"},
            {"type":"image_url","image_url":{"url":data_uri}},
        ]}],
    }),
    timeout=120,
)
print("\nHTTP status:", resp.status_code)
try:
    body = resp.json()
    print("\nFull response:")
    print(json.dumps(body, indent=2)[:2000])
except Exception as e:
    print("Could not parse response:", e)
    print(resp.text[:1000])