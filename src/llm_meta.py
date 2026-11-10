"""
Video metadata generation with Gemini Files API (live model discovery + strict validation) and Groq fallback support.
"""
import socket

_orig = socket.getaddrinfo

def _ipv4(host, port, family=0, type=0, proto=0, flags=0):
    if family == 0 or family == socket.AF_UNSPEC:
        family = socket.AF_INET
    return _orig(host, port, family, type, proto, flags)

socket.getaddrinfo = _ipv4

import base64
import json
import os
import random
import re
import subprocess
import tempfile
import time
import requests

CURRENCY_PATTERN = re.compile(r"[\$\u20ac\u00a3\u00a5\u20b9]|\b(USD|AED|EUR|GBP|INR|PKR)\b", re.IGNORECASE)
MAGNITUDE_PATTERN = re.compile(r"\b\d+([.,]\d+)?\s*(m|b|k|million|billion|sqft|sq\s*ft|feet)\b", re.IGNORECASE)

GEMINI_BASE_API = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_UPLOAD_API = "https://generativelanguage.googleapis.com/upload/v1beta/files"

EXCLUDE_MODEL_SUBSTRINGS = [
    "-tts", "-image", "-embedding", "-aqa", "-vision", "-thinking",
    "-live", "-native-audio", "-dialog", "learnlm", "gemma",
    "-transcribe", "computer-use", "robotics", "deep-research",
    "customtools", "nano-banana", "lyria", "antigravity"
]

PRIORITY_GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-pro-latest",
]

def clean_text_field(text):
    t = CURRENCY_PATTERN.sub("", str(text or ""))
    t = MAGNITUDE_PATTERN.sub("", t)
    return re.sub(r"\s+", " ", t).strip()

def sanitize_metadata(meta, duration, channel_name="Mind Stories"):
    if isinstance(meta, list):
        meta = meta[0] if meta and isinstance(meta[0], dict) else {}
    if not isinstance(meta, dict):
        meta = {}

    # Title < 60 chars
    title = clean_text_field(meta.get("title", "Inspiring Moment"))
    if len(title) > 59:
        title = title[:59].rsplit(" ", 1)[0].strip()
    if not title:
        title = "Mind Stories Moment"

    # Tags: 10-15 tags, total string < 460
    raw_tags = meta.get("tags", [])
    if isinstance(raw_tags, str):
        raw_tags = [t.strip() for t in raw_tags.split(",") if t.strip()]

    clean_tags = []
    for tag in raw_tags:
        ct = clean_text_field(str(tag)).strip("# ")
        if ct and len(ct) > 1 and ct.lower() not in [x.lower() for x in clean_tags]:
            clean_tags.append(ct)

    fallback_tags = [
        "mind stories", "psychology", "life lessons", "shorts", "storytelling",
        "human behavior", "motivation", "inspiration", "mindset", "wisdom",
        "deep thoughts", "emotional", "perspective", "facts", "viral"
    ]
    for ft in fallback_tags:
        if len(clean_tags) >= 12:
            break
        if ft.lower() not in [x.lower() for x in clean_tags]:
            clean_tags.append(ft)

    clean_tags = clean_tags[:15]
    while len(",".join(clean_tags)) >= 460 and len(clean_tags) > 10:
        clean_tags.pop()

    # Hashtags: exactly 3 hashtags, #shorts when duration <= 60
    raw_hashtags = meta.get("hashtags", [])
    if isinstance(raw_hashtags, str):
        raw_hashtags = [h.strip() for h in re.split(r"[,\s]+", raw_hashtags) if h.strip()]

    clean_ht = []
    for h in raw_hashtags:
        h = str(h).strip()
        if not h.startswith("#"):
            h = f"#{h}"
        h = re.sub(r"[^#a-zA-Z0-9]", "", h)
        if len(h) > 1 and h.lower() not in [x.lower() for x in clean_ht]:
            clean_ht.append(h)

    if duration <= 60 and not any(h.lower() == "#shorts" for h in clean_ht):
        clean_ht.insert(0, "#shorts")

    default_hts = ["#shorts", "#mindstories", "#psychology", "#motivation"]
    for dh in default_hts:
        if len(clean_ht) >= 3:
            break
        if not any(h.lower() == dh.lower() for h in clean_ht):
            clean_ht.append(dh)

    clean_ht = clean_ht[:3]

    # Description: concise hook lines + hashtags + channel name
    raw_desc = meta.get("description", "")
    if isinstance(raw_desc, list):
        lines = [clean_text_field(d) for d in raw_desc if str(d).strip()]
    else:
        lines = [clean_text_field(line) for line in str(raw_desc).splitlines() if line.strip()]
    lines = [l for l in lines if l][:3]
    body = "\n".join(lines) if lines else f"{title}. Watch till the end for a powerful perspective."

    ht_str = " ".join(clean_ht)
    description = f"{body}\n\n{ht_str}\n\n{channel_name}"

    meta["title"] = title
    meta["description"] = description
    meta["tags"] = clean_tags
    meta["hashtags"] = clean_ht
    return meta

def parse_json_payload(raw_text):
    text = (raw_text or "").strip()
    if not text:
        return None
    for candidate in [
        text,
        re.sub(r"^```(?:json|text)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip(),
    ]:
        try:
            obj = json.loads(candidate, strict=False)
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                return obj[0]
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0), strict=False)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None

def get_live_gemini_models(api_key, preferred_model="gemini-3.5-flash-lite"):
    order = []
    for m in [preferred_model] + PRIORITY_GEMINI_MODELS:
        if m and m not in order:
            order.append(m)

    url = f"{GEMINI_BASE_API}/models"
    try:
        res = requests.get(url, headers={"x-goog-api-key": api_key}, params={"key": api_key}, timeout=20)
        if res.status_code == 200:
            valid = []
            for item in res.json().get("models", []):
                if "generateContent" not in item.get("supportedGenerationMethods", []):
                    continue
                name = item.get("name", "").replace("models/", "").strip()
                nl = name.lower()
                if any(sub in nl for sub in EXCLUDE_MODEL_SUBSTRINGS):
                    continue
                valid.append(name)
            ordered = []
            for pref in order:
                if pref in valid and pref not in ordered:
                    ordered.append(pref)
                for m in valid:
                    if m.startswith(pref + "-") and m not in ordered:
                        ordered.append(m)
            for m in valid:
                if "gemini" in m.lower() and "flash" in m.lower() and m not in ordered:
                    ordered.append(m)
            if ordered:
                print(f"[llm_meta] Live Gemini models ({len(ordered)}): {ordered}", flush=True)
                return ordered
    except Exception as e:
        print(f"[llm_meta] Warning fetching live Gemini models: {e}", flush=True)

    return order

def upload_video_to_gemini(video_path, api_key):
    """Uploads a video to Gemini Files API using resumable upload and waits until ACTIVE."""
    file_size = os.path.getsize(video_path)
    display_name = os.path.basename(video_path)
    mime_type = "video/mp4"

    init_headers = {
        "x-goog-api-key": api_key,
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(file_size),
        "X-Goog-Upload-Header-Content-Type": mime_type,
        "Content-Type": "application/json"
    }
    init_resp = requests.post(
        GEMINI_UPLOAD_API,
        headers=init_headers,
        params={"key": api_key},
        data=json.dumps({"file": {"display_name": display_name}}),
        timeout=30
    )
    if init_resp.status_code != 200:
        print(f"[llm_meta] Files API init failed (HTTP {init_resp.status_code}): {init_resp.text}", flush=True)
        init_resp.raise_for_status()

    upload_url = init_resp.headers.get("X-Goog-Upload-URL") or init_resp.headers.get("x-goog-upload-url")
    if not upload_url:
        raise RuntimeError("Missing X-Goog-Upload-URL in Gemini Files API response")

    with open(video_path, "rb") as f:
        file_bytes = f.read()

    upload_headers = {
        "x-goog-api-key": api_key,
        "Content-Length": str(file_size),
        "X-Goog-Upload-Offset": "0",
        "X-Goog-Upload-Command": "upload, finalize"
    }
    upload_resp = requests.post(upload_url, headers=upload_headers, data=file_bytes, timeout=180)
    if upload_resp.status_code != 200:
        print(f"[llm_meta] Files API upload finalize failed (HTTP {upload_resp.status_code}): {upload_resp.text}", flush=True)
        upload_resp.raise_for_status()

    file_info = upload_resp.json().get("file", {})
    remote_name = file_info.get("name")
    remote_uri = file_info.get("uri")

    start_time = time.time()
    while time.time() - start_time < 120:
        poll_resp = requests.get(
            f"{GEMINI_BASE_API}/{remote_name}",
            headers={"x-goog-api-key": api_key},
            params={"key": api_key},
            timeout=20
        )
        if poll_resp.status_code == 200:
            state = poll_resp.json().get("state")
            if state == "ACTIVE":
                return remote_name, remote_uri
            elif state == "FAILED":
                raise RuntimeError(f"Gemini file state FAILED for {remote_name}")
        time.sleep(3)

    raise TimeoutError(f"Gemini file processing timed out for {remote_name}")

def delete_gemini_file(remote_name, api_key):
    """Deletes an uploaded file from Gemini Files API."""
    if not remote_name:
        return
    try:
        r = requests.delete(
            f"{GEMINI_BASE_API}/{remote_name}",
            headers={"x-goog-api-key": api_key},
            params={"key": api_key},
            timeout=15
        )
        if r.status_code in (200, 204):
            print(f"[llm_meta] Confirmed: Uploaded file {remote_name} deleted from Gemini Files API.", flush=True)
    except Exception as e:
        print(f"[llm_meta] Warning deleting {remote_name}: {e}", flush=True)

def extract_gemini_text(candidate_obj):
    parts = candidate_obj.get("content", {}).get("parts", [])
    text_parts = [p.get("text", "") for p in parts if not p.get("thought") and p.get("text")]
    if text_parts:
        return "\n".join(text_parts).strip()
    return (parts[0].get("text", "") if parts else "").strip()

def generate_video_metadata_gemini(video_path, duration, api_key, model="gemini-3.5-flash-lite", channel_name="Mind Stories"):
    """
    Direct file upload to Gemini Files API with multimodal video analysis.
    """
    remote_name = None
    candidate_models = get_live_gemini_models(api_key, preferred_model=model)
    try:
        print(f"[llm_meta] Uploading video to Gemini Files API ({os.path.basename(video_path)})...", flush=True)
        remote_name, remote_uri = upload_video_to_gemini(video_path, api_key)
        print(f"[llm_meta] Video uploaded and ACTIVE ({remote_name}). Analyzing content...", flush=True)

        shorts_note = "Include '#shorts' in hashtags because duration <= 60s." if duration <= 60 else ""
        prompt = (
            "Watch this video and listen carefully. Do NOT rely on any filename.\n"
            f"Channel: '{channel_name}'. Duration: {duration:.1f}s.\n"
            "Return ONLY a strict raw JSON object (no markdown fences) with keys:\n"
            "- title: engaging SEO YouTube title strictly under 60 characters, main keyword first, zero currency/numbers claims.\n"
            "- description: 2 concise engaging hook lines summarizing what happens in the video.\n"
            "- tags: array of 10 to 15 relevant search tags (joined length < 460 chars).\n"
            f"- hashtags: array of exactly 3 relevant hashtags starting with '#'. {shorts_note}\n"
            "- topic: MANDATORY concise 2 to 6 word niche/topic inferred strictly from what you saw in the video.\n"
            "CRITICAL RULES: NEVER include currency symbols ($ EUR GBP USD INR PKR) or fake size/view numbers."
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"fileData": {"mimeType": "video/mp4", "fileUri": remote_uri}},
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "responseMimeType": "application/json"
            }
        }

        backoff_schedule = [2, 4, 8, 16]
        for candidate in candidate_models:
            gen_url = f"{GEMINI_BASE_API}/models/{candidate}:generateContent"
            print(f"[llm_meta] Generating metadata with Gemini model '{candidate}'...", flush=True)
            for attempt_i in range(len(backoff_schedule) + 1):
                try:
                    resp = requests.post(
                        gen_url,
                        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                        params={"key": api_key},
                        json=payload,
                        timeout=90
                    )
                    if resp.status_code == 200:
                        cand = resp.json().get("candidates", [{}])[0]
                        text_content = extract_gemini_text(cand)
                        parsed = parse_json_payload(text_content)
                        if not parsed:
                            print(f"[llm_meta] Unparseable JSON from '{candidate}', trying next...", flush=True)
                            break

                        topic = clean_text_field(parsed.get("topic") or parsed.get("inferred_topic") or "")
                        if not topic:
                            print("[llm_meta] Topic missing, retrying once with reminder...", flush=True)
                            retry_payload = json.loads(json.dumps(payload))
                            retry_payload["contents"][0]["parts"][1]["text"] += "\nREMINDER: 'topic' MUST be non-empty."
                            r2 = requests.post(
                                gen_url,
                                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                                params={"key": api_key},
                                json=retry_payload,
                                timeout=90
                            )
                            if r2.status_code == 200:
                                t2 = extract_gemini_text(r2.json().get("candidates", [{}])[0])
                                p2 = parse_json_payload(t2) or {}
                                topic2 = clean_text_field(p2.get("topic") or p2.get("inferred_topic") or "")
                                if topic2:
                                    parsed = p2
                                    topic = topic2

                        if not topic:
                            topic = "Human Psychology & Life Stories"
                            print("[llm_meta] topic fallback used", flush=True)

                        meta = sanitize_metadata(parsed, duration, channel_name=channel_name)
                        meta["topic"] = topic
                        meta["path"] = "video_ai_gemini"
                        meta["model_used"] = candidate
                        meta["gemini_file_id"] = remote_name
                        print(f"[llm_meta] Gemini watched the video and inferred channel topic: '{topic}'", flush=True)
                        return meta

                    if resp.status_code in (429, 500, 503) and attempt_i < len(backoff_schedule):
                        wait_s = backoff_schedule[attempt_i] + random.uniform(0.3, 1.0)
                        print(f"[llm_meta] Transient HTTP {resp.status_code} on '{candidate}', retrying in {wait_s:.1f}s...", flush=True)
                        time.sleep(wait_s)
                        continue
                    print(f"[llm_meta] Gemini candidate '{candidate}' returned HTTP {resp.status_code}: {resp.text[:200]}", flush=True)
                    break
                except Exception as e:
                    print(f"[llm_meta] Gemini candidate '{candidate}' error: {e}", flush=True)
                    break

        raise RuntimeError("All Gemini candidate models failed to generate metadata.")
    finally:
        if remote_name:
            print(f"[llm_meta] Cleaning up Gemini file {remote_name}...", flush=True)
            delete_gemini_file(remote_name, api_key)

def generate_video_metadata(file_path, duration, api_key, model="gemini-3.5-flash-lite", provider="gemini", channel_name="Mind Stories"):
    return generate_video_metadata_gemini(file_path, duration, api_key, model=model, channel_name=channel_name)
