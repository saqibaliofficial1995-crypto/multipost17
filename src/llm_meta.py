"""
Video metadata generation with Groq vision (keyframes) and Gemini Files API, with strict validation.
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
import re
import subprocess
import tempfile
import time
import requests

CURRENCY_PATTERN = re.compile(r"[\$\u20ac\u00a3\u00a5\u20b9]| (USD|AED|EUR|GBP|INR|PKR) ", re.IGNORECASE)
MAGNITUDE_PATTERN = re.compile(r" \d+([.,]\d+)?\s*(m|b|k|million|billion|sqft|sq\s*ft|feet) ", re.IGNORECASE)

GEMINI_BASE_API = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_UPLOAD_API = "https://generativelanguage.googleapis.com/upload/v1beta/files"

def clean_text_field(text):
    t = CURRENCY_PATTERN.sub("", text)
    t = MAGNITUDE_PATTERN.sub("", t)
    return re.sub(r"\s+", " ", t).strip()

def sanitize_metadata(meta, duration, channel_name="Mind Stories"):
    # Title < 60 chars
    title = clean_text_field(meta.get("title", "Inspiring Moment"))
    if len(title) > 59:
        title = title[:59].rsplit(" ", 1)[0]
    if not title:
        title = "Inspiring Moment"

    # Tags: 10-15 tags, total string < 460
    # Normalize string to list (handles comma-separated strings from Groq)
    raw_tags = meta.get("tags", [])
    if isinstance(raw_tags, str):
        raw_tags = [t.strip() for t in raw_tags.split(",") if t.strip()]

    clean_tags = []
    for tag in raw_tags:
        ct = clean_text_field(str(tag))
        if ct and len(ct) > 1 and ct.lower() not in [x.lower() for x in clean_tags]:
            clean_tags.append(ct)

    # Ensure 10-15 tags
    fallback_tags = [
        "lifestyle", "shorts", "daily", "inspiration", "trends",
        "video", "moments", "viral", "creative", "explore",
        "focus", "mindset", "entertainment", "reels", "highlights"
    ]
    for ft in fallback_tags:
        if len(clean_tags) >= 12:
            break
        if ft.lower() not in [x.lower() for x in clean_tags]:
            clean_tags.append(ft)

    clean_tags = clean_tags[:15]
    while len(",".join(clean_tags)) > 450 and len(clean_tags) > 10:
        clean_tags.pop()

    # Hashtags: exactly 3 hashtags, #shorts when duration <= 60
    # Normalize string to list (handles space or comma separated strings from Groq)
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

    default_hts = ["#shorts", "#lifestyle", "#viral", "#inspire"]
    for dh in default_hts:
        if len(clean_ht) >= 3:
            break
        if not any(h.lower() == dh.lower() for h in clean_ht):
            clean_ht.append(dh)

    clean_ht = clean_ht[:3]

    # Description = body + blank line + 3 hashtags + blank line + channel_name
    body = clean_text_field(meta.get("description", ""))
    if not body:
        body = f"{title}. Watch till the end for something special!"

    ht_str = " ".join(clean_ht)
    description = f"{body}\n\n{ht_str}\n\n{channel_name}"

    meta["title"] = title
    meta["description"] = description
    meta["tags"] = clean_tags
    meta["hashtags"] = clean_ht
    return meta

def get_best_groq_vision_model(api_key, fallback_model="qwen/qwen3.8-27b"):
    """
    GROQ MODEL AUTO-SELECT: Query live models endpoint at runtime,
    pick a live vision-capable model, and fall back to configured model if query fails.
    """
    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            models_data = resp.json().get("data", [])
            # Priority candidates for vision on Groq
            active_models = [m["id"] for m in models_data if m.get("active", True)]
            candidates = [
                m for m in active_models
                if "vision" in m.lower() or "qwen" in m.lower()
            ]
            preferred_order = [
                "meta-llama/llama-3.2-11b-vision-preview",
                "llama-3.2-11b-vision-preview",
                "meta-llama/llama-3.2-90b-vision-preview",
                "llama-3.2-90b-vision-preview",
                "qwen/qwen3.8-27b",
            ]
            for pref in preferred_order:
                if pref in candidates:
                    print(f"[groq] Auto-selected live vision model: {pref}")
                    return pref
            if candidates:
                chosen = candidates[0]
                print(f"[groq] Auto-selected live vision model: {chosen}")
                return chosen
    except Exception as e:
        print(f"[groq] Warning querying live models endpoint: {e}")

    print(f"[groq] Using configured fallback model: {fallback_model}")
    return fallback_model

def extract_keyframes(video_path, num_frames=3, duration_seconds=60):
    temp_dir = tempfile.mkdtemp(prefix="groq_frames_")
    b64_frames = []
    try:
        dur = max(1.0, float(duration_seconds))
        fractions = [0.2, 0.5, 0.8] if num_frames == 3 else [0.25, 0.5, 0.75]
        for i, frac in enumerate(fractions):
            ts = max(0.1, dur * frac)
            frame_path = os.path.join(temp_dir, f"frame_{i}.jpg")
            cmd = ["ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", video_path, "-vframes", "1", "-q:v", "2", frame_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if os.path.exists(frame_path):
                with open(frame_path, "rb") as f:
                    b64_frames.append(base64.b64encode(f.read()).decode("utf-8"))
    finally:
        for f in os.listdir(temp_dir):
            try: os.remove(os.path.join(temp_dir, f))
            except: pass
        try: os.rmdir(temp_dir)
        except: pass
    return b64_frames

def generate_video_metadata_groq(video_path, duration, api_key, model="qwen/qwen3.8-27b", channel_name="Mind Stories"):
    # Auto-select active vision model
    selected_model = get_best_groq_vision_model(api_key, fallback_model=model)

    b64_frames = extract_keyframes(video_path, num_frames=3, duration_seconds=duration)
    if not b64_frames:
        raise RuntimeError("Failed to extract keyframes with ffmpeg.")

    prompt = (
        "Watch these 3 video frames carefully. Do NOT use the video filename to guess content.\n"
        "Return ONLY a raw JSON object with no markdown fences, no formatting before or after.\n"
        "Required JSON keys:\n"
        "- title: engaging YouTube title, strictly under 60 characters, no currency symbols or number claims.\n"
        "- description: 2-3 engaging sentences describing what is happening visually in the video.\n"
        "- tags: array of 10 to 15 relevant keyword tags (single words or short phrases).\n"
        "- hashtags: array of exactly 3 relevant hashtags starting with '#' (include #shorts if short/vertical).\n"
        "- topic: a specific 1 to 4 word thematic niche/topic of this video inferred strictly from the video visuals.\n"
        "CRITICAL RULES: Do NOT include currency ($ € £ ¥ ₹) or monetary words. "
        "Do NOT mention sizes or magnitude claims. The 'topic' field is strictly MANDATORY."
    )

    content = [{"type": "text", "text": prompt}]
    for b64_img in b64_frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
        })

    payload = {
        "model": selected_model,
        "messages": [
            {"role": "user", "content": content}
        ],
        "temperature": 0.2,
        "max_tokens": 1000,
        "response_format": {"type": "json_object"}
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    url = "https://api.groq.com/openai/v1/chat/completions"
    res = None
    for retry_i in range(3):
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=60)
            if res.status_code == 200:
                break
            time.sleep(2 * (retry_i + 1))
        except Exception as e:
            print(f"[warning] Groq request error: {e}", flush=True)
            time.sleep(2)

    if res is None or res.status_code != 200:
        err_msg = res.text[:200] if res is not None else "No response"
        raise RuntimeError(f"Groq API call failed: {err_msg}")

    result_json = res.json()
    raw_text = result_json["choices"][0]["message"]["content"]
    meta = json.loads(raw_text)

    # Mandatory topic check and retry
    topic = meta.get("topic") or meta.get("inferred_topic") or ""
    topic = str(topic).strip()
    if not topic:
        print("[meta] Missing topic. Retrying Groq call with explicit topic reminder...", flush=True)
        retry_prompt = prompt + "\n\nREMINDER: You forgot 'topic' previously. You MUST provide the 'topic' key in the JSON output!"
        payload["messages"][0]["content"][0]["text"] = retry_prompt
        try:
            r_retry = requests.post(url, headers=headers, json=payload, timeout=60)
            if r_retry.status_code == 200:
                retry_json = r_retry.json()
                meta_retry = json.loads(retry_json["choices"][0]["message"]["content"])
                topic = meta_retry.get("topic") or meta_retry.get("inferred_topic") or ""
                topic = str(topic).strip()
                if topic:
                    meta = meta_retry
        except Exception as e:
            print(f"[meta] Retry error: {e}", flush=True)

    if not topic:
        topic = "Lifestyle & Inspiration"
        print("[meta] topic fallback used", flush=True)

    meta = sanitize_metadata(meta, duration, channel_name=channel_name)
    meta["topic"] = topic
    meta["path"] = "video_ai_groq"
    meta["model_used"] = selected_model
    return meta

def upload_video_to_gemini(video_path, api_key):
    """Uploads a video to Gemini Files API using resumable upload and waits until ACTIVE."""
    file_size = os.path.getsize(video_path)
    display_name = os.path.basename(video_path)
    mime_type = "video/mp4"

    # 1. Initiate resumable upload
    init_headers = {
        "x-goog-api-key": api_key,
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(file_size),
        "X-Goog-Upload-Header-Content-Type": mime_type,
        "Content-Type": "application/json"
    }
    init_body = json.dumps({"file": {"display_name": display_name}})
    init_resp = requests.post(
        GEMINI_UPLOAD_API,
        headers=init_headers,
        params={"key": api_key},
        data=init_body,
        timeout=30
    )
    if init_resp.status_code != 200:
        print(f"[llm_meta] Files API init failed (HTTP {init_resp.status_code}): {init_resp.text}")
        init_resp.raise_for_status()

    upload_url = init_resp.headers.get("X-Goog-Upload-URL") or init_resp.headers.get("x-goog-upload-url")
    if not upload_url:
        raise RuntimeError("Missing X-Goog-Upload-URL in Gemini Files API response")

    # 2. Upload file content
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
        print(f"[llm_meta] Files API upload chunk failed (HTTP {upload_resp.status_code}): {upload_resp.text}")
        upload_resp.raise_for_status()

    file_info = upload_resp.json().get("file", {})
    remote_name = file_info.get("name")
    remote_uri = file_info.get("uri")

    # 3. Wait for processing to become ACTIVE
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
            print(f"[llm_meta] Confirmed: Uploaded file {remote_name} deleted from Gemini Files API.")
    except Exception as e:
        print(f"[llm_meta] Warning deleting {remote_name}: {e}")

def generate_video_metadata_gemini(video_path, duration, api_key, model="gemini-2.0-flash", channel_name="Mind Stories"):
    """
    Direct file upload to Gemini Files API with multimodal video analysis.
    """
    remote_name = None
    candidate_models = [model, "gemini-2.0-flash", "gemini-1.5-flash", "gemini-flash-latest"]
    try:
        print(f"[llm_meta] Uploading video to Gemini Files API ({os.path.basename(video_path)})...")
        remote_name, remote_uri = upload_video_to_gemini(video_path, api_key)
        print(f"[llm_meta] Video uploaded and ACTIVE ({remote_name}). Analyzing content...")

        prompt = (
            "Watch this video carefully and generate YouTube metadata in strict JSON format.\n"
            "Channel creator: 'Mind Stories'.\n"
            "Video duration: " + str(duration) + " seconds.\n"
            "Required JSON keys:\n"
            "- title: engaging YouTube title, strictly under 60 characters, no currency symbols or number claims.\n"
            "- description: 2-3 engaging sentences describing what is happening in the video.\n"
            "- tags: array of 10 to 15 relevant keyword tags (single words or short phrases).\n"
            "- hashtags: array of exactly 3 relevant hashtags starting with '#' (include #shorts if short/vertical).\n"
            "- topic: a specific 1 to 4 word thematic niche/topic of this video.\n"
            "CRITICAL RULES: Do NOT include currency ($ € £ ¥ ₹) or monetary words. "
            "Do NOT mention sizes or magnitude claims. Ground everything strictly in the video visuals."
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
                "temperature": 0.2,
                "responseMimeType": "application/json"
            }
        }

        for candidate in candidate_models:
            gen_url = f"{GEMINI_BASE_API}/models/{candidate}:generateContent"
            print(f"[llm_meta] Generating metadata with Gemini model '{candidate}'...")
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
                    text_content = cand.get("content", {}).get("parts", [{}])[0].get("text", "")
                    parsed = json.loads(text_content)
                    meta = sanitize_metadata(parsed, duration, channel_name=channel_name)
                    topic = str(parsed.get("topic") or "Lifestyle & Inspiration").strip()
                    meta["topic"] = topic
                    meta["path"] = "video_ai_gemini"
                    meta["model_used"] = candidate
                    print(f"[llm_meta] Gemini analysis succeeded with model '{candidate}'.")
                    return meta
                else:
                    print(f"[llm_meta] Gemini candidate '{candidate}' returned HTTP {resp.status_code}: {resp.text[:200]}")
            except Exception as e:
                print(f"[llm_meta] Gemini candidate '{candidate}' error: {e}")

        raise RuntimeError("All Gemini candidate models failed to generate metadata.")
    finally:
        if remote_name:
            print(f"[llm_meta] Cleaning up Gemini file {remote_name}...")
            delete_gemini_file(remote_name, api_key)

def generate_video_metadata(file_path, duration, api_key, model="qwen/qwen3.8-27b", provider="groq", channel_name="Mind Stories"):
    if provider == "groq" or (model and "qwen" in str(model).lower()):
        return generate_video_metadata_groq(file_path, duration, api_key, model=model, channel_name=channel_name)
    else:
        return generate_video_metadata_gemini(file_path, duration, api_key, model=model, channel_name=channel_name)
