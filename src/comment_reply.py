"""
Automated YouTube Comment Reply module powered by Groq and Gemini.
"""
import socket

_orig = socket.getaddrinfo

def _ipv4(host, port, family=0, type=0, proto=0, flags=0):
    if family == 0 or family == socket.AF_UNSPEC:
        family = socket.AF_INET
    return _orig(host, port, family, type, proto, flags)

socket.getaddrinfo = _ipv4

import argparse
import datetime
import json
import os
import re
import requests
from src.auth import get_credentials, get_gemini_key, get_groq_key, validate_environment
from src.yt_client import get_youtube_service, get_channel_info

EMOJI_PATTERN = re.compile(
    r"[\U00010000-\U0010ffff]|[\u2600-\u27ff]",
    flags=re.UNICODE
)

def is_spam_or_ineligible(text):
    t = text.strip()
    if len(t) < 3:
        return True
    non_emoji = EMOJI_PATTERN.sub("", t).strip()
    if len(non_emoji) == 0:
        return True
    spam_phrases = ["sub4sub", "check my channel", "follow me", "free money", "whatsapp", "telegram", "crypto"]
    if any(p in t.lower() for p in spam_phrases):
        return True
    return False

def generate_reply_with_groq(video_title, video_description, comment_text, api_key, model="qwen/qwen3.8-27b", channel_name="Mind Stories"):
    prompt = (
        f"You are the creator of YouTube channel '{channel_name}'.\n"
        f"Video Title: {video_title}\n"
        f"Video Description: {video_description}\n"
        f"Viewer Comment: {comment_text}\n\n"
        "Write ONE natural, friendly reply to this comment.\n"
        "Rules:\n"
        "- Exactly 1 to 2 short sentences.\n"
        "- Friendly, authentic, creator tone.\n"
        "- Do NOT include any links or URLs.\n"
        "- Do NOT ask for subscribers, likes, or shares.\n"
        "- Maximum 1 emoji total.\n"
        "- Respond in the same language as the viewer's comment.\n"
        "- Return ONLY the reply text, no quotes or explanations."
    )

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 150
    }

    res = requests.post(url, headers=headers, json=payload, timeout=30)
    res.raise_for_status()
    reply = res.json()["choices"][0]["message"]["content"].strip()
    if reply.startswith('"') and reply.endswith('"'):
        reply = reply[1:-1].strip()
    return reply

def generate_reply_with_gemini(video_title, video_description, comment_text, api_key, model="gemini-3.5-flash-lite", channel_name="Mind Stories"):
    prompt = (
        f"You are the creator of YouTube channel '{channel_name}'.\n"
        f"Video Title: {video_title}\n"
        f"Video Description: {video_description}\n"
        f"Viewer Comment: {comment_text}\n\n"
        "Write ONE natural, friendly reply to this comment.\n"
        "Rules:\n"
        "- Exactly 1 to 2 short sentences.\n"
        "- Friendly, authentic, creator tone.\n"
        "- Do NOT include any links or URLs.\n"
        "- Do NOT ask for subscribers, likes, or shares.\n"
        "- Maximum 1 emoji total.\n"
        "- Respond in the same language as the viewer's comment.\n"
        "- Return ONLY the reply text, no quotes or explanations."
    )

    candidates = []
    for m in [
        model,
        "gemini-3.5-flash-lite",
        "gemini-flash-lite-latest",
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-flash-latest",
        "gemini-3.1-flash-lite",
    ]:
        if m and m not in candidates:
            candidates.append(m)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 150}
    }
    last_err = None
    for cand_model in candidates:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{cand_model}:generateContent"
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json"
        }
        try:
            res = requests.post(url, headers=headers, params={"key": api_key}, json=payload, timeout=30)
            if res.status_code != 200:
                last_err = f"HTTP {res.status_code}: {res.text[:150]}"
                continue
            cand = res.json().get("candidates", [{}])[0]
            parts = cand.get("content", {}).get("parts", [])
            text_parts = [p.get("text", "") for p in parts if not p.get("thought") and p.get("text")]
            reply = (" ".join(text_parts) if text_parts else (parts[0].get("text", "") if parts else "")).strip()
            if reply:
                if reply.startswith('"') and reply.endswith('"'):
                    reply = reply[1:-1].strip()
                return reply
        except Exception as e:
            last_err = str(e)
    raise RuntimeError(f"All Gemini models failed for comment reply: {last_err}")

def run_comment_replies(dry_run=False):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(base_dir, "config.json")
    with open(config_path, "r", encoding="utf-8-sig") as f:
        config = json.load(f)

    reply_cfg = config.get("comment_reply", {})
    if not reply_cfg.get("enabled", True):
        print("[reply] Comment reply module is disabled in config.json")
        return

    ai_provider = config.get("ai_provider", "groq")
    model = config.get("model", "qwen/qwen3.8-27b")

    # Provider-aware validation
    validate_environment(ai_provider=ai_provider, config=config)

    creds = get_credentials()
    if ai_provider == "groq":
        ai_key = get_groq_key()
    else:
        ai_key = get_gemini_key()

    service = get_youtube_service(creds)
    channel_info = get_channel_info(service)
    channel_id = channel_info["id"]

    max_replies = reply_cfg.get("max_replies_per_run", 15)
    lookback_days = reply_cfg.get("lookback_days", 7)
    cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=lookback_days)

    state_path = os.path.join(base_dir, "state", "replied.json")
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8-sig") as f:
            state = json.load(f)
    else:
        state = {"replied": [], "history": []}
    replied_ids = set(state.get("replied", []))

    print(f"[reply] Fetching comment threads for channel {channel_id} (last {lookback_days} days)...")
    threads_to_process = []
    page_token = None

    while True:
        try:
            req = service.commentThreads().list(
                part="snippet,replies",
                allThreadsRelatedToChannelId=channel_id,
                order="time",
                maxResults=100,
                pageToken=page_token
            )
            res = req.execute()
        except Exception as e:
            print(f"[reply] Note: No comments found or error listing comments: {e}")
            break

        items = res.get("items", [])
        for item in items:
            thread_id = item["id"]
            if thread_id in replied_ids:
                continue

            top = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
            published_str = top.get("publishedAt")
            if published_str:
                pub_time = datetime.datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                if pub_time < cutoff_date:
                    continue

            author_channel = top.get("authorChannelId", {}).get("value")
            if author_channel == channel_id:
                continue

            replies = item.get("replies", {}).get("comments", [])
            already_replied = any(r.get("snippet", {}).get("authorChannelId", {}).get("value") == channel_id for r in replies)
            if already_replied:
                continue

            comment_text = top.get("textOriginal", "")
            if is_spam_or_ineligible(comment_text):
                continue

            video_id = top.get("videoId")
            threads_to_process.append({
                "thread_id": thread_id,
                "video_id": video_id,
                "comment_text": comment_text,
            })
            if len(threads_to_process) >= max_replies:
                break

        page_token = res.get("nextPageToken")
        if not page_token or len(threads_to_process) >= max_replies:
            break

    if not threads_to_process:
        print("[reply] No eligible comments found to reply to.")
        return

    print(f"[reply] Found {len(threads_to_process)} eligible comments to reply.")
    for item in threads_to_process:
        thread_id = item["thread_id"]
        video_id = item["video_id"]
        text = item["comment_text"]

        v_title = "Mind Stories Short"
        v_desc = ""
        if video_id:
            try:
                vres = service.videos().list(part="snippet", id=video_id).execute()
                v_items = vres.get("items", [])
                if v_items:
                    v_title = v_items[0]["snippet"]["title"]
                    v_desc = v_items[0]["snippet"]["description"]
            except Exception:
                pass

        if ai_provider == "groq":
            reply_text = generate_reply_with_groq(v_title, v_desc, text, ai_key, model=model)
        else:
            reply_text = generate_reply_with_gemini(v_title, v_desc, text, ai_key, model=model)

        print(f'[reply] {video_id} {thread_id} -> "{reply_text}"')

        if not dry_run:
            service.comments().insert(
                part="snippet",
                body={
                    "snippet": {
                        "parentId": thread_id,
                        "textOriginal": reply_text
                    }
                }
            ).execute()
            state.setdefault("replied", []).append(thread_id)
            state.setdefault("history", []).append({
                "thread_id": thread_id,
                "video_id": video_id,
                "comment": text,
                "reply": reply_text,
                "time": datetime.datetime.now(datetime.timezone.utc).isoformat()
            })

    if not dry_run:
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without posting replies")
    args = parser.parse_args()
    run_comment_replies(dry_run=args.dry_run)
