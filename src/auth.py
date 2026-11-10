"""
Authentication helper for YouTube, Drive, Analytics, Gemini, and Groq APIs.
"""
import socket

_orig = socket.getaddrinfo

def _ipv4(host, port, family=0, type=0, proto=0, flags=0):
    if family == 0 or family == socket.AF_UNSPEC:
        family = socket.AF_INET
    return _orig(host, port, family, type, proto, flags)

socket.getaddrinfo = _ipv4

import os
import json
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
    "https://www.googleapis.com/auth/drive",
]

def get_base_dir():
    curr = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(curr, ".."))

def get_tokens_dir():
    base_dir = get_base_dir()
    p1 = os.path.join(base_dir, "tokens")
    if os.path.exists(p1):
        return p1
    p2 = os.path.abspath(os.path.join(base_dir, "..", "tokens"))
    if os.path.exists(p2):
        return p2
    os.makedirs(p1, exist_ok=True)
    return p1

def validate_environment(ai_provider=None, config=None):
    """
    PROVIDER-AWARE VALIDATION everywhere.
    Always require YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN, DRIVE_FOLDER_ID.
    Then require GROQ_API_KEY when ai_provider=="groq"
    and GEMINI_API_KEY when =="gemini".
    Applies to upload pipeline, comment reply, and analytics.
    """
    missing = []
    tokens_dir = get_tokens_dir()
    has_local_token = os.path.exists(os.path.join(tokens_dir, "token.json"))

    # Validate YouTube credentials
    for var in ["YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN"]:
        val = os.environ.get(var)
        if not val or not val.strip():
            if not has_local_token:
                missing.append(var)

    # Validate DRIVE_FOLDER_ID
    drive_id = os.environ.get("DRIVE_FOLDER_ID")
    if not drive_id and config:
        drive_id = config.get("drive_folder_id")
    if not drive_id or not drive_id.strip() or drive_id.startswith("__"):
        missing.append("DRIVE_FOLDER_ID")

    # Validate AI Provider Key
    provider = (ai_provider or (config.get("ai_provider") if config else None) or "groq").lower()
    if provider == "groq":
        key = os.environ.get("GROQ_API_KEY")
        local_key_file = os.path.join(tokens_dir, "groq_key.txt")
        has_local_key = os.path.exists(local_key_file)
        if (not key or not key.strip()) and not has_local_key:
            missing.append("GROQ_API_KEY")
    elif provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        local_key_file = os.path.join(tokens_dir, "gemini_key.txt")
        has_local_key = os.path.exists(local_key_file)
        if (not key or not key.strip()) and not has_local_key:
            missing.append("GEMINI_API_KEY")

    if missing:
        raise ValueError(
            f"PROVIDER-AWARE VALIDATION FAILED: Missing required configuration/keys: {', '.join(missing)} "
            f"(provider={provider})"
        )

def get_credentials():
    # 1. Env-first credentials
    client_id = os.environ.get("YT_CLIENT_ID")
    client_secret = os.environ.get("YT_CLIENT_SECRET")
    refresh_token = os.environ.get("YT_REFRESH_TOKEN")

    if client_id and client_secret and refresh_token:
        creds = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
            scopes=SCOPES,
        )
        creds.refresh(Request())
        return creds

    # 2. tokens/token.json fallback
    tokens_dir = get_tokens_dir()
    token_path = os.path.join(tokens_dir, "token.json")
    client_secret_path = os.path.join(tokens_dir, "client_secret.json")

    creds = None
    if os.path.exists(token_path):
        with open(token_path, "r", encoding="utf-8-sig") as f:
            token_data = json.load(f)
        creds = Credentials.from_authorized_user_info(token_data, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(client_secret_path):
                raise FileNotFoundError(f"Client secret not found at {client_secret_path}")
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save tokens/token.json (UTF-8 no BOM)
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    return creds

def get_gemini_key():
    key = os.environ.get("GEMINI_API_KEY")
    if key and key.strip():
        return key.strip()

    tokens_dir = get_tokens_dir()
    key_file = os.path.join(tokens_dir, "gemini_key.txt")
    if os.path.exists(key_file):
        with open(key_file, "r", encoding="utf-8-sig") as f:
            k = f.read().strip()
            if k:
                return k

    raise ValueError("GEMINI_API_KEY not found in environment or tokens/gemini_key.txt")

def get_groq_key():
    key = os.environ.get("GROQ_API_KEY")
    if key and key.strip():
        return key.strip()

    tokens_dir = get_tokens_dir()
    key_file = os.path.join(tokens_dir, "groq_key.txt")
    if os.path.exists(key_file):
        with open(key_file, "r", encoding="utf-8-sig") as f:
            k = f.read().strip()
            if k:
                return k

    raise ValueError("GROQ_API_KEY not found in environment or tokens/groq_key.txt")
