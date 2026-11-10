"""
CI Orchestrator: Drive Audit -> Download -> Normalize -> LLM Metadata -> Validate -> Upload -> Move -> Report.
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
import tempfile
from src.auth import get_credentials, get_gemini_key, get_groq_key, validate_environment
from src.drive_client import (
    get_drive_service,
    list_all_files_in_folder,
    get_or_create_uploaded_folder,
    download_drive_file,
    move_file_to_folder,
)
from src.normalize import get_video_duration, normalize_video
from src.llm_meta import generate_video_metadata
from src.yt_client import get_youtube_service, get_channel_info, upload_video

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without uploading")
    parser.add_argument("--max-uploads", type=int, default=None, help="Max videos to upload")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(base_dir, "config.json")
    with open(config_path, "r", encoding="utf-8-sig") as f:
        config = json.load(f)

    max_uploads = args.max_uploads if args.max_uploads is not None else config.get("max_uploads_per_run", 1)
    folder_id = os.environ.get("DRIVE_FOLDER_ID") or config.get("drive_folder_id")
    privacy_status = config.get("privacy_status", "public")
    made_for_kids = config.get("made_for_kids", False)
    ai_provider = config.get("ai_provider", "groq")
    model = config.get("model", "qwen/qwen3.8-27b")

    # Provider-aware validation
    validate_environment(ai_provider=ai_provider, config=config)

    creds = get_credentials()
    if ai_provider == "groq":
        ai_key = get_groq_key()
    else:
        ai_key = get_gemini_key()

    drive_service = get_drive_service(creds)
    yt_service = get_youtube_service(creds)

    # Channel info before
    ch_info_before = get_channel_info(yt_service)
    print(f"[ci_run] Target Channel: {ch_info_before['title']} (ID: {ch_info_before['id']})")
    print(f"[ci_run] Channel Video Count Before: {ch_info_before['video_count']}")

    # Drive audit
    uploaded_folder_id = get_or_create_uploaded_folder(drive_service, folder_id)
    videos, non_conforming = list_all_files_in_folder(drive_service, folder_id)
    drive_count_before = len(videos)
    print(f"[drive] Drive True Video Count Before: {drive_count_before}")
    print(f"[drive] Drive Non-conforming Files: {len(non_conforming)}")

    # Load state
    state_path = os.path.join(base_dir, "state", "uploaded.json")
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8-sig") as f:
            state = json.load(f)
    else:
        state = {"items": {}, "history": []}
    uploaded_items = state.get("items", {})

    pending_videos = [v for v in videos if v["id"] not in uploaded_items]
    print(f"[drive] Pending videos found: {len(pending_videos)}")

    if not pending_videos:
        print("[ci_run] No pending videos to upload. Exiting.")
        return

    to_upload = pending_videos[:max_uploads]

    with tempfile.TemporaryDirectory() as tmpdir:
        for vid in to_upload:
            file_id = vid["id"]
            file_name = vid["name"]
            print(f"[drive] Processing video: {file_name} (ID: {file_id})")

            raw_path = os.path.join(tmpdir, f"raw_{file_id}.mp4")
            norm_path = os.path.join(tmpdir, f"norm_{file_id}.mp4")

            # 1. Download
            download_drive_file(drive_service, file_id, raw_path)
            print(f"[drive] Download complete: {raw_path}")

            # 2. Normalize & measure
            duration = get_video_duration(raw_path)
            normalize_video(raw_path, norm_path)
            norm_duration = get_video_duration(norm_path)
            print(f"[normalize] Normalized video duration: {norm_duration:.2f}s")

            # 3. LLM metadata
            meta = generate_video_metadata(
                norm_path,
                norm_duration,
                ai_key,
                model=model,
                provider=ai_provider
            )

            print(f"[meta] Model used: {meta.get('model_used', model)}")
            print(f"[meta] Inferred topic: {meta.get('topic')}")
            print(f"[meta] Title ({len(meta['title'])} chars): {meta['title']}")
            print(f"[meta] Tags ({len(meta['tags'])}): {', '.join(meta['tags'])}")
            print(f"[meta] Hashtags: {' '.join(meta['hashtags'])}")

            if args.dry_run:
                print(f"[upload] DRY RUN: Would upload video '{meta['title']}' to YouTube.")
                continue

            # 4. Upload
            yt_id = upload_video(yt_service, norm_path, meta, privacy_status, made_for_kids)
            print(f"[upload] Uploaded to YouTube with ID: {yt_id} (URL: https://youtu.be/{yt_id})")

            # 5. Move to Uploaded
            move_file_to_folder(drive_service, file_id, folder_id, uploaded_folder_id)
            print(f"[drive] Moved file {file_id} to Uploaded folder")

            # 6. Update state
            state["items"][file_id] = {
                "name": file_name,
                "yt_id": yt_id,
                "yt_url": f"https://youtu.be/{yt_id}",
                "title": meta["title"],
                "topic": meta["topic"],
                "tags_count": len(meta["tags"]),
                "hashtags": meta["hashtags"],
                "uploaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            state.setdefault("history", []).append({
                "drive_id": file_id,
                "yt_id": yt_id,
                "title": meta["title"],
                "topic": meta["topic"],
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            })

    if not args.dry_run:
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        print("[state] Uploaded state saved to state/uploaded.json")

        # After counts
        videos_after, _ = list_all_files_in_folder(drive_service, folder_id)
        drive_count_after = len(videos_after)
        ch_info_after = get_channel_info(yt_service)

        print("[ci_run] === UPLOAD REPORT ===")
        print(f"Drive Count: {drive_count_before} -> {drive_count_after}")
        print(f"Channel Video Count: {ch_info_before['video_count']} -> {ch_info_after['video_count']}")

if __name__ == "__main__":
    main()
