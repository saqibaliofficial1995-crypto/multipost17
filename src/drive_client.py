"""
Google Drive client with full pagination, download, and file movement.
"""
import socket

_orig = socket.getaddrinfo

def _ipv4(host, port, family=0, type=0, proto=0, flags=0):
    if family == 0 or family == socket.AF_UNSPEC:
        family = socket.AF_INET
    return _orig(host, port, family, type, proto, flags)

socket.getaddrinfo = _ipv4

import io
import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm")

def get_drive_service(creds):
    return build("drive", "v3", credentials=creds)

def list_all_files_in_folder(service, folder_id):
    """
    List ALL files with full pagination (pageSize 100 + nextPageToken loop).
    Returns (video_files, non_conforming).
    """
    query = f"'{folder_id}' in parents and trashed = false"
    page_token = None
    all_items = []

    while True:
        resp = service.files().list(
            q=query,
            pageSize=100,
            fields="nextPageToken, files(id, name, mimeType, size, createdTime)",
            pageToken=page_token
        ).execute()

        files = resp.get("files", [])
        all_items.extend(files)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    video_files = []
    non_conforming = []

    for item in all_items:
        mime = item.get("mimeType", "")
        name = item.get("name", "")
        is_folder = (mime == "application/vnd.google-apps.folder")
        is_video = (mime.startswith("video/") or name.lower().endswith(VIDEO_EXTENSIONS)) and not is_folder

        if is_video:
            video_files.append(item)
        else:
            non_conforming.append(item)

    return video_files, non_conforming

def get_or_create_uploaded_folder(service, parent_folder_id, folder_name="Uploaded"):
    query = f"'{parent_folder_id}' in parents and name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    resp = service.files().list(q=query, fields="files(id, name)").execute()
    files = resp.get("files", [])
    if files:
        return files[0]["id"]

    metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id]
    }
    created = service.files().create(body=metadata, fields="id").execute()
    return created["id"]

def download_drive_file(service, file_id, dest_path):
    request = service.files().get_media(fileId=file_id)
    with open(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=1024*1024*8)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"[drive] Download progress: {int(status.progress() * 100)}%")

def move_file_to_folder(service, file_id, source_folder_id, target_folder_id):
    service.files().update(
        fileId=file_id,
        addParents=target_folder_id,
        removeParents=source_folder_id,
        fields="id, parents"
    ).execute()
