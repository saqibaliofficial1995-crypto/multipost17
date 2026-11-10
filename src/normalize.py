"""
Video normalization and duration measurement via ffprobe and ffmpeg.
"""
import os
import shutil
import subprocess

def get_video_duration(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    try:
        res = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return float(res)
    except Exception as e:
        print(f"[normalize] ffprobe failed to get duration: {e}. Defaulting to 60.0")
        return 60.0

def normalize_video(input_path, output_path):
    """
    Normalizes video to 1080x1920 30fps H.264 + AAC, with graceful copy fallback.
    """
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black",
        "-r", "30",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        output_path
    ]
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print("[normalize] Normalized video created successfully.")
            return True
    except Exception as e:
        print(f"[normalize] ffmpeg failed: {e}. Falling back to copy.")

    # Graceful fallback: copy original
    shutil.copyfile(input_path, output_path)
    return False
