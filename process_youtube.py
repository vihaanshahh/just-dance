#!/usr/bin/env python3
"""
Process a YouTube video directly into Just Dance style.

Usage:
    python process_youtube.py
    python process_youtube.py "https://www.youtube.com/watch?v=MG8r5YlHurE"
    python process_youtube.py "https://www.youtube.com/watch?v=MG8r5YlHurE" output.mp4
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from just_dance.core.video_loader import YouTubeDownloader, VideoReader
from just_dance.core.pipeline import ProcessingPipeline, PipelineConfig


def log(message, stage=""):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if stage:
        print(f"[{timestamp}] [{stage}] {message}")
    else:
        print(f"[{timestamp}] {message}")


def format_time(seconds):
    """Format seconds as MM:SS or HH:MM:SS."""
    if seconds < 3600:
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main():
    print()
    print("╔══════════════════════════════════════════╗")
    print("║     JUST DANCE VIDEO PROCESSOR           ║")
    print("╚══════════════════════════════════════════╝")
    print()

    if len(sys.argv) < 2:
        url = input("Enter YouTube URL: ").strip()
        if not url:
            print("No URL provided. Exiting.")
            sys.exit(1)
        output_path = None
    else:
        url = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None

    # ═══════════════════════════════════════════════════════════
    # STAGE 1: DOWNLOAD
    # ═══════════════════════════════════════════════════════════
    print("─" * 50)
    log("Starting download...", "DOWNLOAD")
    log(f"URL: {url}", "DOWNLOAD")

    # Get video info first
    downloader = YouTubeDownloader()
    try:
        info = downloader.get_video_info(url)
        if info.get("title"):
            log(f"Title: {info['title']}", "DOWNLOAD")
        if info.get("duration"):
            log(f"Duration: {format_time(info['duration'])}", "DOWNLOAD")
    except Exception:
        pass  # Info fetch failed, continue anyway

    print()

    download_start = time.time()
    last_percent = [0]

    def download_progress(percent):
        last_percent[0] = percent
        elapsed = time.time() - download_start

        bar_length = 25
        filled = int(bar_length * percent / 100)
        bar = "█" * filled + "░" * (bar_length - filled)

        # Estimate remaining time
        if percent > 0:
            eta = (elapsed / percent) * (100 - percent)
            eta_str = format_time(eta)
        else:
            eta_str = "--:--"

        status = f"\r    [{bar}] {percent:5.1f}%  |  Elapsed: {format_time(elapsed)}  |  ETA: {eta_str}"
        print(status, end="", flush=True)

    try:
        video_path = downloader.download(url, download_progress)
        download_time = time.time() - download_start
        print()  # New line after progress
        print()
        log(f"Download complete in {format_time(download_time)}", "DOWNLOAD")
        log(f"Saved to: {Path(video_path).name}", "DOWNLOAD")

        # Verify file exists and is readable
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Downloaded file not found: {video_path}")

        file_size = Path(video_path).stat().st_size / (1024 * 1024)  # MB
        log(f"File size: {file_size:.1f} MB", "DOWNLOAD")

    except Exception as e:
        print()
        log(f"ERROR: {e}", "DOWNLOAD")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Get video info
    print()
    log("Reading video metadata...", "INFO")
    try:
        reader = VideoReader(video_path)
        total_frames = reader.frame_count
        fps = reader.fps
        duration = total_frames / fps if fps > 0 else 0
        width, height = reader.resolution
        reader.release()

        log(f"Resolution: {width}x{height}", "INFO")
        log(f"Duration: {format_time(duration)} ({total_frames} frames)", "INFO")
        log(f"FPS: {fps:.2f}", "INFO")
    except Exception as e:
        log(f"Warning: Could not read video info: {e}", "INFO")
        total_frames = 0
        fps = 30

    # Determine output path (absolute path in current directory)
    if output_path is None:
        output_path = str(Path.cwd() / (Path(video_path).stem + "_justdance.mp4"))

    print()
    log(f"Output will be saved to: {output_path}", "INFO")

    # ═══════════════════════════════════════════════════════════
    # STAGE 2: PROCESS
    # ═══════════════════════════════════════════════════════════
    print()
    print("─" * 50)
    log("Initializing pipeline...", "PROCESS")

    config = PipelineConfig()
    pipeline = ProcessingPipeline(config)

    log("Pipeline ready. Starting frame processing...", "PROCESS")
    print()

    process_start = time.time()

    def process_progress(percent, current, total):
        elapsed = time.time() - process_start
        fps = current / elapsed if elapsed > 0 else 0
        eta = (elapsed / percent) * (100 - percent) if percent > 0 else 0

        bar_len = 20
        filled = int(bar_len * percent / 100)
        bar = "█" * filled + "░" * (bar_len - filled)

        print(f"\033[2K\r    [{bar}] {percent:.0f}% | {current}/{total} | {fps:.0f}fps | ETA: {format_time(eta)}", end="", flush=True)

    try:
        success, message = pipeline.process_video(video_path, output_path, process_progress)
        process_time = time.time() - process_start
        print()  # New line after progress
        print()

        if success:
            # Extract frame count from message (format: "Processed N frames")
            try:
                frames_count = int(message.split()[1])
                avg_fps = frames_count / process_time if process_time > 0 else 0
            except:
                avg_fps = 0
            log(f"Processing complete!", "PROCESS")
            log(f"Time: {format_time(process_time)} ({avg_fps:.1f} fps average)", "PROCESS")

            # Final summary
            print()
            print("═" * 50)
            print("  SUCCESS!")
            print(f"  Output: {output_path}")
            print(f"  Total time: {format_time(time.time() - download_start)}")
            print("═" * 50)
            print()
        else:
            log(f"ERROR: {message}", "PROCESS")
            sys.exit(1)

    except KeyboardInterrupt:
        print()
        print()
        log("Cancelled by user", "PROCESS")
        sys.exit(1)
    except Exception as e:
        print()
        log(f"ERROR: {e}", "PROCESS")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
