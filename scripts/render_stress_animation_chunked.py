from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import h5py

from render_stress_frames import (
    COMBINED_STRESS_MODE,
    SINGLE_STRESS_MODES,
    _frame_is_complete,
    _frame_path,
    make_video,
    render_all_frames,
)


def chunk_ranges(total_frames: int, chunk_frames: int) -> list[tuple[int, int]]:
    if total_frames <= 0:
        raise ValueError("The HDF5 history contains no frames.")
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive.")
    return [
        (start, min(total_frames, start + chunk_frames))
        for start in range(0, total_frames, chunk_frames)
    ]


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict[str, object], destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _segment_is_current(
    segment: Path,
    marker: Path,
    signature: dict[str, object],
) -> bool:
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        return (
            payload.get("signature") == signature
            and segment.stat().st_size > 0
            and payload.get("segment_bytes") == segment.stat().st_size
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _clear_frame_window(frames_dir: Path, start: int, stop: int) -> int:
    removed = 0
    for frame_idx in range(start, stop):
        frame = _frame_path(frames_dir, frame_idx)
        try:
            frame.unlink()
            removed += 1
        except FileNotFoundError:
            pass
    return removed


def _verify_frame_window(frames_dir: Path, start: int, stop: int) -> None:
    incomplete = [
        frame_idx
        for frame_idx in range(start, stop)
        if not _frame_is_complete(_frame_path(frames_dir, frame_idx))
    ]
    if incomplete:
        preview = ", ".join(str(frame_idx) for frame_idx in incomplete[:10])
        raise RuntimeError(
            f"Frame window [{start}, {stop}) has {len(incomplete)} incomplete PNGs: "
            f"{preview}"
        )


def _concat_segments(
    segments: list[Path],
    scratch_dir: Path,
    output: Path,
) -> None:
    manifest = scratch_dir / "segments.txt"
    manifest.write_text(
        "".join(f"file '{segment.resolve()}'\n" for segment in segments),
        encoding="utf-8",
    )
    scratch_video = scratch_dir / "complete-animation.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(scratch_video),
        ],
        check=True,
    )
    if not scratch_video.is_file() or scratch_video.stat().st_size == 0:
        raise RuntimeError("FFmpeg produced an empty final animation.")
    _atomic_copy(scratch_video, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render every stress frame in bounded batches, encode each batch, and "
            "release its PNG files before continuing."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, default=3000)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--dpi", type=int, default=320)
    parser.add_argument("--width", type=float, default=15.75)
    parser.add_argument("--height", type=float, default=6.75)
    parser.add_argument("--deform-scale", type=float, default=None)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", type=str, default="medium")
    parser.add_argument("--stress-percentile", type=float, default=99.5)
    parser.add_argument(
        "--stress-mode",
        choices=[*SINGLE_STRESS_MODES, COMBINED_STRESS_MODE],
        default=COMBINED_STRESS_MODE,
    )
    parser.add_argument("--swap-axes", dest="swap_axes", action="store_true")
    parser.add_argument("--no-swap-axes", dest="swap_axes", action="store_false")
    parser.set_defaults(swap_axes=False)
    parser.add_argument("--margin", type=float, default=8.0)
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Retain PNG files after each segment (not suitable for tight quotas).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    frames_dir = args.frames_dir.expanduser().resolve()
    video = args.video.expanduser().resolve()
    scratch_dir = args.scratch_dir.expanduser().resolve()
    segments_dir = frames_dir / ".video_segments"
    frames_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as h5:
        total_frames = int(h5["history"].shape[0])
    source_stat = input_path.stat()
    common_signature: dict[str, object] = {
        "source": str(input_path),
        "source_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "total_frames": total_frames,
        "fps": args.fps,
        "crf": args.crf,
        "preset": args.preset,
        "dpi": args.dpi,
        "width": args.width,
        "height": args.height,
        "deform_scale": args.deform_scale,
        "stress_percentile": args.stress_percentile,
        "stress_mode": args.stress_mode,
        "swap_axes": args.swap_axes,
        "margin": args.margin,
    }
    windows = chunk_ranges(total_frames, args.chunk_frames)
    print(
        f"[chunked] {total_frames} frames in {len(windows)} chunks of at most "
        f"{args.chunk_frames}; scratch={scratch_dir}",
        flush=True,
    )

    segments: list[Path] = []
    try:
        for chunk_number, (start, stop) in enumerate(windows, start=1):
            stem = f"stress_{start:07d}_{stop - 1:07d}"
            segment = segments_dir / f"{stem}.mp4"
            marker = segments_dir / f"{stem}.json"
            signature = {**common_signature, "frame_start": start, "frame_stop": stop}
            segments.append(segment)

            if _segment_is_current(segment, marker, signature):
                removed = 0 if args.keep_frames else _clear_frame_window(
                    frames_dir, start, stop
                )
                print(
                    f"[chunked] {chunk_number}/{len(windows)} [{start}, {stop}) "
                    f"already encoded; removed {removed} leftover PNGs",
                    flush=True,
                )
                continue

            segment.unlink(missing_ok=True)
            marker.unlink(missing_ok=True)
            print(
                f"[chunked] {chunk_number}/{len(windows)} rendering [{start}, {stop})",
                flush=True,
            )
            render_all_frames(
                input_path,
                frames_dir,
                workers=max(1, args.workers),
                dpi=args.dpi,
                width=args.width,
                height=args.height,
                deform_scale=args.deform_scale,
                frame_limit=None,
                stress_percentile=args.stress_percentile,
                swap_axes=args.swap_axes,
                margin=args.margin,
                stress_mode=args.stress_mode,
                frame_start=start,
                frame_stop=stop,
            )
            _verify_frame_window(frames_dir, start, stop)

            scratch_segment = scratch_dir / f"{stem}.mp4"
            print(
                f"[chunked] {chunk_number}/{len(windows)} encoding [{start}, {stop})",
                flush=True,
            )
            make_video(
                frames_dir,
                scratch_segment,
                fps=args.fps,
                crf=args.crf,
                preset=args.preset,
                start_number=start,
                frame_count=stop - start,
            )
            if not scratch_segment.is_file() or scratch_segment.stat().st_size == 0:
                raise RuntimeError(f"FFmpeg produced an empty segment: {scratch_segment}")

            removed = 0 if args.keep_frames else _clear_frame_window(
                frames_dir, start, stop
            )
            _atomic_copy(scratch_segment, segment)
            _atomic_json(
                {
                    "signature": signature,
                    "segment_bytes": segment.stat().st_size,
                },
                marker,
            )
            scratch_segment.unlink(missing_ok=True)
            print(
                f"[chunked] {chunk_number}/{len(windows)} committed; "
                f"removed {removed} PNGs",
                flush=True,
            )

        print(f"[chunked] concatenating {len(segments)} segments", flush=True)
        _concat_segments(segments, scratch_dir, video)
        print(f"[chunked] complete video: {video}", flush=True)

        for segment in segments:
            segment.unlink(missing_ok=True)
            segment.with_suffix(".json").unlink(missing_ok=True)
        try:
            segments_dir.rmdir()
        except OSError:
            pass
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
