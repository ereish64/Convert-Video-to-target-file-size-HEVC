#!/usr/bin/env python3
"""
Video Optimizer - Encode video to target file size with maximum quality

Usage:
    python video_optimizer.py <target_size> <input_file> <output_file>
    
    target_size: Target file size with unit (e.g., 10MB, 500KB, 1GB)
    input_file: Path to input video file
    output_file: Path for output video file

How it works:
    First pass: Analyze input video to extract duration, resolution, fps and bitrates
    Calculate optimal encoding parameters (resolution, fps, video bitrate) to meet target size.
    Target size is calculated to fit within a specified list of resolution and fps tiers.

    Second pass: Encode video using ffmpeg with H.265 codec with calculated parameters from first pass
"""

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class VideoInfo:
    """Container for video metadata."""
    duration: float  # seconds
    width: int
    height: int
    fps: float
    video_bitrate: Optional[int]  # kbps
    audio_bitrate: Optional[int]  # kbps


def parse_size(size_str: str) -> int:
    """Parse a size string like '10MB' or '500KB' into bytes."""
    size_str = size_str.strip().upper()
    
    units = {
        'B': 1,
        'KB': 1024,
        'K': 1024,
        'MB': 1024 ** 2,
        'M': 1024 ** 2,
        'GB': 1024 ** 3,
        'G': 1024 ** 3,
    }
    
    match = re.match(r'^([\d.]+)\s*([A-Z]+)?$', size_str)
    if not match:
        raise ValueError(f"Invalid size format: {size_str}")
    
    value = float(match.group(1))
    unit = match.group(2) or 'B'
    
    if unit not in units:
        raise ValueError(f"Unknown unit: {unit}")
    
    return int(value * units[unit])


def get_video_info(input_file: str, invert_aspect_ratio: bool) -> VideoInfo:
    """Extract video information using ffprobe."""
    cmd = [
        './ffprobe.exe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        input_file
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    
    data = json.loads(result.stdout)
    
    # extract video and audio streams
    video_stream = None
    audio_stream = None
    
    for stream in data.get('streams', []):
        if stream.get('codec_type') == 'video' and video_stream is None:
            video_stream = stream
        elif stream.get('codec_type') == 'audio' and audio_stream is None:
            audio_stream = stream
    
    if not video_stream:
        raise RuntimeError("No video stream found in input file")
    
    # Extract duration of video
    duration = float(data.get('format', {}).get('duration', 0))
    if duration == 0:
        duration = float(video_stream.get('duration', 0))
    
    if duration == 0:
        raise RuntimeError("Could not determine video duration")
    
    # Extract video height/width
    width = int(video_stream.get('width', 0))
    height = int(video_stream.get('height', 0))
    # If the user has requested to invert the aspect ratio, flip width and height variables
    if invert_aspect_ratio:
        width, height = height, width
    
    # Extract fps
    fps_str = video_stream.get('r_frame_rate', '30/1')
    if '/' in fps_str:
        num, den = fps_str.split('/')
        fps = float(num) / float(den) if float(den) != 0 else 30.0
    else:
        fps = float(fps_str)
    
    # Extract bitrates (optional)
    video_bitrate = None
    if 'bit_rate' in video_stream:
        video_bitrate = int(video_stream['bit_rate']) // 1000
    
    audio_bitrate = None
    if audio_stream and 'bit_rate' in audio_stream:
        audio_bitrate = int(audio_stream['bit_rate']) // 1000
    
    return VideoInfo(
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        video_bitrate=video_bitrate,
        audio_bitrate=audio_bitrate
    )


def calculate_optimal_params(
    video_info: VideoInfo,
    target_bytes: int,
    audio_bitrate_kbps: int = 16,
    fixed_fps: Optional[float] = None
) -> Tuple[int, int, float, int]:
    """
    Calculate optimal encoding parameters for target file size.
    
    Returns: (width, height, fps, video_bitrate_kbps)
    """
    # Reserve some space for container overhead (~2%)
    usable_bytes = int(target_bytes * 0.98)
    
    # Calculate total available bitrate in kbps
    # bitrate (kbps) = (bytes * 8) / (duration * 1000)
    total_bitrate_kbps = (usable_bytes * 8) / (video_info.duration * 1000)
    
    # Subtract audio bitrate to get video bitrate budget
    video_bitrate_kbps = total_bitrate_kbps - audio_bitrate_kbps
    
    # Determine effective FPS for minimum bitrate calculation
    # Use fixed_fps if provided, otherwise use source fps
    effective_fps = fixed_fps if fixed_fps is not None else video_info.fps
    # Scale minimum bitrate by fps (base assumption: 30fps needs 50kbps minimum)
    fps_factor = effective_fps / 30.0
    min_video_bitrate = max(50 * fps_factor, 20)  # Floor at 20 kbps absolute minimum
    
    # if video_bitrate_kbps < min_video_bitrate:
    #     raise ValueError(
    #         f"Target size too small. Need at least "
    #         f"{int((min_video_bitrate + audio_bitrate_kbps) * video_info.duration * 1000 / 8 / 1024)}KB "
    #         f"for a {video_info.duration:.1f}s video at {effective_fps:.1f}fps"
    #     )
    
    # Define resolution/fps tiers with recommended minimum bitrates for H.265
    # Format: (width, height, min_bitrate_kbps, ideal_bitrate_kbps)
    resolution_tiers = [
        (1920, 1080, 1500, 4000),  # 1080p
        (1280, 720, 800, 2500),    # 720p
        (854, 480, 400, 1200),     # 480p
        (640, 360, 200, 700),      # 360p
        (426, 240, 100, 400),      # 240p
        (320, 180, 50, 200),       # 180p
    ]
    
    fps_tiers = [60, 30, 24, 15, 10]
    
    # Calculate original aspect ratio
    aspect_ratio = video_info.width / video_info.height
    
    # Find the best resolution that fits the bitrate budget
    best_width, best_height = resolution_tiers[-1][0], resolution_tiers[-1][1]
    
    for width, height, min_br, ideal_br in resolution_tiers:
        # Adjust dimensions to match source aspect ratio
        if aspect_ratio > width / height:
            # Source is wider, adjust height
            adjusted_height = int(width / aspect_ratio)
            adjusted_height = adjusted_height - (adjusted_height % 2)  # Make even
            adjusted_width = width
        else:
            # Source is taller, adjust width
            adjusted_width = int(height * aspect_ratio)
            adjusted_width = adjusted_width - (adjusted_width % 2)  # Make even
            adjusted_height = height
        
        # Don't upscale
        if adjusted_width > video_info.width or adjusted_height > video_info.height:
            adjusted_width = video_info.width - (video_info.width % 2)
            adjusted_height = video_info.height - (video_info.height % 2)
        
        # Scale bitrate requirements based on actual resolution vs reference
        pixel_ratio = (adjusted_width * adjusted_height) / (width * height)
        scaled_min_br = min_br * pixel_ratio
        
        if video_bitrate_kbps >= scaled_min_br:
            best_width, best_height = adjusted_width, adjusted_height
            break
    
    # Handle FPS: use fixed value if provided, otherwise calculate optimal
    if fixed_fps is not None:
        best_fps = fixed_fps
        # Warn if fixed FPS exceeds source (handled in main)
    else:
        # Find the best FPS that doesn't waste bitrate
        # Higher FPS needs more bitrate to look good
        best_fps = fps_tiers[-1]
        
        for fps in fps_tiers:
            if fps > video_info.fps:
                continue  # Don't increase FPS beyond source
            
            # Rough estimate: each fps tier needs ~20% more bitrate
            fps_factor = fps / 30.0
            effective_bitrate = video_bitrate_kbps / max(fps_factor, 0.5)
            
            # Calculate pixels per second
            pixels_per_sec = best_width * best_height * fps
            
            # Bits per pixel (higher is better quality)
            bpp = (video_bitrate_kbps * 1000) / pixels_per_sec
            
            # H.265 typically needs 0.03-0.1 bpp for decent quality
            if bpp >= 0.03:
                best_fps = fps
                break
    
    # Round bitrate to reasonable value
    video_bitrate_kbps = int(video_bitrate_kbps)
    
    return best_width, best_height, best_fps, video_bitrate_kbps


def encode_video(
    input_file: str,
    output_file: str,
    width: int,
    height: int,
    fps: float,
    video_bitrate_kbps: int,
    audio_bitrate_kbps: int = 16,
    two_pass: bool = True,
    target_bytes: Optional[int] = None,
    size_tolerance: float = 0.05
) -> bool:
    """
    Encode video with specified parameters using libx265 (CPU).
    
    If target_bytes is provided, will do an additional calibration pass
    to adjust for bitrate inaccuracy.
    
    Returns True if successful.
    """
    
    def run_encode(bitrate_kbps: int, output_path: str, pass_name: str = "Encoding") -> bool:
        """Run the actual encode with given bitrate."""
        bufsize_kbps = max(bitrate_kbps, 300)
        vf_string = f'scale={width}:{height}:flags=lanczos,fps={fps}'
        
        # Create temp directory for pass logs
        tmpdir = 'tmp/'
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)
        passlog = os.path.join(tmpdir, 'ffmpeg2pass')
        
        base_args = [
            './ffmpeg.exe',
            '-y',
            '-i', input_file,
            '-vf', vf_string,
            '-c:v', 'libx265',
            '-preset', 'slow',
            '-profile:v', 'main',
            '-pix_fmt', 'yuv420p',
            '-tag:v', 'hvc1',
            '-c:a', 'aac',
            '-ac', '1',
            '-ar', '22050',
            '-b:a', f'{audio_bitrate_kbps}k',
            '-movflags', '+faststart',
        ]
        
        if two_pass:
            # First pass (analysis)
            print(f"{pass_name}: Running analysis pass...")
            pass1_args = base_args + [
                '-b:v', f'{bitrate_kbps}k',
                '-maxrate', f'{int(bitrate_kbps * 1.5)}k',
                '-bufsize', f'{bufsize_kbps}k',
                '-x265-params', f'pass=1:stats={passlog}:aq-mode=3:aq-strength=1.0:psy-rd=2.0:psy-rdoq=1.0:rc-lookahead=32:bframes=4:b-adapt=2:subme=7:deblock=-1,-1',
                '-an',
                '-f', 'null',
                '/dev/null' if os.name != 'nt' else 'NUL'
            ]
            
            result = subprocess.run(pass1_args, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Analysis pass failed: {result.stderr}")
                return False
            
            # Second pass (encode)
            print(f"{pass_name}: Running encode pass...")
            pass2_args = base_args + [
                '-b:v', f'{bitrate_kbps}k',
                '-maxrate', f'{int(bitrate_kbps * 1.5)}k',
                '-bufsize', f'{bufsize_kbps}k',
                '-x265-params', f'pass=2:stats={passlog}:aq-mode=3:aq-strength=1.0:psy-rd=2.0:psy-rdoq=1.0:rc-lookahead=32:bframes=4:b-adapt=2:subme=7:deblock=-1,-1',
                output_path
            ]
            
            result = subprocess.run(pass2_args, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Encode pass failed: {result.stderr}")
                return False
        else:
            # Single pass
            print(f"{pass_name}...")
            single_args = base_args + [
                '-b:v', f'{bitrate_kbps}k',
                '-maxrate', f'{int(bitrate_kbps * 1.5)}k',
                '-bufsize', f'{bufsize_kbps}k',
                '-x265-params', 'aq-mode=3:aq-strength=1.0:psy-rd=2.0:psy-rdoq=1.0:rc-lookahead=32:bframes=4:b-adapt=2:subme=7:deblock=-1,-1',
                output_path
            ]
            
            result = subprocess.run(single_args, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Encoding failed: {result.stderr}")
                return False
        
        return True
    
    # If no target size specified, just do a normal encode
    if target_bytes is None:
        return run_encode(video_bitrate_kbps, output_file, "Encoding video")
    
    # Create temp directory for calibration encode
    tmpdir = 'tmp'
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    temp_output = os.path.join(tmpdir, 'calibration_encode.mp4')
    
    # First encode: calibration pass
    print("Pass 1/2: Calibration encode...")
    if not run_encode(video_bitrate_kbps, temp_output, "Calibration"):
        return False
    
    # Measure actual size
    actual_size = os.path.getsize(temp_output)
    size_ratio = target_bytes / actual_size
    size_diff = abs(1.0 - size_ratio)
    
    print(f"  Calibration result: {actual_size / 1024:.1f} KB (target: {target_bytes / 1024:.1f} KB)")
    print(f"  Ratio: {size_ratio:.3f} (diff: {size_diff * 100:.1f}%)")
    
    # If within tolerance, just move the temp file
    if size_diff <= size_tolerance:
        print(f"  Within {size_tolerance * 100:.0f}% tolerance, using calibration encode.")
        import shutil
        shutil.move(temp_output, output_file)
        return True
    
    # Calculate adjusted bitrate
    adjusted_bitrate = int(video_bitrate_kbps * size_ratio)
    print(f"  Adjusting bitrate: {video_bitrate_kbps} -> {adjusted_bitrate} kbps")
    
    # Final encode with adjusted bitrate
    print("Pass 2/2: Final encode with adjusted bitrate...")
    success = run_encode(adjusted_bitrate, output_file, "Final encode")
    
    # Clean up temp file
    if os.path.exists(temp_output):
        os.remove(temp_output)
    
    return success


def format_size(bytes_val: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} TB"


def main():
    parser = argparse.ArgumentParser(
        description='Optimize video encoding for a target file size',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s 10MB input.mp4 output.mp4
    %(prog)s 500KB input.mov output.mp4
    %(prog)s --single-pass 25MB input.mp4 output.mp4
    %(prog)s --fps 24 10MB input.mp4 output.mp4
        """
    )
    
    parser.add_argument(
        'target_size',
        help='Target file size (e.g., 10MB, 500KB, 1GB)'
    )
    parser.add_argument(
        'input_file',
        help='Input video file'
    )
    parser.add_argument(
        'output_file',
        help='Output video file'
    )
    parser.add_argument(
        '--single-pass',
        action='store_true',
        help='Use single-pass encoding (faster but less accurate file size)'
    )
    parser.add_argument(
        '--audio-bitrate',
        type=int,
        default=16,
        help='Audio bitrate in kbps (default: 16)'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=None,
        help='Fixed output FPS (default: auto-calculated for optimal quality)'
    )
    parser.add_argument(
        '--invert-aspect-ratio',
        action='store_true',
        default=False,
        help='Invert the aspect ratio of the output video (swap width and height) If your video comes out squashed, use this.'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Parse target size
    try:
        target_bytes = parse_size(args.target_size)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"Target file size: {format_size(target_bytes)}")
    
    # Get video info
    print(f"Analyzing input video: {args.input_file}")
    try:
        video_info = get_video_info(args.input_file, invert_aspect_ratio=args.invert_aspect_ratio)
    except Exception as e:
        print(f"Error analyzing video: {e}")
        sys.exit(1)
    
    print(f"  Duration: {video_info.duration:.2f}s")
    print(f"  Resolution: {video_info.width}x{video_info.height}")
    print(f"  FPS: {video_info.fps:.2f}")
    
    # Warn if fixed FPS exceeds source FPS
    if args.fps is not None and args.fps > video_info.fps:
        print(f"  ⚠️  Warning: Requested FPS ({args.fps}) exceeds source FPS ({video_info.fps:.2f})")
    
    # Calculate optimal parameters
    try:
        width, height, fps, video_bitrate = calculate_optimal_params(
            video_info,
            target_bytes,
            args.audio_bitrate,
            fixed_fps=args.fps
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"\nOptimal encoding parameters:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}" + (" (fixed)" if args.fps is not None else " (auto)"))
    print(f"  Video bitrate: {video_bitrate} kbps")
    print(f"  Audio bitrate: {args.audio_bitrate} kbps")
    print(f"  Encoder: libx265 (8-bit)")
    
    # Calculate expected file size
    expected_size = int((video_bitrate + args.audio_bitrate) * video_info.duration * 1000 / 8)
    print(f"  Expected size: ~{format_size(expected_size)}")
    
    # Encode video
    print(f"\nEncoding to: {args.output_file}")
    success = encode_video(
        args.input_file,
        args.output_file,
        width,
        height,
        fps,
        video_bitrate,
        args.audio_bitrate,
        two_pass=not args.single_pass,
        target_bytes=target_bytes
    )
    
    if not success:
        print("Encoding failed!")
        sys.exit(1)
    
    # Report results
    actual_size = os.path.getsize(args.output_file)
    print(f"\nEncoding complete!")
    print(f"  Target size: {format_size(target_bytes)}")
    print(f"  Actual size: {format_size(actual_size)}")
    print(f"  Difference: {(actual_size - target_bytes) / target_bytes * 100:+.1f}%")
    
if __name__ == '__main__':
    main()