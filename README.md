This application must be run in the command line/terminal, there is no UI.

Arguments are:

```text
video_compressor.exe:
    OPTIONAL ARGUMENTS:
        -h, --help                          show this help message and exit
        --single-pass                       Use single pass encoding (faster, not recommended. File size might be less optimized)
        --audio-bitrate AUDIO_BITRATE       Set audio bitrate in kbps (default: 16)
        --fps FPS                           Clamp output frames per second (default: source FPS)
        --invert-aspect-ratio               Invert the aspect ratio / height and width of the output video. If your video comes out squashed, use this.

    MANDATORY ARGUMENTS:
        target_size           Target file size with unit (e.g., 10MB, 500KB, 1GB)
        input_file            Path to input video file
        output_file           Path for output video file
```

Example usage:

```text
./compress_video_cpu.exe 640KB tank.mp4 output.mp4 --fps 30
./compress_video_cpu.exe 640KB input.mov output.mp4 --audio-bitrate 32 --invert-aspect-ratio
```

GPU processing:
```text
GPU support is available in `compress_video_gpu.exe` with the same arguments. Make sure your GPU supports NVENC. (Nvidia GPU)
```