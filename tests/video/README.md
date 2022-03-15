# Video Parser

```bash
python ./video_parse.py
```

You can change parameters (start_frame_id, max_frames) in script.
```python
streamer = Streamer(
    "./palace.h264.mp4", start_frame_id=200, max_frames=200, step=2
)
```
