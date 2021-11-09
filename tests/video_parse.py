from kkannotation.streamer import Streamer


if __name__ == "__main__":
    streamer = Streamer(
        "./palace.h264.mp4", start_frame_id=200, max_frames=200, step=2
    )
    streamer.play()
    streamer.save_images("./output_video", exist_ok=True, remake=False)