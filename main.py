from face_tracker_logger.face_tracker import FaceTracker

if __name__ == "__main__":
    face_tracker = FaceTracker()
    face_tracker.real_time_camera(
        'Front', "rtsp-ip", 2)
