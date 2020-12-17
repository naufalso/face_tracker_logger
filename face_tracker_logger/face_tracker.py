import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# from skimage.transform import resize

from .deep_sort import nn_matching
from .deep_sort.detection import Detection
from .deep_sort.tracker import Tracker
from .object_log import ObjectLog
from .preprocessing import ImagePreprocessing
from .retinaface.retinaface import RetinaFace

import multiprocessing as mp


class Camera:
    def __init__(self, rtsp_url):
        # load pipe for data transmittion to the process
        self.parent_conn, child_conn = mp.Pipe()
        # load process
        self.p = mp.Process(target=self.update, args=(child_conn, rtsp_url))
        # start process
        self.p.daemon = True
        self.p.start()

    def end(self):
        # send closure request to process

        self.parent_conn.send(2)

    def update(self, conn, rtsp_url):
        # load cam into seperate process

        print("Cam Loading...")
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        print("Cam Loaded...")
        run = True

        while run:

            # grab frames from the buffer
            cap.grab()

            # recieve input data
            rec_dat = conn.recv()

            if rec_dat == 1:
                # if frame requested
                ret, frame = cap.read()
                conn.send(frame)

            elif rec_dat == 2:
                # if close requested
                cap.release()
                run = False

        print("Camera Connection Closed")
        conn.close()

    def get_frame(self, resize=None):
        ###used to grab frames from the cam connection process

        ##[resize] param : % of size reduction or increase i.e 0.65 for 35% reduction  or 1.5 for a 50% increase

        # send request
        self.parent_conn.send(1)
        frame = self.parent_conn.recv()

        # reset request
        self.parent_conn.send(0)

        # resize if needed
        if resize == None:
            return frame
        else:
            return self.rescale_frame(frame, resize)

    def rescale_frame(self, frame, percent=65):

        return cv2.resize(frame, None, fx=percent, fy=percent)


class FaceTracker:
    def __init__(
        self,
        face_detector_threshold=0.985,
        max_euclidean_distance=0.75,
        max_age=30,
        n_init=10,
        buffer_size=10,
    ):
        self.face_detector_threshold = face_detector_threshold
        self.max_euclidean_distance = max_euclidean_distance
        self.max_age = max_age
        self.n_init = n_init
        self.buffer_size = buffer_size

        print(os.getcwd())
        print('Loading RetinaFace Model')
        self.face_detector = RetinaFace(
            os.getcwd()
            + "/face_tracker_logger/model/retinaface-tensorrt",
            False,
            0.4,
        )

        print('Loading FaceNet Model')
        self.face_recognization = tf.saved_model.load(
            os.getcwd() + "/face_tracker_logger/model/facenet-tensorrt/"
        )
        self.face_recognization_size = (160, 160, 3)

        nn_budget = None
        metric = nn_matching.NearestNeighborDistanceMetric(
            "euclidean", self.max_euclidean_distance, nn_budget
        )
        self.tracker = Tracker(metric, max_age=self.max_age, n_init=self.n_init)
        self.image_preprocessing = ImagePreprocessing()

    def reset_tracker(self, max_euclidean_distance=None, max_age=None,
            n_init=None):
        if max_euclidean_distance is None:
            max_euclidean_distance = self.max_euclidean_distance
        if max_age is None:
            max_age = self.max_age
        if n_init is None:
            n_init = self.n_init
        nn_budget = None
        metric = nn_matching.NearestNeighborDistanceMetric(
            "euclidean", max_euclidean_distance, nn_budget
        )
        self.tracker = Tracker(metric, max_age=max_age, n_init=n_init)

    def draw_box_extract_face(
        self,
        img,
        faces,
        landmarks,
        margin=10,
        draw_box=True,
        show_landmark=True,
        calculate_emb=True,
    ):
        cropped_faces = np.zeros(
            (
                faces.shape[0],
                self.face_recognization_size[0],
                self.face_recognization_size[1],
                self.face_recognization_size[2],
            )
        )
        ori_image = np.copy(img)

        for i in range(faces.shape[0]):
            box = faces[i].astype(np.int)
            color = (255, 0, 0)
            cropped_face = cv2.resize(
                ori_image[box[1] : box[3], box[0] : box[2], :],
                (160, 160),
                interpolation=cv2.INTER_AREA,
            )
            cropped_faces[i] = cropped_face
            if draw_box:
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            if show_landmark:
                if landmarks is not None:
                    landmark5 = landmarks[i].astype(np.int)
                    for l in range(landmark5.shape[0]):
                        color = (0, 255, 255)
                        if l == 0 or l == 3:
                            color = (0, 255, 0)
                        cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 1)
        embs = None
        if calculate_emb:
            embs = self.image_preprocessing.calc_embs(
                self.face_recognization, cropped_faces, 0, faces.shape[0]
            )

        return img, cropped_faces, embs

    def convert_box_and_pred(self, box):
        box[:, 2] = np.abs(box[:, 0] - box[:, 2])
        box[:, 3] = np.abs(box[:, 1] - box[:, 3])
        return box[:, :4], box[:, -1]

    def real_time_camera(self, location_id, input_video, show_display=1):
        if show_display == 1:
            from IPython import display

        object_log = {}
        id_count = 0

        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
        cam = Camera(input_video)

        print(f"Camera is alive?: {cam.p.is_alive()}")

        while cam.p.is_alive():
            start = time.time()

            frame = cam.get_frame()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces_boxes, landmarks = self.face_detector.detect(
                frame, self.face_detector_threshold
            )
            detections = []

            if faces_boxes.shape[0] > 0:
                frame, cropped_faces, embs = self.draw_box_extract_face(
                    frame,
                    faces_boxes,
                    landmarks,
                    margin=10,
                    draw_box=False,
                    show_landmark=True,
                    calculate_emb=True,
                )

                converted_boxes, scores = self.convert_box_and_pred(faces_boxes)
                names = [
                    {"face": face, "emb": emb} for face, emb in zip(cropped_faces, embs)
                ]

                detections = [
                    Detection(bbox, score, feature, name)
                    for bbox, score, feature, name in zip(
                        converted_boxes, scores, embs, names
                    )
                ]

                # Call the tracker
                self.tracker.predict()
                self.tracker.update(detections)

                for track in self.tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()

                    if track.track_id not in object_log:
                        object_log[track.track_id] = ObjectLog(
                            object_id=id_count,
                            location_id=location_id,
                            image_size=self.face_recognization_size,
                            rep_vec_size=track.name["emb"].shape[-1],
                            image=track.name["face"],
                            rep_vec=track.name["emb"],
                            time_start=time.time(),
                        )
                        id_count += 1

                    object_log[track.track_id].append(
                        track.name["face"],
                        track.name["emb"],
                        self.image_preprocessing.calculate_clearness(
                            track.name["face"]
                        ),
                    )

                    color = colors[int(track.track_id) % len(colors)]
                    color = [i * 255 for i in color]
                    cv2.rectangle(
                        frame,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])),
                        color,
                        2,
                    )
                    cv2.rectangle(
                        frame,
                        (int(bbox[0]), int(bbox[1] - 30)),
                        (
                            int(bbox[0]) + (len(str(track.track_id))) * 17,
                            int(bbox[1]),
                        ),
                        color,
                        -1,
                    )
                    cv2.putText(
                        frame,
                        str(object_log[track.track_id].object_id + 1),
                        (int(bbox[0]), int(bbox[1] - 10)),
                        0,
                        0.75,
                        (255, 255, 255),
                        2,
                    )

            end = time.time()

            cv2.rectangle(
                frame,
                (0, 0),
                (325, 100),
                (0, 0, 0),
                -1,
            )

            frame = cv2.putText(
                frame,
                f"Elapsed time: {(end - start):.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                1,
            )

            frame = cv2.putText(
                frame,
                f"Detections: {len(detections)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                1,
            )

            frame = cv2.putText(
                frame,
                f"Tracker Ids: {id_count}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                1,
            )

            if show_display == 1:
                plt.figure(figsize=(20, 10))
                plt.imshow(frame, aspect="auto")
                plt.xticks([])
                plt.yticks([])
            elif show_display == 2:
                cv2.imshow(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if show_display == 1:
                display.clear_output(wait=True)
                try:
                    plt.pause(0.01)
                except Exception as e:
                    print("Error", e)
                    pass
        


    def video_tracking(
        self, location_id, input_video, output_video=None, show_display=1
    ):
        if show_display == 1:
            from IPython import display
        # TODO use blureness level for delete the buffer
        object_log = {}
        id_count = 0

        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        vid = cv2.VideoCapture(input_video)
#         vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if output_video is not None:
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(output_video, codec, fps, (width, height))

        if vid.isOpened():
            is_capturing = True
        else:
            is_capturing = False

        while is_capturing:
            start = time.time()
            is_capturing, frame = vid.read()
            if not is_capturing:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            faces_boxes, landmarks = self.face_detector.detect(
                frame, self.face_detector_threshold
            )
            detections = []

            if faces_boxes.shape[0] > 0:
                frame, cropped_faces, embs = self.draw_box_extract_face(
                    frame,
                    faces_boxes,
                    landmarks,
                    margin=10,
                    draw_box=False,
                    show_landmark=True,
                    calculate_emb=True,
                )

                converted_boxes, scores = self.convert_box_and_pred(faces_boxes)
                names = [
                    {"face": face, "emb": emb} for face, emb in zip(cropped_faces, embs)
                ]

                detections = [
                    Detection(bbox, score, feature, name)
                    for bbox, score, feature, name in zip(
                        converted_boxes, scores, embs, names
                    )
                ]

                # Call the tracker
                self.tracker.predict()
                self.tracker.update(detections)

                for track in self.tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()

                    if track.track_id not in object_log:
                        object_log[track.track_id] = ObjectLog(
                            object_id=id_count,
                            location_id=location_id,
                            image_size=self.face_recognization_size,
                            rep_vec_size=track.name["emb"].shape[-1],
                            image=track.name["face"],
                            rep_vec=track.name["emb"],
                            time_start=time.time(),
                        )
                        id_count += 1

                    object_log[track.track_id].append(
                        track.name["face"],
                        track.name["emb"],
                        self.image_preprocessing.calculate_clearness(
                            track.name["face"]
                        ),
                    )

                    color = colors[int(track.track_id) % len(colors)]
                    color = [i * 255 for i in color]
                    cv2.rectangle(
                        frame,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])),
                        color,
                        2,
                    )
                    cv2.rectangle(
                        frame,
                        (int(bbox[0]), int(bbox[1] - 30)),
                        (
                            int(bbox[0]) + (len(str(track.track_id))) * 17,
                            int(bbox[1]),
                        ),
                        color,
                        -1,
                    )
                    cv2.putText(
                        frame,
                        str(object_log[track.track_id].object_id + 1),
                        (int(bbox[0]), int(bbox[1] - 10)),
                        0,
                        0.75,
                        (255, 255, 255),
                        2,
                    )

            end = time.time()

            cv2.rectangle(
                frame,
                (0, 0),
                (325, 100),
                (0, 0, 0),
                -1,
            )

            frame = cv2.putText(
                frame,
                f"Elapsed time: {(end - start):.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                1,
            )

            frame = cv2.putText(
                frame,
                f"Detections: {len(detections)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                1,
            )

            frame = cv2.putText(
                frame,
                f"Tracker Ids: {id_count}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                1,
            )

            if show_display:
                plt.figure(figsize=(20, 10))
                plt.imshow(frame, aspect="auto")
                plt.xticks([])
                plt.yticks([])

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if output_video is not None:
                out.write(frame)

            if show_display == 1:
                display.clear_output(wait=True)
                try:
                    plt.pause(0.01)
                except Exception as e:
                    print("Error", e)
                    pass
                
            elif show_display == 2:
                cv2.imshow(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

        return object_log
