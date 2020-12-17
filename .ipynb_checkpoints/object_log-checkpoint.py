import time
from enum import Enum

import numpy as np


class ObjectBufferMode(Enum):
    Clearness = 1
    Center = 2
    Medoid = 3


class ObjectLog:

    """
    An ObjectLog Class used for storing the information of tracked object

    Attributes
    ----------
    object_id: object
        unique id of the object
    location_id: object
        unique id of the location where the object is captured
    image_size: tuple
        the size of image width, height, channel
    rep_vec_size: int
        the size/dimension of representation vector
    rep_vec: numpy
        representation vector of the object. Will be  used to calculate similarity
    start_time: float
        first time when object detected
    time_end: float
        last time when object detected
    buffer_size: int
        size of buffer for storing detection (images, rep_vec, clearness_level)
    buffer_mode: ObjectBufferMode
        mode for selecting which object that will be eliminated when the buffer is full
    """

    def __init__(
        self,
        object_id: object,
        location_id: object,
        image_size: tuple,
        rep_vec_size: int,
        image: np.array = None,
        rep_vec: np.array = None,
        time_start: float = None,
        time_end: float = None,
        buffer_size: int = 10,
        buffer_mode: ObjectBufferMode = ObjectBufferMode.Clearness,
    ):
        self.object_id = object_id
        self.location_id = location_id
        self.image_size = image_size
        self.rep_vec_size = rep_vec_size
        self.image = image
        self.rep_vec = rep_vec
        self.time_start = time_start
        self.time_end = time_end
        self.buffer_size = buffer_size
        self.buffer_mode = buffer_mode

        if self.time_start is None:
            self.time_start = time.time()

        assert np.ndim(self.image_size) != 3

        self.buffer_count = 0
        self.buffer_image = np.zeros(
            (
                self.buffer_size,
                self.image_size[0],
                self.image_size[1],
                self.image_size[2],
            )
        )
        self.buffer_rep_vec = np.zeros((self.buffer_size, self.rep_vec_size))

        if self.buffer_mode == ObjectBufferMode.Clearness:
            self.buffer_image_clearness = np.zeros(self.buffer_size)

    def sort_rep_center(self):
        center = np.mean(self.buffer_rep_vec, axis=0)
        duplicate_center = np.tile(center, (self.buffer_size, 1))
        l2_norm = np.linalg.norm(duplicate_center - self.buffer_rep_vec, axis=1)
        sorted_idx = np.argsort(l2_norm)

        return sorted_idx

    def sort_rep_medoid(self):
        center = np.mean(self.buffer_rep_vec, axis=0)
        duplicate_center = np.tile(center, (self.buffer_size, 1))
        l2_norm = np.linalg.norm(duplicate_center - self.buffer_rep_vec, axis=1)
        sorted_idx = np.argsort(l2_norm)

        # Calculate distance from medoid
        duplicate_medoid = np.tile(
            self.buffer_rep_vec[sorted_idx[0]], (self.buffer_size, 1)
        )
        l2_norm = np.linalg.norm(duplicate_medoid - self.buffer_rep_vec, axis=1)
        sorted_idx = np.argsort(l2_norm)

        return sorted_idx

    def append(
        self, image: np.array, rep_vec: np.array, clearness=None, custom_time=None
    ):
        """Append object log

        Parameters
        ----------
        image: numpy
            detected image
        rep_vec: numpy
            representation vector
        clearness: float
            clearness of image
        """
        self.buffer_image[self.buffer_count] = image
        self.buffer_rep_vec[self.buffer_count] = rep_vec

        if self.buffer_mode == ObjectBufferMode.Clearness:
            assert clearness is not None
            self.buffer_image_clearness[self.buffer_count] = clearness

        self.buffer_count += 1

        if custom_time is None:
            self.time_end = time.time()
        else:
            self.time_end = custom_time

        # Remove the selected buffer (based on mode) if the buffer is full
        if self.buffer_count >= self.buffer_size:
            if self.buffer_mode == ObjectBufferMode.Clearness:
                sorted_idx = np.argsort(self.buffer_image_clearness)[::-1]
            elif self.buffer_mode == ObjectBufferMode.Center:
                sorted_idx = self.sort_rep_center()
            else:
                sorted_idx = self.sort_rep_medoid()

            self.buffer_image = self.buffer_image[sorted_idx]
            self.buffer_rep_vec = self.buffer_rep_vec[sorted_idx]
            self.buffer_image_clearness = self.buffer_image_clearness[sorted_idx]
            self.buffer_count -= 1

    def get_buffer_image(self):
        return self.buffer_image[:self.buffer_count]

    def get_buffer_rep_vec(self):
        return self.buffer_rep_vec[:self.buffer_count]

    def get_buffer_image_clearness(self):
        return self.buffer_image_clearness[:self.buffer_count]

    def update_center(self):
        sorted_idx = self.sort_rep_center()
        self.image = self.buffer_image[sorted_idx[0]]
        self.rep_vec = self.buffer_rep_vec[sorted_idx[0]]

    def calculate_center_image(self):
        self.update_center()
        return self.image

    def calculate_center_emb(self):
        return np.mean(self.buffer_rep_vec, axis=0)

    def calculate_medoid_emb(self):
        self.update_center()
        return self.rep_vec

    def compare_center_emb(self, other_emb):
        center = np.mean(self.buffer_rep_vec, axis=0)
        l2_norm = np.linalg.norm(center - other_emb)
        return l2_norm

    def compare_medoid_emb(self, other_emb, update_center=True):
        if update_center:
            self.update_center()
        l2_norm = np.linalg.norm(self.rep_vec - other_emb)
        return l2_norm