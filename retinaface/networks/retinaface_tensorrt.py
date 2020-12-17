import tensorflow as tf

class RetinaFaceNetworkRT(object):
    """
    RetinaFace TensorRT Network. Can be applied to any input image size without having to be reloaded.
    """
    def __init__(self, saved_model_path):
        self.model = tf.saved_model.load(saved_model_path)