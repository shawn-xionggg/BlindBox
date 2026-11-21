"""Optional glove segmentation adapter. Stored blob size is AREA, not sqrt(area)."""
import cv2
import numpy as np
from looming_ttc import ColorModel, find_blobs, PRESETS


class GloveSegmenter:
    def __init__(self, preset=None, min_area=100):
        self.color = ColorModel(PRESETS.get(preset))
        self.hsv = None
        self.min_area = min_area
        self.bound = False

    def __call__(self, frame):
        self.hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self.color.mask(self.hsv)
        out = np.full((2, 3), np.nan)
        if mask is not None:
            for i, (center, size) in enumerate(find_blobs(mask, self.min_area)):
                out[i] = (*center, size*size)
        return out

    def bind(self, window):
        # imshow creates the window; avoid requiring GUI setup for headless tests.
        if self.bound:
            return
        try:
            cv2.setMouseCallback(window, self.click)
            self.bound = True
        except cv2.error:
            pass

    def click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.hsv is not None:
            self.color.sample(self.hsv, x, y)
