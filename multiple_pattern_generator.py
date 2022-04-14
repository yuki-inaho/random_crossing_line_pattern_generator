import cv2
import numpy as np
import itertools
from typing import Tuple
from pathlib import Path
from src.image import save_image
from src.utils import convert_mm_to_pixel

NUMBER_OF_LANDMARKS = 100
NUMBER_OF_SEGMENTS = 100

NUMBER_OF_SHEETS = 10
OUTPUT_DIR = "outputs"

PAPER_SIZE_HORISONTAL = 297  # [mm]
PAPER_SIZE_VERTICAL = 210  # [mm]

IMAGE_WIDTH = convert_mm_to_pixel(PAPER_SIZE_HORISONTAL)
IMAGE_HEIGHT = convert_mm_to_pixel(PAPER_SIZE_VERTICAL)


def calculate_slope(x0: float, y0: float, x1: float, y1: float, epsilon: float = 10e-8) -> float:
    return (y1 - y0) / (x1 - x0 + epsilon)


def draw_full_line(image: np.ndarray, point_1: Tuple[int], point_2: Tuple[int], color=(0, 0, 0), thickness=2):
    assert (len(image.shape) == 3) and (image.shape[2] == 3)
    image_with_line = image.copy()

    image_height, image_width, _ = image.shape
    point_end_upper = np.asarray([0, 0], dtype=np.int16)
    point_end_lower = np.asarray([image_width, image_height], dtype=np.int16)

    slope = calculate_slope(point_1[0], point_1[1], point_2[0], point_2[1])
    point_end_upper[1] = -(point_1[0] - point_end_upper[0]) * slope + point_1[1]
    point_end_lower[1] = -(point_2[0] - point_end_lower[0]) * slope + point_2[1]

    cv2.line(
        image_with_line, (point_end_upper), point_end_lower, color=color, thickness=thickness, lineType=cv2.LINE_AA
    )
    return image_with_line


def generate_pattarn(save_path):
    canvas = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    canvas[:] = 255

    landmarks_xy = np.random.rand(NUMBER_OF_LANDMARKS * 2).reshape(-1, 2)
    landmarks_xy[:, 0] = landmarks_xy[:, 0] * IMAGE_WIDTH
    landmarks_xy[:, 1] = landmarks_xy[:, 1] * IMAGE_HEIGHT
    landmarks_xy = landmarks_xy.astype(np.int16)

    landmark_pair_candidates = np.asarray(list(itertools.combinations(np.arange(NUMBER_OF_LANDMARKS), 2)))
    landmark_pairs = landmark_pair_candidates.take(
        np.random.choice(landmark_pair_candidates.shape[0], NUMBER_OF_SEGMENTS, replace=False), axis=0
    )

    for i in range(NUMBER_OF_SEGMENTS):
        canvas = draw_full_line(
            canvas,
            landmarks_xy[landmark_pairs[i, 0], :],
            landmarks_xy[landmark_pairs[i, 1], :],
            color=(0, 0, 0),
            thickness=3,
        )
    save_image(save_path, canvas)


output_dir_pathlib = Path(OUTPUT_DIR)
for i in range(NUMBER_OF_SHEETS):
    output_image_path = str(output_dir_pathlib.joinpath(f"{i}.png"))
    generate_pattarn(output_image_path)
