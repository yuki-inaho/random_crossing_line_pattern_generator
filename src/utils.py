import numpy as np


def convert_mm_to_pixel(value_mm, dpi=300):
    return int(np.ceil(value_mm / 25.4 * dpi))  # value [mm] / 25.4 [mm/inch] * dpi [pixel/inch]
