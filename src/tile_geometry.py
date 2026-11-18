"""Geometry shared by fixed-grid training and full-frame inference."""

from PIL import ImageOps

TILES = {'A': (0, 0), 'B': (1, 0), 'C': (2, 0), 'D': (3, 0),
         'E': (0, 1), 'F': (1, 1), 'G': (2, 1), 'H': (3, 1),
         'I': (0, 2), 'L': (1, 2), 'M': (2, 2), 'N': (3, 2)}
ORDER = tuple(TILES)
MIRRORED_ORDER = tuple(ORDER[row * 4 + 3 - col] for col, row in TILES.values())


def transform_frame(image, mirror=False):
    """Mirror the whole frame and return source tiles for transformed labels."""
    if image.size != (2048, 1536):
        raise ValueError(f'Expected 2048x1536 frame, got {image.size}')
    if mirror:
        return ImageOps.mirror(image), MIRRORED_ORDER
    return image, ORDER
