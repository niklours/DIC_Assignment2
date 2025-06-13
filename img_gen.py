import numpy as np
import cv2  

import numpy as np
import cv2

def get_grid_image(self, resolution=256):
    """Returns a top-down RGB image (H, W, 3) as a NumPy array."""
    img = np.ones((resolution, resolution, 3), dtype=np.uint8) * 255

    def world_to_pixel(x, y):
        px = int(x / self.width * resolution)
        py = int((self.height - y) / self.height * resolution)
        return px, py

    def draw_square(x, y, size, color):
        px, py = world_to_pixel(x, y)
        ps = int(size / self.width * resolution)
        px1, py1 = max(0, px), max(0, py - ps)
        px2, py2 = min(resolution, px + ps), min(resolution, py)
        img[py1:py2, px1:px2] = color

    # Draw boundaries, obstacles, targets
    for obj in self.objects:
        color = {
            self.objects_map["boundary"]: (100, 100, 100),
            self.objects_map["obstacle"]: (139, 69, 19),
            self.objects_map["target"]: (255, 165, 0),
        }.get(obj["type"], (0, 0, 0))
        draw_square(obj["x"], obj["y"], obj["size"], color)

    # Draw plate start positions if any
    if hasattr(self, "plate_start_positions"):
        for x, y, size in self.plate_start_positions:
            draw_square(x, y, size, (255, 220, 180))

    # Draw agent as a circle
    if self.agent:
        x, y = self.agent[0]
        px, py = world_to_pixel(x, y)
        radius = int((self.agent[1] / self.width) * resolution / 2)
        radius = max(1, radius)  # minimum radius 1 pixel
        cv2.circle(img, (px, py), radius, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)

    if len(self.path) > 1:
        points = [world_to_pixel(x, y) for x, y in self.path]
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i + 1], color=(0, 191, 255), thickness=2, lineType=cv2.LINE_AA)

    return img


