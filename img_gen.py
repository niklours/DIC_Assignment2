import numpy as np
import cv2

def get_grid_image(self, resolution=64):
    img = np.ones((resolution, resolution, 3), dtype=np.uint8) * 255

    def world_to_pixel(x, y):
        px = int(x / self.width * resolution)
        py = int((self.height - y) / self.height * resolution)
        return px, py

    def draw_rect_from_bottom_left(x, y, size, color):
        px1, py1 = world_to_pixel(x, y)
        px2, py2 = world_to_pixel(x + size, y + size)
        cv2.rectangle(img, (px1, py2), (px2, py1), color, thickness=-1)

    if hasattr(self, "plate_start_positions"):
        for x, y, size in self.plate_start_positions:
            draw_rect_from_bottom_left(x, y, size, (255, 220, 180)) 

    for obj in self.objects:
        color = {
            self.objects_map["boundary"]: (100, 100, 100),    
            self.objects_map["obstacle"]: (139, 69, 19),      
            self.objects_map["target"]: (255, 165, 0),        
        }.get(obj["type"], (0, 0, 0))
        draw_rect_from_bottom_left(obj["x"], obj["y"], obj["size"], color)

    if len(self.path) > 1:
        points = [world_to_pixel(x, y) for x, y in self.path]
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i + 1], color=(200, 230, 255), thickness=1, lineType=cv2.LINE_AA)

    if self.agent:
        ax, ay = self.agent[0]
        cx, cy = world_to_pixel(ax + self.agent[1] / 2, ay + self.agent[1] / 2)
        radius = max(2, int(self.agent[1] / self.width * resolution / 2))
        cv2.circle(img, (cx, cy), radius, (0, 0, 255), -1)

    return img


