import pygame
import math
import numpy as np
import random   
class ContinuousSpace:
    def __init__(self, width: float, height: float, wall_size: float = 1.0):
        """Grid representation of the world."""
        self.width = width
        self.height = height
        self.wall_size = wall_size
        self.total_targets = 0
        self.collected_targets = 0
        self.obstacles = []
        self.objects = []
        self.agent = None
        self.plate_start_positions = []

        self.total_reward = 0.0
        self.target = None
        self.bot_radius = 0.1
        self.path = [] 
        self.inventory = 0
        self.inventory_capacity = float('inf')

        # Define object types FIRST
        self.objects_map = {
            "empty": 0,
            "boundary": 1,
            "obstacle": 2,
            "target": 3,
            "kitchen": 4,
            "occupied": 5,  
        }

        self.create_boundary_walls()

    def add_object(self, x, y, size, obj_type):
        if obj_type not in self.objects_map:
            raise ValueError(f"Unknown object type: {obj_type}")
        if obj_type == "target":
            self.total_targets += 1
            self.plate_start_positions.append((x, y, size))
        self.objects.append({
            "x": x,
            "y": y,
            "size": size,
            "type": self.objects_map[obj_type]
        })
        
    
    def place_agent(self, x: float, y: float, size: float):
        # if not self._is_inside_inner_area(x, y, size):
        #     raise ValueError("Agent out of bounds.")
        # if self._collides((x, y), size):
        #     raise ValueError("Agent collides with wall.")
        # self.agent = ((x, y), size)

        max_trials = 1000
        if x is None or y is None:
            for _ in range(max_trials):
                rx = random.uniform(self.wall_size, self.width  - self.wall_size - size)
                ry = random.uniform(self.wall_size, self.height - self.wall_size - size)
                if self._is_inside_inner_area(rx, ry, size) and not self._collides((rx, ry), size):
                    x, y = rx, ry
                    break
                else:
                    raise RuntimeError(
                        f"Could not find a legal spawn point after {max_trials} attempts."
                    )
            
        if not self._is_inside_inner_area(x, y, size):
            raise ValueError("Agent out of bounds.")
        if self._collides((x, y), size):
            raise ValueError("Agent collides with wall or object.")
        
        self.agent = ((x, y), size)

    def place_target(self, x: float, y: float, size: float):
        if not self._is_inside_inner_area(x, y, size):
            raise ValueError("Target out of bounds.")
        if self._collides((x, y), size):
            raise ValueError("Target collides with wall.")
        self.target = ((x, y), size)

    def create_boundary_walls(self):
        s = self.wall_size
        for i in range(math.ceil(self.width / s)):
            x = i * s
            self.add_object(x, 0.0, s, "boundary")
            self.add_object(x, self.height - s, s, "boundary")
        for j in range(math.ceil(self.height / s)):
            y = j * s
            self.add_object(0.0, y, s, "boundary")
            self.add_object(self.width - s, y, s, "boundary")

    def _is_inside_inner_area(self, x: float, y: float, size: float) -> bool:
        s = self.wall_size
        return (s <= x <= self.width - s - size) and (s <= y <= self.height - s - size)
    
    ## Function to collect targets, if the robot finds them
    def collect_target_if_reached(self, tolerance: float = 0.2):
        if self.agent is None:
            return

        (ax, ay), a_size = self.agent
        remaining_objects = []

        for obj in self.objects:
            if obj["type"] == self.objects_map["target"]:
                ox, oy = obj["x"], obj["y"]
                osize = obj["size"]
                if self._squares_overlap_with_margin(ax, ay, a_size, ox, oy, osize, margin=tolerance):
                    if self.inventory < self.inventory_capacity:
                        self.collected_targets += 1
                        self.inventory += 1
                        print(f"[INFO] Collected plate at ({ox:.1f}, {oy:.1f}) — carrying {self.inventory}")
                        continue  
            remaining_objects.append(obj)

        self.objects = remaining_objects
        
    def deliver_plates_if_at_kitchen(self):
        if self.agent is None or self.inventory == 0:
            return False

        (ax, ay), a_size = self.agent
        for obj in self.objects:
            if obj["type"] == self.objects_map["kitchen"]:  # kitchen
                ox, oy, osize = obj["x"], obj["y"], obj["size"]
                if self._squares_overlap_with_margin(ax, ay, a_size, ox, oy, osize, margin=0.2):
                    print(f"[INFO] Delivered {self.inventory} plate(s) to the kitchen.")
                    self.inventory = 0
                    return True
        return False
    

    def step_with_reward(self, direction: str, step_size: float = 1.0, sub_step: float = 0.2) -> float:
        """
        Move agent in a direction and compute reward.
        Returns:
            total_reward (float)
        """
        reward = 0.0
        collided = False
        hit_obstacle = False

        direction_map = {
            'up': (0, 1),
            'down': (0, -1),
            'left': (-1, 0),
            'right': (1, 0),
            'up_left': (-1, 1),
            'up_right': (1, 1),
            'down_left': (-1, -1),
            'down_right': (1, -1),
        }

        if direction not in direction_map:
            return -1.0 

        dx, dy = direction_map[direction]
        if dx != 0 and dy != 0:
            norm = math.sqrt(2)
            dx /= norm
            dy /= norm

        total_distance = step_size
        moved_distance = 0

        while moved_distance < total_distance:
            remaining = min(sub_step, total_distance - moved_distance)
            delta_x = dx * remaining
            delta_y = dy * remaining

            result = self._try_move_with_collision_type(delta_x, delta_y)
            if result == "wall":
                reward += -50
                collided = True
                break
            elif result == "obstacle":
                reward += -10
                collided = True
                break
            elif result == "ok":
                moved_distance += remaining
                reward += -1

            #self.detect_objects()

        
        if not collided:
            pre_count = self.collected_targets
            self.collect_target_if_reached()
            if self.collected_targets > pre_count:
                reward += 10

        if self.is_task_complete():
            self.deliver_plates_if_at_kitchen()
            reward += 1000

        if self.agent is not None:
            (ax, ay), a_size = self.agent
            for obj in self.objects:
                if obj["type"] == self.objects_map["occupied"]:
                    ox, oy, osize = obj["x"], obj["y"], obj["size"]
                    if self._squares_overlap_with_margin(ax, ay, a_size, ox, oy, osize):
                        reward += -5
                        break
        self.total_reward += reward
        return reward

    

    def get_state_vector(self):
        """Returns a fixed-length state vector representation."""
        if self.agent is None:
            raise ValueError("Agent not placed.")

        (ax, ay), _ = self.agent
        inventory = self.inventory

        
        nearest_plate = [0.0, 0.0]
        min_dist = float("inf")
        for obj in self.objects:
            if obj["type"] == self.objects_map["target"]:
                px, py = obj["x"], obj["y"]
                dist = math.hypot(ax - px, ay - py)
                if dist < min_dist:
                    nearest_plate = [px, py]
                    min_dist = dist

       
        kitchen_pos = [0.0, 0.0]
        for obj in self.objects:
            if obj["type"] == self.objects_map["kitchen"]:
                kitchen_pos = [obj["x"], obj["y"]]
                break

        
        state = [
            ax / self.width, ay / self.height,
            inventory / 10.0,  
            nearest_plate[0] / self.width, nearest_plate[1] / self.height,
            kitchen_pos[0] / self.width, kitchen_pos[1] / self.height
        ]
        return state
    

    def is_task_complete(self) -> bool:
        """Return True if all plates are collected AND agent is in the kitchen."""
        if self.collected_targets < self.total_targets:
            return False

        
        if self.agent is None:
            return False

        (ax, ay), a_size = self.agent
        for obj in self.objects:
            if obj["type"] == self.objects_map["kitchen"]:  
                ox, oy, osize = obj["x"], obj["y"], obj["size"]
                if self._squares_overlap_with_margin(ax, ay, a_size, ox, oy, osize, margin=0.2):
                    return True
        return False
    

    def _try_move_with_collision_type(self, dx: float, dy: float) -> str:
        if self.agent is None:
            raise ValueError("Agent not placed.")

        (x, y), size = self.agent
        new_x = x + dx
        new_y = y + dy

        if not self._is_inside_inner_area(new_x, new_y, size):
            return "wall"

        for obj in self.objects:
            ox, oy, osize, otype = obj["x"], obj["y"], obj["size"], obj["type"]
            if self._squares_overlap(new_x, new_y, size, ox, oy, osize):
                if otype == self.objects_map["obstacle"]:
                    return "obstacle"
                elif otype == self.objects_map["boundary"]:
                    return "wall"

        self.agent = ((new_x, new_y), size)
        self.path.append((new_x, new_y))  
        return "ok"

    def _collides(self, pos, size) -> bool:
        px, py = pos
        for obj in self.objects:
            ox, oy = obj["x"], obj["y"]
            osize = obj["size"]
            if self._squares_overlap(px, py, size, ox, oy, osize):
                return True
        return False

    @staticmethod
    def _squares_overlap(x1, y1, s1, x2, y2, s2) -> bool:
        return not (x1 + s1 <= x2 or x2 + s2 <= x1 or y1 + s1 <= y2 or y2 + s2 <= y1)
    
    @staticmethod
    def _squares_overlap_with_margin(x1, y1, s1, x2, y2, s2, margin=0.1) -> bool:
        """Check overlap with relaxed margin."""
        return not (
            x1 + s1 < x2 - margin or
            x2 + s2 < x1 - margin or
            y1 + s1 < y2 - margin or
            y2 + s2 < y1 - margin
        )
    
    ## Possible actions the robot can take
    def move_agent_direction(self, direction: str, step_size: float = 1.0, sub_step: float = 0.2):
        direction_map = {
            'up': (0, 1),
            'down': (0, -1),
            'left': (-1, 0),
            'right': (1, 0),
            'up_left': (-1, 1),
            'up_right': (1, 1),
            'down_left': (-1, -1),
            'down_right': (1, -1),
        }

        if direction not in direction_map:
            print("Invalid direction")
            return

        dx, dy = direction_map[direction]
        if dx != 0 and dy != 0:
            norm = math.sqrt(2)
            dx /= norm
            dy /= norm

        total_distance = step_size
        moved_distance = 0

        while moved_distance < total_distance:
            remaining = min(sub_step, total_distance - moved_distance)
            delta_x = dx * remaining
            delta_y = dy * remaining

            if not self._try_move_with_collision_type(delta_x, delta_y):
                break

            moved_distance += remaining

        
        self.collect_target_if_reached()

    def detect_objects(self):
        if self.agent is None:
            return

        (ax, ay), a_size = self.agent
        agent_center_x = ax + a_size / 2
        agent_center_y = ay + a_size / 2

        for obj in self.objects:
            ox, oy = obj["x"], obj["y"]
            osize = obj["size"]
            obj_center_x = ox + osize / 2
            obj_center_y = oy + osize / 2

            distance = math.hypot(agent_center_x - obj_center_x, agent_center_y - obj_center_y)

            # First check: is object inside the robot's radius
            if distance <= self.bot_radius:
                obj_type_str = [key for key, value in self.objects_map.items() if value == obj["type"]][0]

                if obj_type_str == "target":
                    self.detected_targets_counter += 1
                    print(f"[INFO] Target detected inside bot radius! Total detections: {self.detected_targets_counter}")

                    if self.detected_targets_counter >= self.max_detected_targets:
                        self.go_to_charger = True
                        print("[INFO] Max targets detected. Switching to charger mode.")
