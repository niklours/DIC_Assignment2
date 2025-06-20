import math
from collections import deque
class ContinuousSpace:
    def __init__(self, width: float, height: float, wall_size: float = 1.0):
        self.width = width
        self.height = height
        self.wall_size = wall_size
        self.plate_start_positions = []
        self.prev_positions = deque(maxlen=20)
        self.objects = []
        self.agent = None
        self.target = None
        self.bot_radius = 4.0 
        self.inventory = 0
        self.path = []

        self.objects_map = {
            "empty": 0,
            "boundary": 1,
            "obstacle": 2,
            "target": 3,
        }

        self.create_boundary_walls()

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

    def add_object(self, x, y, size, obj_type):
        if obj_type not in self.objects_map:
            raise ValueError(f"Unknown object type: {obj_type}")
        self.objects.append({"x": x, "y": y, "size": size, "type": self.objects_map[obj_type]})
        if obj_type == "target":
            self.target = (x, y)
            self.plate_start_positions.append((x, y, size))

    def place_agent(self, x: float, y: float, size: float):
        if not self._is_inside_inner_area(x, y, size):
            raise ValueError("Agent out of bounds.")
        if self._collides((x, y), size):
            raise ValueError("Agent collides with wall.")
        self.agent = ((x, y), size)
        self.path = [(x, y)]

    def _is_inside_inner_area(self, x: float, y: float, size: float) -> bool:
        s = self.wall_size
        return (s <= x <= self.width - s - size) and (s <= y <= self.height - s - size)

    def _squares_overlap(self, x1, y1, s1, x2, y2, s2) -> bool:
        margin = 0.05
        return not (
            x1 + s1 <= x2 + margin or x2 + s2 <= x1 + margin or
            y1 + s1 <= y2 + margin or y2 + s2 <= y1 + margin
        )

    def _collides(self, pos, size) -> bool:
        px, py = pos
        for obj in self.objects:
            if obj["type"] in (self.objects_map["boundary"], self.objects_map["obstacle"]):
                if self._squares_overlap(px, py, size, obj["x"], obj["y"], obj["size"]):
                    return True
        return False

    def move_agent_direction(self, action_idx: int,  step_size=0.5, sub_step=0.1):
        direction_map = {
            0: (0, 1),
            1: (1, 0),
            2: (0, -1),
            3: (-1, 0),
            4: (0.7, 0.7),
            5: (-0.7, 0.7),
            6: (0.7, -0.7),
            7: (-0.7, -0.7),
        }

        if action_idx not in direction_map:
            return False

        dx, dy = direction_map[action_idx]
        total_distance = step_size
        moved_distance = 0

        while moved_distance < total_distance:
            remaining = min(sub_step, total_distance - moved_distance)
            delta_x = dx * remaining
            delta_y = dy * remaining

            if not self._try_move(delta_x, delta_y):
                return False

            moved_distance += remaining

        return True

    def _try_move(self, dx: float, dy: float):
        if self.agent is None:
            raise ValueError("Agent not placed.")

        (x, y), size = self.agent
        new_x = x + dx
        new_y = y + dy

        if not self._is_inside_inner_area(new_x, new_y, size):
            return False
        if self._collides((new_x, new_y), size):
            return False

        self.agent = ((new_x, new_y), size)
        self.path.append((new_x, new_y))
        self.prev_positions.append((round(new_x, 1), round(new_y, 1)))
        return True

    def collect_target(self):
        if self.agent is None or self.target is None:
            return False

        (ax, ay), asize = self.agent
        tx, ty = self.target
        if self._squares_overlap(ax, ay, asize, tx, ty, 1.0):
            self.target = None
            self.inventory = 1
            self.objects = [o for o in self.objects if o["type"] != self.objects_map["target"]]
            return True
        return False

    def is_task_complete(self):
        return self.inventory == 1 and self.target is None

    def get_state_vector(self):
        if self.agent is None:
            raise ValueError("Agent not placed.")

        (ax, ay), _ = self.agent
        inv = self.inventory
        #sensor_radius = 4.0

        target_near, _ = self.target_sense(self.bot_radius)

       
        dx = dy = 0.0            
        norm_dist = 1.0 if target_near else 0.0   # 1 = sensor target, 0 = nothing

        # --- --------------------------------------

        # nearby = self.detect_objects(sensor_radius)
        # target_detected = None
        # for obj in nearby:
        #     if obj["type"] == self.objects_map["target"]:
        #         target_detected = obj
        #         break

        # if target_detected:
        #     tx, ty = target_detected["position"]
        #     dx = (tx - ax) / self.width
        #     dy = (ty - ay) / self.height
        #     dist = math.hypot(tx - ax, ty - ay)
        #     max_dist = math.hypot(self.width, self.height)
        #     norm_dist = dist / max_dist
        # else:
        #     dx = dy = 0.0
        #     norm_dist = 1.0  

        # dx = dy = 0.0
        # norm_dist = 1.0  
        near_obstacles = sum(  ## checking for possible obstacles
            1 for obj in self.objects
            if obj["type"] == self.objects_map["obstacle"]
            and math.hypot(ax - obj["x"], ay - obj["y"]) < 1.5
        )
        near_obstacles = min(near_obstacles, 5) / 5.0

        rounded_pos = (round(ax, 1), round(ay, 1))
        loop_count = self.prev_positions.count(rounded_pos)
        loop_signal = loop_count / len(self.prev_positions) if self.prev_positions else 0.0

        return [
            ax / self.width,
            ay / self.height,
            dx,
            dy,
            norm_dist,
            near_obstacles,
            loop_signal
        ]
    
    def target_sense(self, radius: float):
        """
        Returns a pair of booleans:
            (target_near, obstacle_near)
        """
        if self.agent is None:
            return False, False

        (ax, ay), _ = self.agent
        target_near = False
        obstacle_near = False

        for obj in self.objects:
            if math.hypot(ax - obj["x"], ay - obj["y"]) > radius:
                continue

            if obj["type"] == self.objects_map["target"]:
                target_near = True
            elif obj["type"] == self.objects_map["obstacle"]:
                obstacle_near = True

            if target_near and obstacle_near:
                break

        return target_near, obstacle_near

    
    
    ## Attempt for step_with_reward WITHOUT distance information
    def step_with_reward(self, action_idx, step_size=0.5, sub_step=0.1):
        if self.agent is None:
            return -10.0                            

        reward = -0.2                           

        if not self.move_agent_direction(action_idx, step_size, sub_step):
            reward -= 2.0                           

        if self.collect_target():
            reward += 20.0

        if self.is_task_complete():
            reward += 1000.0

        target_near, obstacle_near = self.target_sense(self.bot_radius) 
        if target_near: # possible target            
            reward += 5.0
        elif obstacle_near: # possible obstacle          
            reward -= 2.0
        elif self.target is not None: # wandering
            reward -= 1.0

        # ---------- loop penalty ----------------------------------------
        (nx, ny), _ = self.agent
        cell = (round(nx, 1), round(ny, 1))
        self.prev_positions.append(cell)
        loop_signal = self.prev_positions.count(cell) / len(self.prev_positions)
        if loop_signal > 0.1:
            reward -= loop_signal * 10.0

        return reward
    
    # ## Third try
    # def detect_objects(self, radius: float):
    #     """Return at most one {'type': …} for each kind detected."""
    #     if self.agent is None:
    #         return []

    #     (ax, ay), _ = self.agent
    #     seen: set[int] = set()
    #     nearby = []
    #     for obj in self.objects:
    #         if obj["type"] in seen:
    #             continue                 
    #         if math.hypot(ax - obj["x"], ay - obj["y"]) <= radius:
    #             nearby.append({"type": obj["type"]})
    #             seen.add(obj["type"])
    #             if len(seen) == 2:   
    #                 break
    #     return nearby

    ##Second try
    # def detect_objects(self, radius: float):
        
    #     if self.agent is None:
    #         return []

    #     (ax, ay), _ = self.agent
    #     nearby = []
    #     for obj in self.objects:
    #         if math.hypot(ax - obj["x"], ay - obj["y"]) <= radius:
    #             nearby.append({"type": obj["type"]})
    #     return nearby


    def add_rectangle_object(self, x1, y1, x2, y2, size: float, obj_type: str):
        """
        Adds a filled rectangular area of objects (e.g., obstacles) between (x1, y1) and (x2, y2).
        The area is filled with `size`-sized square blocks.
        
        Args:
            x1, y1: one corner of the rectangle
            x2, y2: opposite corner
            size: size of each square block
            obj_type: string type like "obstacle"
        """
        if obj_type not in self.objects_map:
            raise ValueError(f"Unknown object type: {obj_type}")

        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])

        x = x_min
        while x < x_max:
            y = y_min
            while y < y_max:
                self.add_object(x, y, size, obj_type)
                y += size
            x += size

# def step_with_reward(self, action_idx, step_size=0.5, sub_step=0.1):
    #     if self.agent is None:
    #         return -10.0  # Stronger penalty if agent not initialized

    #     (x, y), _ = self.agent
    #     reward = -0.2  # small time penalty

    #     if self.target:
    #         prev_dist = math.hypot(x - self.target[0], y - self.target[1])
    #     else:
    #         prev_dist = 0

    #     moved = self.move_agent_direction(action_idx, step_size, sub_step)

    #     if not moved:
    #         reward -= 2.0  

    #     if self.collect_target():
    #         reward += 20.0

    #     if self.is_task_complete():
    #         reward += 1000.0

    #     (nx, ny), _ = self.agent
    #     # if self.target:
    #     #     new_dist = math.hypot(nx - self.target[0], ny - self.target[1])
    #     #     delta = prev_dist - new_dist
    #     #     if delta > 0.01:
    #     #         reward += delta*5.0  # stronger reward
    #     #     else:
    #     #         reward -= 1.0

    #     # Proximity detection
    #     proximity_flag = False
    #     sensor_radius = 2.0
    #     self.bot_radius = 2.0
    #     nearby = self.detect_objects(self.bot_radius)
    #     for obj in nearby:
    #         if obj["type"] == self.objects_map["target"]:
    #             reward += 5.0
    #             proximity_flag = True
    #         elif obj["type"] == self.objects_map["obstacle"]:
    #             reward -= 2.0 

    #     if (self.target is not None) and (proximity_flag == False):
    #         reward -= 1.0


    #     # Loop penalty
    #     rounded_pos = (round(nx, 1), round(ny, 1))
    #     self.prev_positions.append(rounded_pos)
    #     if len(self.prev_positions) > 100:
    #         self.prev_positions.pop(0)

    #     loop_count = self.prev_positions.count(rounded_pos)
    #     loop_signal = loop_count / len(self.prev_positions)
    #     if loop_signal > 0.1:
    #         reward -= loop_signal * 10.0  # stronger loop penalty

    #     return reward