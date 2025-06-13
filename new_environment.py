import math

class ContinuousSpace:
    def __init__(self, width: float, height: float, wall_size: float = 1.0):
        self.width = width
        self.height = height
        self.wall_size = wall_size
        self.plate_start_positions = []

        self.objects = []
        self.agent = None
        self.target = None
        self.bot_radius = 0.1
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

    def move_agent_direction(self, action_idx: int, step_size: float = 1.0, sub_step: float = 0.2):
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
        """
        Dummy get_state_vector method that returns a vector representation of the environment state.
        """
        if self.agent is None:
            raise ValueError("Agent not placed.")

        (ax, ay), _ = self.agent
        inv = self.inventory
        tx, ty = (0.0, 0.0)
        dist = 0.0
        if self.target:
            tx, ty = self.target
            dist = math.hypot(ax - tx, ay - ty)
        max_dist = math.hypot(self.width, self.height)

        return [
            ax / self.width, ay / self.height,
            inv,
            tx / self.width, ty / self.height,
            dist / max_dist
        ]

    def step_with_reward(self, action, step_size=1.0, sub_step=0.2):
        """
        Dummy reward function that returns a reward based on the agent's movement and task completion.
        """
        if self.agent is None:
            return -1.0

        (x, y), _ = self.agent
        reward = -1

        if self.target:
            prev_dist = math.hypot(x - self.target[0], y - self.target[1])
        else:
            prev_dist = 0

        moved = self.move_agent_direction(action, step_size, sub_step)

        if not moved:
            reward -= 5

        if self.is_task_complete():
            reward += 10000

        (nx, ny), _ = self.agent
        if self.target:
            new_dist = math.hypot(nx - self.target[0], ny - self.target[1])
            shaped = prev_dist - new_dist
            if shaped > 0.05:
                reward += shaped * 5

        return reward

    def detect_objects(self, radius: float):
        if self.agent is None:
            return []

        (ax, ay), a_size = self.agent
        cx = ax + a_size / 2
        cy = ay + a_size / 2

        nearby = []
        for obj in self.objects:
            ox, oy = obj["x"], obj["y"]
            osize = obj["size"]
            ocx = ox + osize / 2
            ocy = oy + osize / 2
            distance = math.hypot(cx - ocx, cy - ocy)
            if distance <= radius:
                nearby.append({
                    "position": (ox, oy),
                    "size": osize,
                    "type": obj["type"]
                })
        return nearby

    def add_rectangle_object(self, x1, y1, x2, y2, size: float, obj_type: str):
        """
        Adds a filled rectangular area of objects (e.g., obstacles) between two points (inclusive).
        The area is filled with `size`-sized blocks.
        
        Args:
            x1, y1: first corner of the rectangle
            x2, y2: opposite corner
            size: size of each square block
            obj_type: string type like "obstacle"
        """
        if obj_type not in self.objects_map:
            raise ValueError(f"Unknown object type: {obj_type}")

        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])

        x = x_min
        while x <= x_max - size / 2:
            y = y_min
            while y <= y_max - size / 2:
                self.add_object(x, y, size, obj_type)
                y += size
            x += size

