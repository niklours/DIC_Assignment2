import math
from collections import deque
class ContinuousSpace:
    """
    Initialize the environment.

    Params:
        width (float): Width of the environment.
        height (float): Height of the environment.
        wall_size (float): Size of the boundary walls.

    Returns:
        None
    """
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
        """
        Create boundary walls around the edges of the environment.
        """
        s = self.wall_size
        for i in range(math.ceil(self.width / s)):
            x = i * s
            self.add_object(x, 0.0, s, "boundary")
            self.add_object(x, self.height - s, s, "boundary")
        for j in range(math.ceil(self.height / s)):
            y = j * s
            self.add_object(0.0, y, s, "boundary")
            self.add_object(self.width - s, y, s, "boundary")


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

    def add_object(self, x, y, size, obj_type):
        """
        Add a single object to the environment.

        Args:
            x (float): X-coordinate of the object.
            y (float): Y-coordinate of the object.
            size (float): Size of the object.
            obj_type (str): Type of the object.

        Returns:
            None
        """
        if obj_type not in self.objects_map:
            raise ValueError(f"Unknown object type: {obj_type}")
        self.objects.append({"x": x, "y": y, "size": size, "type": self.objects_map[obj_type]})
        if obj_type == "target":
            self.target = (x, y)
            self.plate_start_positions.append((x, y, size))

    def place_agent(self, x: float, y: float, size: float):
        """
        Place the agent at a specific position in the environment.

        Args:
            x (float): X-coordinate.
            y (float): Y-coordinate.
            size (float): Size of the agent.

        Returns:
            None
        """
        if not self._is_inside_inner_area(x, y, size):
            raise ValueError("Agent out of bounds.")
        if self._collides((x, y), size):
            raise ValueError("Agent collides with wall.")
        self.agent = ((x, y), size)
        self.path = [(x, y)]

    def _is_inside_inner_area(self, x: float, y: float, size: float) -> bool:
        """
        Check if the given area is within the inner bounds of the environment.

        Args:
            x (float): X-coordinate.
            y (float): Y-coordinate.
            size (float): Size of the square.

        Returns:
            bool: True if the area is inside the inner bounds, False otherwise.
        """
        s = self.wall_size
        return (s <= x <= self.width - s - size) and (s <= y <= self.height - s - size)

    def _squares_overlap(self, x1, y1, s1, x2, y2, s2) -> bool:
        """
        Check whether two squares overlap.

        Args:
            x1, y1 (float): Coordinates of first square.
            s1 (float): Size of first square.
            x2, y2 (float): Coordinates of second square.
            s2 (float): Size of second square.

        Returns:
            bool: True if the squares overlap, False otherwise.
        """
        margin = 0.05
        return not (
            x1 + s1 <= x2 + margin or x2 + s2 <= x1 + margin or
            y1 + s1 <= y2 + margin or y2 + s2 <= y1 + margin
        )

    def _collides(self, pos, size) -> bool:
        """
        Determine whether a square at the given position collides with any boundaries or obstacles.

        Args:
            pos (tuple): Position (x, y).
            size (float): Size of the square.

        Returns:
            bool: True if collision occurs, False otherwise.
        """
        px, py = pos
        for obj in self.objects:
            if obj["type"] in (self.objects_map["boundary"], self.objects_map["obstacle"]):
                if self._squares_overlap(px, py, size, obj["x"], obj["y"], obj["size"]):
                    return True
        return False

    def move_agent_direction(self, action_idx: int,  step_size=0.5, sub_step=0.1):
        """
        Move the agent in the direction specified by the action index.

        Args:
            action_idx (int): Direction index (0-7).
            step_size (float): Distance to move in total.
            sub_step (float): Distance per incremental step.

        Returns:
            bool: True if the agent moved successfully, False otherwise.
        """
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
        """
        Attempt to move the agent by a delta.

        Args:
            dx (float): Change in x position.
            dy (float): Change in y position.

        Returns:
            bool: True if the move is successful, False otherwise.
        """
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
        """
        Check if the agent overlaps with the target and collect it if true.

        Args:
            None

        Returns:
            bool: True if the target is collected, False otherwise.
        """
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
        """
        Check whether the task is complete.

        Args:
            None

        Returns:
            bool: True if task is complete, False otherwise.
        """
        return self.inventory == 1 and self.target is None

    def get_state_vector(self):
        """
        Get the normalized state vector of the agent.

        Args:
            None

        Returns:
            list: State vector containing normalized position, target/obstacle signal, and loop signal.
        """
        if self.agent is None:
            raise ValueError("Agent not placed.")

        (ax, ay), _ = self.agent
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
            near_obstacles,
            loop_signal
        ]
    
    # The agent learns a target or an obstacle is probably near it without knowing it's exact distance from it
    def target_sense(self, radius: float):
        """
        Detect whether a target or obstacle is within a sensing radius.

        Args:
            radius (float): Sensing radius.

        Returns:
            tuple: (target_near (bool), obstacle_near (bool))
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
                target_near = True # it only provides info if the target is inside the radius, not its position
            elif obj["type"] == self.objects_map["obstacle"]:
                obstacle_near = True # it only provides info if the obstacle is inside the radius, not its position

            if target_near and obstacle_near:
                break

        return target_near, obstacle_near

    
    ## Attempt for step_with_reward WITHOUT distance information
    def step_with_reward(self, action_idx, step_size=0.5, sub_step=0.1):
        """
        Setting the reward function of the environment.

        Args:
            action_idx (int): Index of action to take.
            step_size (float): Total step size.
            sub_step (float): Incremental sub-step size.

        Returns:
            float: Computed reward for each action.
        """
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
        elif self.target is not None: # wandering, no purpose
            reward -= 1.0

        # ---------- loop penalty ----------------------------------------
        (nx, ny), _ = self.agent
        cell = (round(nx, 1), round(ny, 1))
        self.prev_positions.append(cell)
        loop_signal = self.prev_positions.count(cell) / len(self.prev_positions)
        if loop_signal > 0.1:
            reward -= loop_signal * 10.0

        return reward
  
