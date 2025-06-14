import pygame
import random
from img_gen import get_grid_image
from new_environment import ContinuousSpace


COLOR_MAP = {
    1: (100, 100, 100),  # boundary (gray)
    2: (139, 69, 19),    # obstacle (brown)
    3: (255, 165, 0),    # plate target (orange)
    4: (0, 255, 0),      # kitchen (green)
    5: (255, 0, 0),    # occupied table (red)
}

width = 20.0
height = 15.0
def main():
    
    

    world = ContinuousSpace(width=20.0, height=15.0, wall_size=1.0)

    kitchen_x, kitchen_y = 1.0, 13.0
    #world.add_object(kitchen_x, kitchen_y, 2.0, "kitchen")

    plate_table_positions = [(5, 5), (10, 10), (14, 7)]
    for x, y in plate_table_positions:
        world.add_object(x, y, 1.0, "target")

    obstacle_positions = [(7, 5), (12, 8), (4, 10), (9, 3)]
    for x, y in obstacle_positions:
        world.add_object(x, y, 1.0, "obstacle")

    # occupied_positions = [(6, 9), (11, 4)]
    # for x, y in occupied_positions:
    #     world.add_object(x, y, 2.0, "occupied")


    world.place_agent(2.0, 2.0, 0.6)

    pygame.init()
    scale = 40
    screen = pygame.display.set_mode((int(world.width * scale), int(world.height * scale)))
    pygame.display.set_caption("Robot Plate Collector")
    clock = pygame.time.Clock()

    directions = ['up', 'down', 'left', 'right', 'up_left', 'up_right', 'down_left', 'down_right']
    key_to_direction = {
        pygame.K_UP: "up",
        pygame.K_DOWN: "down",
        pygame.K_LEFT: "left",
        pygame.K_RIGHT: "right",

        pygame.K_w: "up",
        pygame.K_s: "down",
        pygame.K_a: "left",
        pygame.K_d: "right",

        pygame.K_q: "up_left",
        pygame.K_e: "up_right",
        pygame.K_z: "down_left",
        pygame.K_c: "down_right"
    }

    running = True
    while running:
        screen.fill((255, 255, 255))

        for obj in world.objects:
            x, y, size, obj_type = obj["x"], obj["y"], obj["size"], obj["type"]
            color = COLOR_MAP.get(obj_type, (0, 0, 0))
            rect = pygame.Rect(x * scale, (world.height - y - size) * scale, size * scale, size * scale)
            pygame.draw.rect(screen, color, rect)

        if world.agent:
            (x, y), size = world.agent
            rect = pygame.Rect(x * scale, (world.height - y - size) * scale, size * scale, size * scale)
            pygame.draw.rect(screen, (0, 0, 255), rect)

        pygame.display.flip()

        for event in pygame.event.get():
             if event.type == pygame.QUIT:
                 running = False

        keys = pygame.key.get_pressed()
        for key, direction in key_to_direction.items():
             if keys[key]:
                 world.move_agent_direction(direction, step_size=0.2, sub_step=0.05)
                 reward = world.step_with_reward(direction, step_size=0.2, sub_step=0.05)
                 #print(f"Step reward: {reward:.2f}, Total reward: {world.total_reward:.2f}")

                 break  

             if world.is_task_complete():
                    print("[SUCCESS] All plates delivered to kitchen!")
                    running = False
                    break
        # direction = random.choice(directions)
        # world.move_agent_direction(direction, step_size=0.2, sub_step=0.05)
        # reward = world.step_with_reward(direction, step_size=0.2, sub_step=0.05)
        # print(f"Step reward: {reward:.2f}, Total reward: {world.total_reward:.2f}")
    

        # if world.is_task_complete():
        #     print("[SUCCESS] All plates delivered to kitchen!")
        #     running = False
        #     break
        clock.tick(30)

    pygame.quit()
    import matplotlib.pyplot as plt
    plt.imshow(get_grid_image(world))
    plt.title("Environment Snapshot")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
