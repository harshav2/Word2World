from PIL import Image
import time
import sys
import json
import imageio
import os
import argparse
import pygame


from utils import map_to_list, find_most_similar_images
from solvers import find_characters
from fixers import pad_rows_to_max_length

from configs import Config

# Initialize pygame
cfg = Config()

# Define constants

parser = argparse.ArgumentParser(description="Process game inputs")
parser.add_argument('--game_path', type=str, help="A path to JSON file of your game. Derfaults to 'word2world\examples\example_1.json'")
args = parser.parse_args()

if args.game_path:
    if not os.path.exists(args.game_path):
            raise ValueError(f"{args.game_path} does not exist. Please provide an existing path.")
    with open(args.game_path, 'r') as file:
        data = json.load(file)
else:
    game = "example_1"
    game_dir = os.path.join(f"word2world", "examples")

    with open(f'{game_dir}/{game}.json', 'r') as file:
        data = json.load(file)

round_number = "round_0"
character_descriptions_dict = {}
gen_story = data[round_number]["story"]
grid_str = data[round_number]["world"]
grid_str = pad_rows_to_max_length(grid_str)
grid_world = map_to_list(grid_str)

goals = data[round_number]["goals"]

world_1st_layer = data[round_number]["world_1st_layer"]["world"]
world_1st_layer = pad_rows_to_max_length(world_1st_layer)

character_chars = ''
# Player setup
player_pos = [character_chars['@'][0], character_chars['@'][1]]  # Starting position of the player

picked_objects = {}  # Dictionary to keep track of picked objects

# Enemy setup
enemy_pos = [character_chars['#'][0], character_chars['#'][1]]  # Starting position of the enemy
enemy_direction = 1  # Enemy direction: 1 for right, -1 for left
enemy_bullets = []  # List to store enemy bullets
player_bullets = []  # List to store player bullets

def move_player(dx, dy):
    pass

def pick_object():
    pass

def move_enemy():
    pass

def enemy_detect_player():
    pass

def enemy_attack_player():
    pass

def player_shoot():
    pass

def move_bullets():
    pass

def hit_enemy():
    pass


# Game loop
running = True
move_direction = None
shooting = False

frames = []

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                move_direction = pygame.K_a
            elif event.key == pygame.K_d:
                move_direction = pygame.K_d
            elif event.key == pygame.K_w:
                move_direction = pygame.K_w
            elif event.key == pygame.K_s:
                move_direction = pygame.K_s
            if event.key == pygame.K_SPACE:
                hit_enemy()
            if event.key == pygame.K_z:
                shooting = True
        if event.type == pygame.KEYUP:
            if event.key == move_direction:
                move_direction = None
            if event.key == pygame.K_z:
                shooting = False

    if move_direction:
        direction_offsets = {
            pygame.K_a: (-1, 0),  # Move left
            pygame.K_d: (1, 0),   # Move right
            pygame.K_w: (0, -1),  # Move up
            pygame.K_s: (0, 1)    # Move down
        }
        dx, dy = direction_offsets[move_direction]
        if not move_player(dx, dy):
            move_direction = None

    if shooting:
        player_shoot()

    if enemy_pos[0] != -1 and enemy_pos[1] != -1: 
        if enemy_detect_player():
            enemy_attack_player()
        else:
            move_enemy()

    move_bullets()

    pygame.display.flip()
    frame = pygame.surfarray.array3d(pygame.display.get_surface())
    frame = frame.transpose([1, 0, 2])
    frames.append(frame)

pygame.quit()
sys.exit()