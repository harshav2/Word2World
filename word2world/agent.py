import gym
from gym import spaces
import numpy as np
from PIL import Image

import os
import json
import time
from .utils import map_to_list, load_image_dict
from .solvers import find_characters
import imageio
from .fixers import pad_rows_to_max_length
from PIL import Image, ImageDraw, ImageFont
from rembg import remove

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self):
        
        return self.action_space.sample()
    
class LLMAgent:
    def __init__(self):
        pass

    def action(self, action_string):
        if action_string == 'move_up':
            return 0
        if action_string == 'move_down':
            return 1
        if action_string == 'move_left':
            return 2
        if action_string == 'move_right':
            return 3
        if action_string == 'pick_object':
            return 4
        if action_string == 'hit_enemy':
            return 5
        

'''What is this file. The class I deleted was all just world map setup'''