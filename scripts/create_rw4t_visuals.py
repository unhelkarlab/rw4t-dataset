from PIL import Image
from PIL import ImageDraw
from typing import Tuple
import numpy as np


BACKGROUND = 'images/SimulationTestBed.png'
HAZARD_ICON = 'images/hazard.png'
HUMAN_ICON = 'images/delivery-man.png'
DRONE_ICON = 'images/drone.png'


class Rw4TImage():
    """
    10x10 environment
    
    """
    def __init__(self, continuous=False):
        self.continuous = continuous
        self.background = Image.open(BACKGROUND)
        self.canvas_size = np.array(self.background.size)
        self.hazard = Image.open(HAZARD_ICON).resize((self.canvas_size/10).astype(int))
        self.human = Image.open(HUMAN_ICON).resize((self.canvas_size/10).astype(int))
        self.drone = Image.open(DRONE_ICON).resize((self.canvas_size/10).astype(int))

        if continuous:
            self.normalizer=80
            self.skew = - 0.05 * self.canvas_size

        else:
            self.normalizer=10
            self.skew = 0

    def get_squashed_loc(self, state, max_loc=80):
        return state[:2] / max_loc

    def paste_hazard(self, canvas: Image, loc: Tuple):
        """Paste hazard
        
        canvas: a copy of background to paste on
        loc: x, y coordinates from 0 to 1 to paste
            hazard. Recommended to be a multiple of 0.1
            to fit perfectly, as the enviornment is 
            based on a 10 by 10 grid.
        """
        loc = loc[::]
        canvas.paste(self.hazard, (loc[::-1]/10 * self.canvas_size).astype(int).tolist())

    def paste_human(self, canvas: Image, loc: Tuple):
        loc = loc[::]
        canvas.paste(self.human, (loc[::-1]/self.normalizer * self.canvas_size + self.skew).astype(int).tolist())

    def paste_robot(self, canvas: Image, loc: Tuple):
        loc = loc[::]
        canvas.paste(self.drone, (loc[::-1]/self.normalizer * self.canvas_size + self.skew).astype(int).tolist())
    
    def paste_medical_kit(self, canvas, loc: np.ndarray):
        """loc represents xy coordinates from 0 to 1"""
        pos_start = (loc / 10 + (0.04, 0.04) )* self.canvas_size
        pos_end = pos_start + 20
        pos_start = tuple(pos_start.astype(int))
        pos_end = tuple(pos_end.astype(int))
        ImageDraw.Draw(canvas).ellipse(
            (*pos_start, *pos_end), fill = 'red', outline ='red')
