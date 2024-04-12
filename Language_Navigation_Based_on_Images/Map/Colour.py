import webcolors
import numpy as np
from sklearn.neighbors import KDTree

class Colour:
    """
    This class is initially set up for converting pixels to colour names,
    but it is not used, the specific implementation of identifying colours is implemented in the Detector class.
    """
    def __init__(self):
        self.colors_list = list(webcolors.CSS3_HEX_TO_NAMES.items())
        self.colour_index = self.rgb_to_index()
        self.colors_rgb = np.array([webcolors.hex_to_rgb(hex_color) for hex_color, _ in self.colors_list])
        self.kdtree = KDTree(self.colors_rgb)

    def rgb_to_index(self):
        colour_index = {}
        for index, (_, color_name) in enumerate(self.colors_list):
            colour_index[color_name] = index
        return colour_index


    def index_to_name(self, index):
        return self.colors_list[index][1]


    def closest_colour(self, requested_colour):
        dist, idx = self.kdtree.query([requested_colour], k=1)
        return self.colors_list[idx[0][0]][1]

    def get_colour(self, requested_colour):
        try:
            name = webcolors.rgb_to_name(requested_colour)
        except ValueError:
            name = self.closest_colour(requested_colour)
        return self.colour_index[name]

