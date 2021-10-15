import gym
import sys
sys.path.insert(0, '../../benchmarking')
from model_tester import TesterAgent
import numpy as np
import scipy.ndimage as snd
from PIL import Image

def rgb_to_gray(img):
    return img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114

def detect_edge(img, threshold=10.0):
    """
    Returns black and white image with edges.
    Uses Sobel-filter in x-y directions + thresholding.
    """
    img = rgb_to_gray(img)
    img_sobel_x = snd.filters.sobel(img, axis=0)
    img_sobel_y = snd.filters.sobel(img, axis=1)
    grad_img = np.sqrt(np.square(img_sobel_x) + np.square(img_sobel_y))
    sbl_max = np.amax(abs(grad_img))
    bn_img = np.abs(grad_img) >= (sbl_max / threshold)
    return bn_img.astype(int) * 255

def locate_car(img):
    """
    Locates the center-of-mass of the car based on its known RGB value.
    There are some curbs with red in it which can be an issue.
    TODO: handle red color in curbs
    """
    red = (img[:, :, 0] == 204).astype(int) * 255
    if np.sum(red) == 0:
        loc = (-1, -1)
    else:
        loc = snd.center_of_mass(red)
    return loc

def dist_from_edge_ahead(img):
    """
    Returns the distance from the edge of the road ahead of the car, in pixels.
    If no edge is detected ahead or car cannot be located returns -1.
    TODO: find car length exactly
    """
    edges = detect_edge(img)
    car_x, car_y = locate_car(img)
    if car_x == -1 or car_y == -1:
        return -1
    col_ahead = edges[:, int(car_y)]
    im_height = col_ahead.shape[0]
    col_ahead = np.flip(col_ahead)
    flipped_car_x = int(im_height - car_x)
    col_ahead[: flipped_car_x + 6] = 0  # 6 is a guess for the car length
    # print(col_ahead)
    indx_road = np.argmax(col_ahead)
    return indx_road - flipped_car_x if indx_road > 0 else -1

def dist_from_right_and_left(img):
    """
    Returns the distance from the right and left edge of the road, in pixels.
    If no edge is detected ahead or car cannot be located returns -1.
    """
    edges = detect_edge(img)
    car_x, car_y = locate_car(img)
    if car_x == -1 or car_y == -1:
        return -1, -1
    row_cur = edges[int(car_x), :]
    left_side = np.flip(row_cur[:int(np.floor(car_y))-3])
    right_side = row_cur[int(np.ceil(car_y))+3:]
    # print(left_side)
    # print(right_side)
    return np.argmax(left_side), np.argmax(right_side)

def strip_indicators(img):
    """
    Removes the indicator bar from the observations to clear up the image.
    """
    return img[:84, :, :]

ACTIONS = [
    [-0.85, 0.0, 0.0],
    [0.85, 0.0, 0.0],
    [-0.3, 0.0, 0.03],
    [0.3, 0.0, 0.03],
    [-0.15, 0.0, 0.02],
    [0.15, 0.0, 0.02],
    [0.0, 0.8, 0.15],
    [0.0, 0.75, 0.15],
    [0.0, 0.0, 0.15],
]


class HCTesterAgent(TesterAgent):
    def __init__(self,
                 car_id=0,
                 actions=ACTIONS,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.car_id = car_id
        self.actions = actions

    def state_to_action(self, s):
        """
        This function should take the most recent state and return the
        action vector in the exact same form it is needed for the environment.
        If you are using frame buffer see example in _update_frame_buffer
        how to take care of that.
        """
        try:
            s = s[self.car_id]
            s = strip_indicators(s)  # Get rid of indicator bar on the bottom
            dist = dist_from_edge_ahead(s)
            left_dist, right_dist = dist_from_right_and_left(s)

            if dist > 1 and dist < 20 and right_dist > left_dist + 1:
                cur_action = 1
            elif dist > 1 and dist < 20 and left_dist > right_dist + 1:
                cur_action = 0
                ## Slightly off to the center of the road, small direction change
            elif right_dist >= 2 and right_dist < 4:
                cur_action = 4
            elif left_dist >= 2 and left_dist < 4:
                cur_action = 5
                ## Quite off to the center of the road, high direction change
            elif right_dist >= 0 and right_dist < 2:
                cur_action = 2
            elif left_dist >= 0 and left_dist < 2:
                cur_action = 3
                ## All fine, get faster
            else:
                cur_action = 7
        except:
            cur_action = np.random.randint(0, 9)

        return self.actions[cur_action]

    @staticmethod
    def setup_action_space(env):
        """
        This should be the same action space setup function that you used for training.
        Make sure that the actions set here are the same as the ones used to train the model.
        """
        env.cont_action_space = ACTIONS
        env.action_space = gym.spaces.Discrete(len(env.cont_action_space))

    @staticmethod
    def get_observation_type():
        """
        Simply return 'frames' or 'features'
        """
        return 'frames'