__author__ = 'vasdommes'

from gun_laying import GunLayer
from robo_simulator import RoboSimulator
from numpy import pi

if __name__ == '__main__':
    robot = RoboSimulator((0, 0, 0), 200, 981, pi / 4, 0)
    gun_layer = GunLayer(robot.rotate_and_shoot, robot.last_shot_coords)
    gun_layer.calibrate()
    print(gun_layer.gun_params)