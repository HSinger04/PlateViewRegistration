from argparse import ArgumentParser
from os import makedirs
from os.path import join, isdir

import numpy as np
import cv2

from src.config import arg_configs, args_to_dict


# IDEA: One could try to expand to code to even allow usage of custom aruco dicts.
def main(aruco_dict, aruco_save_dir, side_pixels, num_markers):
    """ Generate and save ArUco markers.

    :param aruco_dict: ArUco dict to generate ArUco markers
    :param aruco_save_dir: Path where ArUco markers get saved.
    :param side_pixels: Height and width pixels of ArUco marker image.
    :param num_markers: How many markers to generate
    """
    # create save_dir if needed
    if not isdir(aruco_save_dir):
        makedirs(aruco_save_dir, exist_ok=True)

    for marker_id in range(1, num_markers+1):
        # image where ArUco marker gets written to
        marker = np.zeros((side_pixels, side_pixels, 1), dtype="uint8")
        cv2.aruco.drawMarker(aruco_dict, marker_id, side_pixels, marker, 1)

        # save marker
        cv2.imwrite(join(aruco_save_dir, "ArUco_" + str(marker_id) + ".jpg"), marker)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--side_pixels", type=int, help="Height and width pixels of ArUco marker image.")
    parser.add_argument("--aruco_save_dir", type=str, help='Path where ArUco markers get saved.')
    parser.add_argument("--num_markers", type=int, help="How many markers to generate")
    eval(arg_configs["ARUCO_DICT"])
    arg_dict = args_to_dict(parser)
    main(**arg_dict)
