from argparse import ArgumentParser
from os import makedirs
from os.path import basename, join, splitext, isdir
from json import loads

import cv2
import numpy as np

from src import create_imgs_from_vid, norm_pos_registration
from src.corner_localizer import localize_aruco
from src.create_vid_from_imgs import register_and_make_vid
from src.config import arg_configs, args_to_dict


def main(raw_vid_load_path, gaze_vid_load_path, gaze_pos_csv_path, fps, ref_img_path, aruco_dict, save_dir,
         ref_plate_corners):
    """ Given videos from the perspective of a user, one with and one without the gaze heatmap, a reference image
    (and its plate corner coordinates) and the gaze csv, save the frames of the videos, create videos from the frames
    registered according to the reference image if for the corresponding frame, a homography can be found and its
    plate's corners are visible and note down the frames that got skipped due to missing homography or non-visible
    plate corners. Also, update the gaze position csv with registered coordinates (see norm_pos_registration.py's main
    for more info).

    :param raw_vid_load_path: Path to world.mp4 in its raw format (i.e. without gaze heatmap)
    :param gaze_vid_load_path: Path to world.mp4 with gaze heatmap.
    :param gaze_pos_csv_path: Path to the file gaze_positions.csv
    :param fps: The FPS to use to create the videos
    :param ref_img_path: Path to the reference image according to whose perspective to register the videos' frames.
    :param aruco_dict: ArUco dict object corresponding to the provided markers in the videos and reference image.
    :param save_dir: Parent directory where the generated files get saved.
    :param ref_plate_corners: Coordinates of reference image's plate corners. You can get them printed out by
    running corner_localizer.py
    """

    ref_img = cv2.imread(ref_img_path)
    _, ref_ids, _ = localize_aruco(ref_img, aruco_dict)
    current_id = 1
    # Check that there are no gaps in the detected markers' ids.
    for id in ref_ids:
        if not current_id == id:
            raise ValueError("Either marker with id == 1 not detected, or there is a gap in the detected marker ids")
        current_id += 1

    raw_imgs_load_dir = join(save_dir, "raw_imgs")
    gaze_imgs_load_dir = join(save_dir, "gaze_imgs")

    # Create dirs if needed
    if not isdir(raw_imgs_load_dir):
        makedirs(raw_imgs_load_dir, exist_ok=True)
    if not isdir(gaze_imgs_load_dir):
        makedirs(gaze_imgs_load_dir, exist_ok=True)

    # Create imgs from videos
    create_imgs_from_vid.main(raw_vid_load_path, raw_imgs_load_dir)
    create_imgs_from_vid.main(gaze_vid_load_path, gaze_imgs_load_dir)

    # register the norm_pos coordinates of the csv file
    gaze_pos_df, homographies = norm_pos_registration.main(gaze_pos_csv_path, raw_imgs_load_dir, ref_img_path,
                                                           aruco_dict)

    # Register and create videos of raw and gaze images
    register_and_make_vid(raw_imgs_load_dir, homographies, fps, ref_plate_corners)
    register_and_make_vid(gaze_imgs_load_dir, homographies, fps, ref_plate_corners)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_vid_load_path", type=str, help="Path to world.mp4 in its raw format "
                                                              "(i.e. without gaze heatmap)")
    parser.add_argument("--gaze_vid_load_path", type=str, help="Path to world.mp4 with gaze heatmap.")
    parser.add_argument("--save_dir", type=str, help="Path where generated data gets saved", default="../data/main/out")
    eval(arg_configs["REF_PLATE_CORNERS"])
    eval(arg_configs["GAZE_POS_CSV_PATH"])
    eval(arg_configs["FPS"])
    eval(arg_configs["REF_IMG_PATH"])
    eval(arg_configs["ARUCO_DICT"])
    arg_dict = args_to_dict(parser)
    main(**arg_dict)