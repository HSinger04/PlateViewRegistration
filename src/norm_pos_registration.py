from argparse import ArgumentParser
from os import listdir, makedirs
from os.path import join, isdir

import cv2
import numpy as np
import pandas as pd
from natsort import natsorted

from src.image_registration import get_homography, register_image
from src.corner_localizer import localize_aruco
from src.config import arg_configs, args_to_dict

HOMOGRAPHIES = "homographies"
NORM_POS_X_REG = "norm_pos_x_registered"
NORM_POS_Y_REG = "norm_pos_y_registered"
GAZE_TIMESTAMP = "gaze_timestamp"
WORLD_INDEX = "world_index"


def register_and_save_gaze_imgs(gaze_imgs_load_dir, gaze_pos_df, imgs_save_dir, homographies):
    """ Registers the frames of world.mp4 with gaze heatmaps and also draws in the gaze points
    from the registered norm_pos_x_registered and norm_pos_y_registered of gaze_pos_df to check if the calculation of
    the norm_pos_x_registered and norm_pos_y_registered was done correctly or not. So ultimately, this function
    just serves for checking if register_and_save_gaze_pos works correctly or not.

    :param gaze_imgs_load_dir: Path to frames of world.mp4 with gaze heatmap.
    :param gaze_pos_df: Pandas dataframe of gaze_positions.csv
    :param imgs_save_dir: Directory where the registered gaze images get saved.
    :param homographies: Homographies mapping from the frames of world.mp4 WITHOUT gaze heatmap to a reference image.
    """
    # create save_dir if needed
    if not isdir(imgs_save_dir):
        makedirs(imgs_save_dir, exist_ok=True)

    file_names = listdir(gaze_imgs_load_dir)
    # natsorted is a sorting algorithm that sorts e.g. ["9.jpg", "10.jpg"] correctly where sorted would fail.
    file_names = natsorted(file_names)

    # get height and width from first found image
    temp_img = cv2.imread(join(gaze_imgs_load_dir, file_names[0]))
    height, width, _ = temp_img.shape

    # extract only relevant subset of gaze_pos_df, hopefully leading to faster computation
    norm_pos_df = gaze_pos_df[[GAZE_TIMESTAMP, WORLD_INDEX, NORM_POS_X_REG, NORM_POS_Y_REG]]

    for world_index, file_name in enumerate(file_names):
        # get corresponding homography
        h = homographies[world_index]

        # only work on images where a valid homography matrix got extracted
        # i.e. corresponding parts between img and ref_img were found
        if type(h) == np.ndarray:
            img = cv2.imread(join(gaze_imgs_load_dir, file_name))
            reg_img = register_image(img, h)

            # function name is self-explanatory
            def draw_and_save_gaze_circles(norm_pos_x_reg, norm_pos_y_reg, gaze_timestamp):
                c_x = int(norm_pos_x_reg * width)
                c_y = int((1 - norm_pos_y_reg) * height)
                temp_img = reg_img.copy()
                cv2.circle(temp_img, (c_x, c_y), 4, (0, 0, 255), -1)
                cv2.putText(temp_img, "gaze point",
                            (c_x, c_y - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
                cv2.imwrite(join(imgs_save_dir, str(gaze_timestamp) + ".jpg"), temp_img)

            # draw and save gaze circles
            img_norm_pos_df = norm_pos_df[norm_pos_df[WORLD_INDEX] == world_index]
            img_norm_pos_df.apply(
                lambda x: draw_and_save_gaze_circles(x[NORM_POS_X_REG], x[NORM_POS_Y_REG], x[GAZE_TIMESTAMP]), axis=1)


def register_and_save_gaze_pos(imgs, ref_img, gaze_pos_df, gaze_save_path, aruco_dict):
    """ Registers norm_pos_x and norm_pos_y according to the homographies extracted from imgs and ref_imgs and adds
    them (as well as the homographies) to gaze_pos_df.

    :param imgs: List of images to register.
    :param ref_img: The reference image according to which to register imgs.
    :param gaze_pos_df: Pandas dataframe of gaze_positions.csv
    :param gaze_save_path: Save path for gaze_positions.csv with added column for the homographies,
    norm_pos_x_registered and norm_pos_y_registered
    :param aruco_dict: ArUco dict object corresponding to the provided markers in the videos and reference image.
    :return: gaze_pos_df with added columns for homographies, norm_pos_x_registered and norm_pos_y_registered and the
    homographies as a list (redundant, but slightly convenient for using the homographies in other functions).
    """
    # Extract ArUco marker coordinates and ids of reference image
    ref_coords_tuple, ref_coords_ids, _ = localize_aruco(ref_img, aruco_dict)
    homographies = []

    # Collect homography matrices for each gaze_timestamp (see gaze_positions.csv)
    homographies_per_gaze_ts = []
    for world_index, img in enumerate(imgs):
        h = get_homography(img, ref_coords_tuple, aruco_dict)
        homographies.append(h)
        homographies_per_gaze_ts += [h for _ in range(len(gaze_pos_df[gaze_pos_df[WORLD_INDEX] == world_index]))]

    # store the homographies for each timestamp in the df
    gaze_pos_df[HOMOGRAPHIES] = homographies_per_gaze_ts
    height, width, _ = ref_img.shape

    # function that registered norm_pos_x and norm_pos_y from gaze_pos_df and returns their registered versions.
    def apply_homographies(x):
        norm_pos_x = x["norm_pos_x"]
        norm_pos_y = x["norm_pos_y"]
        homography = x[HOMOGRAPHIES]

        reg_norm_pos_x = None
        reg_norm_pos_y = None

        if type(homography) == np.ndarray:
            # IDEA: Maybe one can do the perspectiveTransform without having to do (full) unnormalization and
            #  normalization? Since perspectiveTransform is a linear operation AFAIK.

            # unnormalize position
            pos_x = norm_pos_x * width
            pos_y = (1 - norm_pos_y) * height
            pos = np.array([pos_x, pos_y])
            pos = np.reshape(pos, (1, 1, 2))

            # register position
            registered_pos = cv2.perspectiveTransform(pos, homography)

            # normalize position again
            registered_pos = np.reshape(registered_pos, 2)
            reg_norm_pos_x = registered_pos[0] / width
            reg_norm_pos_y = registered_pos[1] / height
            reg_norm_pos_y -= 1
            reg_norm_pos_y = -reg_norm_pos_y

        # save registered versions
        x[NORM_POS_X_REG] = reg_norm_pos_x
        x[NORM_POS_Y_REG] = reg_norm_pos_y
        return x

    # register norm_pos_x and norm_pos_y and save as csv.
    gaze_pos_df = gaze_pos_df.apply(apply_homographies, axis=1)
    gaze_pos_df.to_csv(gaze_save_path, index=False)

    return gaze_pos_df, homographies


def main(gaze_pos_csv_path, raw_imgs_load_dir, ref_img_path, aruco_dict):
    """ Registers norm_pos_x and norm_pos_y according to the homographies extracted from the images of raw_imgs_load_dir
     and reference image from ref_img_path and adds them (as well as the homographies) to the csv given by
     gaze_pos_csv_path.

    :param gaze_pos_csv_path: Path to the file gaze_positions.csv
    :param raw_imgs_load_dir: Path to world.mp4 in its raw format (i.e. without gaze heatmap)
    :param ref_img_path: Path to the reference image according to whose perspective to register the videos' frames.
    :param aruco_dict: ArUco dict object corresponding to the provided markers in the videos and reference image.
    :return: gaze_pos_df with added columns for homographies, norm_pos_x_registered and norm_pos_y_registered and the
    homographies as a list (redundant, but slightly convenient for using the homographies in other functions).
    """
    file_names = listdir(raw_imgs_load_dir)
    # natsorted is a sorting algorithm that sorts e.g. ["9.jpg", "10.jpg"] correctly where sorted would fail.
    file_names = natsorted(file_names)
    imgs = []
    for file_name in file_names:
        imgs.append(cv2.imread(join(raw_imgs_load_dir, file_name)))

    gaze_pos_df = pd.read_csv(gaze_pos_csv_path)
    ref_img = cv2.imread(ref_img_path)

    gaze_pos_df, homographies = register_and_save_gaze_pos(imgs, ref_img, gaze_pos_df, gaze_pos_csv_path, aruco_dict)
    return gaze_pos_df, homographies


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_imgs_load_dir", type=str, help="Path to directory containing extracted frames of "
                                                              "world.mp4 in its raw format (i.e. without gaze heatmap)")
    eval(arg_configs["GAZE_POS_CSV_PATH"])
    eval(arg_configs["REF_IMG_PATH"])
    eval(arg_configs["ARUCO_DICT"])
    arg_dict = args_to_dict(parser)
    main(**arg_dict)
