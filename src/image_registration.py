from argparse import ArgumentParser

import cv2
import numpy as np

from src.corner_localizer import localize_aruco
from src.config import arg_configs, add_aruco_dict, args_to_dict


def find_correspondences(src_img, ref_coords_tuple, aruco_dict):
    # TODO: Change all the docs here. Just copy-paste after having written it in the document
    """ Return 4 corresponding points between source and reference image, given by src_img and ref_coords_tuple by
    picking the corresponding points from the first 4, 3, 2 or 1 ArUco marker (in this order of preference).

    :param src_img: Image to register. Has ArUco markers.
    :param ref_coords_tuple: The corners of the ArUco markers of the reference image, where the corners are sorted
    according to lowest id first (see localize_aruco's docstring).
    :param aruco_dict: ArUco dict object corresponding to the provided markers in src_image.
    :return: 4 corresponding points between source and reference image
    """

    src_coords_tuple, src_coords_ids, _ = localize_aruco(src_img, aruco_dict)

    if src_coords_tuple:
        # Choose the first (up to) 4 ArUco marker coordinates detected in src_img
        max_first_four_src_coords = src_coords_tuple[:4]
        # max_first_four_ref_idxs maps to the 4 ArUco markers in the reference image
        # corresponding to those of max_first_four_src_coords
        max_first_four_ref_idxs = [src_coords_id[0] - 1 for src_coords_id in src_coords_ids[:4]]

        src_coords = []
        ref_coords = []
        # From the first (up to) 4 ArUco markers, choose the top left corners as corresponding points...
        for src_marker_coords, ref_marker_idx in zip(max_first_four_src_coords, max_first_four_ref_idxs):
            src_coords.append(src_marker_coords[0][0])
            ref_coords.append(ref_coords_tuple[ref_marker_idx][0][0])

        # ... and use the first marker's other coordinates for rest of corresponding points if only 1 or 3 markers
        # were detected ...
        if not len(max_first_four_src_coords) == 2:
            for i in range(1, 4 - len(max_first_four_src_coords) + 1):
                src_coords.append(max_first_four_src_coords[0][0][-i])
                ref_coords.append(ref_coords_tuple[max_first_four_ref_idxs[0]][0][-i])

        # ... and if only 2 markers were detected, fill up the last 2 corresponding points with a corner from
        # each of the 2 markers.
        else:
            src_coords.append(max_first_four_src_coords[0][0][1])
            ref_coords.append(ref_coords_tuple[max_first_four_ref_idxs[0]][0][1])
            src_coords.append(max_first_four_src_coords[1][0][1])
            ref_coords.append(ref_coords_tuple[max_first_four_ref_idxs[1]][0][1])

        src_coords = np.reshape(src_coords, (1, 4, 2))
        ref_coords = np.reshape(ref_coords, (1, 4, 2))

        # Deprecated: Initially, I just chose the corners of the first detected marker of the source image
        # as src_coords.
        # # Pick the source image's marker first from the tuple for getting the homography
        # src_coords = src_coords_tuple[0]
        # # Get the index for the corresponding ref coords. Assumes that ref_coords_tuples contains all markers' coords
        # # in the order from lowest to highest id.
        # ref_coords_idx = src_coords_ids[0][0] - 1
        # ref_coords = ref_coords_tuple[ref_coords_idx]

        return src_coords, ref_coords

    # If no markers were detected, return Nones
    return None, None


def get_homography(src_img, ref_coords_tuple, aruco_dict):
    """ Get homography between src_img and reference image represented by ref_coords_tuple if corresponding
    coordinates can be found. Else, return homography.

    :param src_img: We ideally get the homography mapping from src_img to the reference image represented by ref_coords_tuple
    :param ref_coords_tuple: We ideally get the homography mapping from src_img to the reference image represented by ref_coords_tuple
    :param aruco_dict: ArUco dict object corresponding to the provided markers in src_image.
    :return: Get homography between src_img and reference image represented by ref_coords_tuple if corresponding
    coordinates can be found. Else, return homography.
    """
    src_coords, ref_coords = find_correspondences(src_img, ref_coords_tuple, aruco_dict)
    h = None
    if type(src_coords) == np.ndarray:
        # Just use one pair of quadruples of coordinates
        h = cv2.getPerspectiveTransform(src_coords, ref_coords)

        # Deprecated part: Checks if plate corners of source image inside bounds:
        # # Get the plate coords in src_img
        # src_plate_corners = get_src_plate_corners(ref_plate_corners, h)
        # if not coords_inside_bounds(src_plate_corners, src_img.shape[:2]):
        #     # if plate corners not inside source image, treat it as if there was no homography found.
        #     h = None

    return h


def register_image(src_img, homography, height=0, width=0):
    """ Register src_img according to a reference image's perspective.
    Also see https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/

    :param src_img: Image to register
    :param homography: The homography matrix from src_img to a reference image
    :param height: The height of the reference image. If the reference image has the same height as the width,
    you can leave height == 0.
    :param width: The width of the reference image. If the reference image has the same width as the width,
    you can leave width == 0.
    :return: src_img from the perspective given by reference image defined through the homography
    """
    if homography is None:
        raise ValueError("Homography is None.")
    if not height:
        height = src_img.shape[0]
    if not width:
        width = src_img.shape[1]
    reg_img = cv2.warpPerspective(src_img, homography, (width, height))
    return reg_img


def main(src_path, ref_img_path, reg_save_path, aruco_dict):
    """ Register and save image given by src_path according to perspective given by the image behind ref_img_path

    :param src_path: Path to image to register
    :param ref_img_path: Path to reference image
    :param reg_save_path: Path where registered image gets saved
    :param aruco_dict: ArUco dict object corresponding to the provided markers in the source and reference image.
    """

    src_img = cv2.imread(src_path)
    ref_img = cv2.imread(ref_img_path)
    ref_coords_tuple, ref_coords_ids, _ = localize_aruco(ref_img, aruco_dict)
    h = get_homography(src_img, ref_coords_tuple, aruco_dict)
    reg_img = register_image(src_img, h)

    cv2.imwrite(reg_save_path, reg_img)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src_path", type=str,
                        help="Path to image to register.", default="../data/img_registration_data/src_img.jpg")
    parser.add_argument("--reg_save_path", type=str, help="Path to save registered image",
                        default="../data/img_registration_data/reg_img.jpg")
    eval(arg_configs["REF_IMG_PATH"])
    eval(arg_configs["ARUCO_DICT"])
    arg_dict = args_to_dict(parser)
    main(**arg_dict)
