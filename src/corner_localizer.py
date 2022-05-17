from argparse import ArgumentParser

import numpy as np
import cv2
import re

from src.config import arg_configs, args_to_dict


ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()


def localize_aruco(img, aruco_dict):
    """ Detect ArUco markers in image.

    :param img: Image with ArUco markers
    :param aruco_dict: ArUco dict object corresponding to the provided markers.
    :return: The corners and ids of detected ArUco markers and rejected, potential markers. The corner coordinates are
    translation-invariant (e.g. if you were to run this function on an image and a rotated version of it,
    e.g. the first value of the returned coordinates would be corresponding to the same corner). The returned
    corners and ids are sorted according to the lowest id first.
    """
    (corners, ids, rejected) = cv2.aruco.detectMarkers(img, aruco_dict, parameters=ARUCO_PARAMS)

    # sort such that the lowest id comes first
    if type(ids) == np.ndarray:
        sorted_ids_ind = ids.flatten().argsort()
        corners = tuple([corners[i] for i in sorted_ids_ind])
        ids = ids[sorted_ids_ind]

    return corners, ids, rejected


def localize_plate_corners(img, dist_left, dist_up, plate_width, plate_height, marker_size, aruco_dict):
    """ Localizes the plate corners in the given img.

    :param img: Image that has a detectable ArUco marker of id == 1 and all plate corners are visible in it.
    :param dist_left: See help string of ArgumentParser's --dist_left argument
    :param dist_up: See help string of ArgumentParser's --dist_up argument
    :param plate_width: See help string of ArgumentParser's --plate_width argument
    :param plate_height: See help string of ArgumentParser's --plate_height argument
    :param marker_size: See help string of ArgumentParser's --marker_size argument
    :param aruco_dict: ArUco dict object corresponding to the provided markers.
    :return: The plate corners' pixel coordinates in the same format as the return value 'corners' of localize_aruco.
    """
    corners, ids, _ = localize_aruco(img, aruco_dict)
    # Get the corners of marker with id == 1
    (top_left, top_right, _, _) = corners[0][0]
    # top_left = top_left.astype(int)
    # top_right = top_right.astype(int)
    # Get the aruco size in pixels
    pixel_marker_size = np.linalg.norm(top_left - top_right)

    # Function to convert metric values to pixel values
    def to_pixel(length):
        pixel_length = pixel_marker_size * length / marker_size
        return pixel_length.astype(int)
    
    # Convert function arguments with metric values to pixel values
    pixel_dist_left = to_pixel(dist_left)
    pixel_dist_up = to_pixel(dist_up)
    pixel_plate_width = to_pixel(plate_width)
    pixel_plate_height = to_pixel(plate_height)

    # Pixel distance between plate corners and first ArUco marker's id
    plate_edge_left = top_left[0] - pixel_dist_left
    plate_edge_up = top_left[1] - pixel_dist_up

    # Get the pixel plate corner coordinates
    plate_top_left = np.array([plate_edge_left, plate_edge_up])
    plate_top_right = np.array([plate_edge_left + pixel_plate_width, plate_edge_up])
    plate_bottom_right = np.array([plate_edge_left + pixel_plate_width, plate_edge_up + pixel_plate_height])
    plate_bottom_left = np.array([plate_edge_left, plate_edge_up + pixel_plate_height])
    
    # Stack the coordinates together into one in the same format as the return value 'corners' of localize_aruco.
    plate_corners = np.stack([plate_top_left, plate_top_right, plate_bottom_right, plate_bottom_left])
    plate_corners = np.reshape(plate_corners, (1, 4, 2))
    return plate_corners


def my_put_text(img, text, coords):
    """ Convenience wrapper over cv2.putText. Does the operation in-place

    :param img: Image to put text over
    :param text: Text to put in image
    :param coords: Where to put text
    """
    cv2.putText(img, text, coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def draw_markers(img, corners, ids, rejected, draw_all_corners=False):
    """ Draw ids and corners of detected markers in-place

    :param img_copy: Image with markers
    :param corners: Corners of detected markers
    :param ids: IDs of detected markers
    :param rejected: A list of potential markers that were found but ultimately rejected. Doesn't get used but kept
    in case one wants to use them in the future
    :return Image with ids and corners drawn in.
    """

    if len(corners) > 0:
        img_copy = np.copy(img)
        for (marker_corner, marker_id) in zip(corners, ids):
            corners = marker_corner.reshape((4, 2))
            (top_left, top_right, bottom_right, bottom_left) = corners
            # convert each of the (x, y)-coordinate pairs to integers
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
            top_left = (int(top_left[0]), int(top_left[1]))

            # draw the bounding box of the detected marker
            cv2.line(img_copy, top_left, top_right, (0, 255, 0), 2)
            cv2.line(img_copy, top_right, bottom_right, (0, 255, 0), 2)
            cv2.line(img_copy, bottom_right, bottom_left, (0, 255, 0), 2)
            cv2.line(img_copy, bottom_left, top_left, (0, 255, 0), 2)
            # compute and draw the center (x, y)-coordinates of the marker
            center_x = int((top_left[0] + bottom_right[0]) / 2.0)
            center_y = int((top_left[1] + bottom_right[1]) / 2.0)
            cv2.circle(img_copy, (center_x, center_y), 4, (0, 0, 255), -1)
            # draw the marker ID on the image
            my_put_text(img_copy, str(marker_id), (top_left[0], top_left[1] - 15))
            if draw_all_corners:
                # draw the corner names
                my_put_text(img_copy, "top left", (top_left[0] - 15, top_left[1]))
                my_put_text(img_copy, "top right", (top_right[0] - 15, top_right[1]))
                my_put_text(img_copy, "bottom left", (bottom_left[0] - 15, bottom_left[1]))
                my_put_text(img_copy, "bottom right", (bottom_right[0] - 15, bottom_right[1]))
        return img_copy


def get_src_plate_corners(ref_plate_corners, homography):
    """ Get the source image's plate corners based on reference image's plate corners.

    :param ref_plate_corners: Coordinates of reference image's plate corners.
    :param homography: Homography mapping from src image to ref image.
    :return: Inferred source image's plate corners
    """
    h_inv = np.linalg.inv(homography)
    return cv2.perspectiveTransform(ref_plate_corners, h_inv)


def coords_inside_bounds(ref_plate_corners, height_and_width, homography=None):
    """ Check if source image's plate corners are visible in image. If homography is None,
    ref_plate_corners are interpreted as source image's plate corners. If homography is provided, source image's
    plate corners are inferred from ref_plate_corners and homography

    :param ref_plate_corners: The source image's or reference image's plate corner coordinates.
    If homography is None,     ref_plate_corners are interpreted as source image's plate corners.
    If homography is provided, source image's plate corners are inferred from ref_plate_corners and homography
    :param height_and_width: Height and width of the source image as a list-like object.
    :param homography: Homography mapping from src image to ref image.
    If homography is None, ref_plate_corners are interpreted as source image's plate corners.
    If homography is provided, source image's plate corners are inferred from ref_plate_corners and homography
    :return: If the source image's plate edges are visible in the image.
    """
    # Invert height and width
    width_and_height = np.array(height_and_width[::-1])
    src_plate_corners = ref_plate_corners
    # If homography is given, infer source image's plate corners from ref_plate_corners. Otherwise, assume
    # ref_plate_corners is source image's plate corners
    if not homography is None:
        src_plate_corners = get_src_plate_corners(ref_plate_corners, homography)
    src_plate_corners = np.array(src_plate_corners)
    # If any of the corners are out of bound, return False
    if np.any(src_plate_corners < 0):
        return False
    if np.any(src_plate_corners > width_and_height):
        return False
    return True


def main(ref_img_path, save_path, plate_by_aruco, dist_left, dist_up, plate_width, plate_height, marker_size, aruco_dict):
    """ Detect (ArUco) markers and optionally plate corners, draw them in and save the image. Also if plate corners
    are specified, print out the plate corner coordinates.
    Lots of code used from https://pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/.

    :param ref_img_path: Image to detect markers and optionally plate corners in.
    :param save_path: Path where image with detected markers and optionally plate corners gets saved.
    :param plate_by_aruco: Set true if you want to identify plate corners by placing ArUco markers 1 to 4
    on the plate corners i.e. the top left corner of ArUco marker with id 1 is placed on the top left,
    with id 2 on the top right, with id 3 on bottom right and with id 4 on the bottom left corner of the plate
    respectively.
    :param dist_left: See help string of ArgumentParser's --dist_left argument
    :param dist_up: See help string of ArgumentParser's --dist_up argument
    :param plate_width: See help string of ArgumentParser's --plate_width argument
    :param plate_height: See help string of ArgumentParser's --plate_height argument
    :param marker_size: See help string of ArgumentParser's --marker_size argument
    :param aruco_dict: ArUco dict object corresponding to the provided markers.
    """
    img = cv2.imread(ref_img_path)
    corners, ids, rejected = localize_aruco(img, aruco_dict)

    if ids == None:
        raise RuntimeError("None of the markers were detected")

    drawn_img = img.copy()
    drawn_img = draw_markers(img, corners, ids, rejected)

    plate_corners = None
    if plate_by_aruco:
        # Identify plate corners by using the first 4 ArUco markers placed at the corners of the plate.
        first_four_marker_ids = ids[:4]
        if not np.all(np.reshape([1, 2, 3, 4], (4, 1)) == first_four_marker_ids):
            # Save image
            cv2.imwrite(save_path, drawn_img)
            raise RuntimeError("One of the first 4 markers not detected! Detected markers: " + str(ids) +
                               " Check output image for more info")
        first_four_markers = corners[:4]
        # Pick the top left corner of each of the markers for the plate corners.
        plate_corners = np.array(first_four_markers)[:, :, 0, :]
        plate_corners = np.reshape(plate_corners, (1, 4, 2))

    # Check if user wants plate corners identified by measurements.
    else:
        plate_edge_data = [dist_left, dist_up, plate_width, plate_height, marker_size]
        for plate_edge_datum in plate_edge_data:
            if plate_edge_datum is None:
                # User didn't want to localize plate corners
                break
        # Identify plate corners by measurements given by user.
        else:
            plate_corners = localize_plate_corners(img, dist_left, dist_up, plate_width, plate_height, marker_size, aruco_dict)

    if plate_corners is not None:
        drawn_img = draw_markers(drawn_img, plate_corners, [], None, draw_all_corners=True)
        # let user know whether the localized plate corners are within the image or not
        if coords_inside_bounds(plate_corners, img.shape[:2]):
            plate_corners_str = str(plate_corners)
            plate_corners_str = re.sub("\s+", "", plate_corners_str)
            plate_corners_str = plate_corners_str.replace(".", ",")
            plate_corners_str = plate_corners_str.replace(",]", "]")
            plate_corners_str = plate_corners_str.replace("][", "],[")
            print("Plate corners visible! Coordinates in pixels: " + plate_corners_str)
        else:
            print("Plate corners not visible!")

    # Save image
    cv2.imwrite(save_path, drawn_img)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_path", type=str,
                        help="Path where image with detected markers and plate corners gets saved.")
    # TODO: Add that you only need to specify plate_by_aruco or dist_left
    parser.add_argument("--plate_by_aruco", type=bool,
                        help="Set true if you want to identify plate corners by placing ArUco markers 1 to 4 "
                             "on the plate corners i.e. the top left corner of ArUco marker with id 1 is placed on the "
                             "top left, with id 2 on the top right, with id 3 on bottom right "
                             "and with id 4 on the bottom left corner of the plate respectively.")
    parser.add_argument("--dist_left", type=int,
                        help="Horizontal distance from left edge of the plate to the first marker's top left corner"
                             "You can measure the length according to any metric (e.g. millimeter, cm, pixels) as you "
                             "like as long as the used metric stays consistent across all length specifications"
                             "You can see the ids and top left corners of markers in the output of corner_localizer.py"
                             "If the left end of the plate is to the left of the first marker's top left corner, "
                             "provide distance as positive number. If the left end of the plate is to the right of the "
                             "first marker's top left edge, provide distance as negative number.", default=None)
    parser.add_argument("--dist_up", type=int,
                        help="Vertical distance from upper edge of the plate to the first marker's top left corner"
                             "If the upper end of the plate is above of the first marker's top left edge,"
                             "provide distance as positive number. "
                             "If the upper end of the plate is below of the first marker's top left edge,"
                             "provide distance as negative number."
                             "For more info, see help for --dist_left.", default=None)
    parser.add_argument("--plate_width", type=int, help="The plate width. For more info, see help for --dist_left.",
                        default=None)
    parser.add_argument("--plate_height", type=int, help="The plate height. For more info, see help for --dist_left",
                        default=None)
    parser.add_argument("--marker_size", type=int,
                        help="The marker's edge length. For more info, see help for --dist_left", default=None)
    eval(arg_configs["REF_IMG_PATH"])
    eval(arg_configs["ARUCO_DICT"])
    arg_dict = args_to_dict(parser)
    main(**arg_dict)


########################################################################################################################

# Deprecated functions. Just kept for showing what has been initially tried out

# def localize_shapes(img):
#     warn('This is deprecated. Use localize_aruco instead.', DeprecationWarning, stacklevel=2)
#
#     pass


# def localize_colors(img):
#     warn('This is deprecated. Use localize_aruco instead.', DeprecationWarning, stacklevel=2)
#
#     # https://programmingdesignsystems.com/color/color-models-and-color-spaces/index.html
#     # https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
#     # in HSV
#
#     LOWER_BLACK = np.array([0, 0, 25])
#     UPPER_BLACK = np.array([179, 255, 25])
#     LOWER_BLUE = np.array([110, 50, 50])
#     UPPER_BLUE = np.array([130, 255, 255])
#
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     blue_mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
#     black_mask = cv2.inRange(hsv, LOWER_BLACK, UPPER_BLACK)
#     mask = cv2.add(blue_mask, black_mask)
#     res = cv2.bitwise_and(img, img, mask=mask)
#
#     cv2.imwrite("mask.jpg", mask)
#     cv2.imwrite('res.jpg', res)


# def get_marker_to_plate_corners(ref_coords_tuple, ref_coords_ids):
#     """
#
#     :param ref_img: A ref_img that has all the ArUco markers that will show up in src_imgs it will get compared to and
#     all those markers are detectable in ref_img.
#     :return: Mapping from marker id to the difference of the plate edge coordinates and the marker id's edge coordinates
#     """
#     # Also add the plate corners later on
#     marker_to_plate_corners = {}
#     ref_coords_ids = list(ref_coords_ids.flatten())
#     # We need coordinates of marker id == 1 first
#     id_1_coords = ref_coords_tuple[0]
#     plate_coords = id_1_coords
#     for ref_coords, ref_coords_id in zip(ref_coords_tuple, ref_coords_ids):
#         marker_to_plate_corners[ref_coords_id] = plate_coords - ref_coords
#     return marker_to_plate_corners
