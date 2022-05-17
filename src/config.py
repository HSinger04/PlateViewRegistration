import cv2
import numpy as np

arg_configs = {
    "REF_IMG_PATH": '"--ref_img_path", type=str, help="Path to reference image. '
                    'Make sure that all markers and the plate edge get detected by corner_localizer.py"',
    "FPS": '"--fps", type=int, help="FPS for video to create. 30 is the default FPS for recording with Pupil Core", default=30',
    "GAZE_POS_CSV_PATH": '"--gaze_pos_csv_path", type=str, help="Path to the file gaze_positions.csv"',
    "ARUCO_DICT": '"--aruco_dict", type=str, help="What predefined ArUco dict to use. '
                  'See https://pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/ '
                  'for more predefined dicts", default="DICT_4X4_50"',
    "REF_PLATE_CORNERS": '"--ref_plate_corners", type=loads, help="Coordinates of reference image\'s plate corners. '
                         'You can get them printed out by running corner_localizer.py"'
}

for key in arg_configs.keys():
    arg_configs[key] = 'parser.add_argument(' + arg_configs[key] + ')'


def add_aruco_dict(arg_dict):
    """

    :param predef_aruco: The name of the predefined ArUco dict to use
    :return: The corresponding ArUco dict object
    """

    aruco_dict = eval("cv2.aruco." + arg_dict["aruco_dict"])
    aruco_dict = cv2.aruco.Dictionary_get(aruco_dict)
    return aruco_dict


def add_ref_plate_corners(arg_dict):
    return np.array(arg_dict["ref_plate_corners"]).astype(np.float32)


def args_to_dict(parser):
    arg_dict = vars(parser.parse_args())
    keys = arg_dict.keys()
    if "aruco_dict" in keys:
        arg_dict["aruco_dict"] = add_aruco_dict(arg_dict)
    if "ref_plate_corners" in keys:
        arg_dict["ref_plate_corners"] = add_ref_plate_corners(arg_dict)
    return arg_dict
