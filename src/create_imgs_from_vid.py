from argparse import ArgumentParser
from os import makedirs
from os.path import join, isdir

import cv2

from src.config import args_to_dict


def main(vid_load_path, imgs_save_dir):
    """ Extract and save video frames.

    :param vid_load_path: Path for video from which to create images
    :param imgs_save_dir: Save dir for images
    """
    # create imgs_save_dir if needed
    if not isdir(imgs_save_dir):
        makedirs(imgs_save_dir, exist_ok=True)

    vidcap = cv2.VideoCapture(vid_load_path)
    success, image = vidcap.read()
    count = 0
    while success:
        # save frame as JPG file
        cv2.imwrite(join(imgs_save_dir, "frame%d.jpg") % count, image)
        success, image = vidcap.read()
        count += 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--vid_load_path", type=str,
                        help="Path to video to create images from.")
    parser.add_argument("--imgs_save_dir", type=str, help="Directory where extracted frames get saved")
    arg_dict = args_to_dict(parser)
    main(**arg_dict)
