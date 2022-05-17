from argparse import ArgumentParser
from os import listdir
from os.path import join, dirname
from json import dump

import numpy as np
from natsort import natsorted
import cv2

from src.corner_localizer import coords_inside_bounds, get_src_plate_corners
from src.image_registration import register_image
from src.config import arg_configs, args_to_dict


def vid_from_imgs(imgs, vid_save_path, fps):
    """ Create video from imgs.

    :param imgs: Images from which to create video
    :param vid_save_path: Save path for video
    :param fps: FPS to use for creating video
    """
    height, width, _ = imgs[0].shape
    vid = cv2.VideoWriter(vid_save_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))

    for img in imgs:
        vid.write(img)

    cv2.destroyAllWindows()
    vid.release()


def register_and_make_vid(imgs_load_dir, homographies, fps, ref_plate_corners):
    """ Register all images from imgs_load_dir according to homographies if the homography is not None and
    the corresponding image from imgs_load_dir has fully visible plate edges and make a video out of them. Also,
    save the images that got skipped due to invalid homographies or not having fully visible plate edges in a json file.

    :param imgs_load_dir: Directory where the images to register reside.
    :param homographies: List of homographies. For each img from imgs_load_dir, there exists on mapping from said img
    to a reference image. If there isn't a homography from an img to the reference image, the corresponding homography
    entry is None.
    :param fps: FPS to use for creating video.
    :param ref_plate_corners: The plate corners in the reference image. Used to infer if the imgs_load_dir's images'
    plate corners are fully visible or not.
    :return:
    """
    file_names = listdir(imgs_load_dir)
    # natsorted is a sorting algorithm that sorts e.g. ["9.jpg", "10.jpg"] correctly where sorted would fail.
    file_names = natsorted(file_names)

    height, width, _ = cv2.imread(join(imgs_load_dir, file_names[0])).shape

    # register raw images and collect in list
    reg_imgs = []
    skipped_frames = {"no homography": [], "plate corners out of bounds": []}

    for file_name, h in zip(file_names, homographies):
        # If homography exists
        if type(h) == np.ndarray:
            # ...and plate corners are visible in the source image
            src_plate_corners = get_src_plate_corners(ref_plate_corners, h)
            if coords_inside_bounds(src_plate_corners, [height, width], h):
                # ..., register image and append
                img = (cv2.imread(join(imgs_load_dir, file_name)))
                reg_img = register_image(img, h)
                reg_imgs.append(reg_img)
            else:
                skipped_frames["plate corners out of bounds"].append(file_name)
        else:
            skipped_frames["no homography"].append(file_name)

    # Save skipped frames
    skipped_frames_json = join(dirname(imgs_load_dir), "skipped_frames.json")
    with open(skipped_frames_json, "w") as f:
        dump(skipped_frames, f, indent=4)

    # create video
    vid_from_imgs(reg_imgs, imgs_load_dir + "_reg.mp4", fps)


def main(imgs_load_dir, vid_save_path, fps):
    """

    :param imgs_load_dir: Directory of images from which to create video
    :param vid_save_path: Save path for video
    :param fps: FPS to use for created video
    """
    file_names = listdir(imgs_load_dir)
    # natsorted is a sorting algorithm that sorts e.g. ["9.jpg", "10.jpg"] correctly where sorted would fail.
    file_names = natsorted(file_names)
    imgs = []
    for file_name in file_names:
        imgs.append(cv2.imread(join(imgs_load_dir, file_name)))

    vid_from_imgs(imgs, vid_save_path, fps)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--imgs_load_dir", type=str,
                        help="Path to images to create video from.")
    parser.add_argument("--vid_save_path", type=str, help="Path where video gets saved")
    eval(arg_configs["FPS"])
    arg_dict = args_to_dict(parser)
    main(**arg_dict)


