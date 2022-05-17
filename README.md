# Usage

## Essential

In the following, I will go over how to use the written code for the plate experiment in the order of 
steps to be taken. You can (and should) run each of the described modules as entry points.
For more information of a respective module, see its main function's documentation.

### generate_ArUco.py

#### Purpose

Generate and save ArUco markers. After generating the ArUco markers, physically print them out and put 
them on the plates for your experiment. 

### corner_localizer.py

#### Required data

* (Reference) image that contains all the markers and optionally also the plate corners in full.
* For localizing plate corners (necessary to fully use main.py): Either place markers with id 1 to 4
at the corners of the plate (recommended) or specify plate and ArUco marker measurements.
See corner_localizer.py's main function's doc string for more information.

##### Potential prerequisite

generate_ArUco.py to generate ArUco markers

#### Purpose
Detect (ArUco) markers, their ids (shown at the top-left corner of the marker) and optionally plate corners, draw them in 
and save the image. You should run this function on the image you want to use as reference 
image and check it's output to make sure that all markers and their plate corners get successfully 
detected. Also, if the plate corners are to be located, it prints the detected corner
coordinates at the end (copy-paste the WHOLE print output as input for
specifying ref_plate_corners in other functions).

### main.py

#### Required data

* world.mp4 in its raw format (i.e. without the gaze heatmap)
* world.mp4 with gaze heatmap
* The gaze_positions.csv file
* A reference image according to which's perspective the world.mp4's frames get registered
(see part about corner_localizer.py).
* The coordinates of the plate corners as printed out by corner_localizer.py

##### Potential prerequisite

corner_localizer.py (to assert that all markers and plate corners were successfully detected and 
for getting the plate corner coordinates) and its prerequisites. 

#### Purpose

Given videos from the perspective of a user, one with and one without the gaze heatmap, a reference image
    (and its plate corner coordinates) and the gaze csv, save the frames of the videos, create videos from the frames
    registered according to the reference image if for the corresponding frame, a homography can be found and its
    plate's corners are visible and note down the frames that got skipped due to missing homography or non-visible
    plate corners. Also, update the gaze position csv with registered coordinates (see norm_pos_registration.py's main
    for more info).

## Optional

### create_imgs_from_vid.py

#### Required data

A video file

#### Purpose

Extracts and saves frames of video file.

### create_vid_from_imgs.py

#### Required data

Images from which to create the video

#### Purpose

Create video from images.

### norm_pos_registration.py

#### Required data

* Extracted frames of world.mp4 in its raw format (i.e. without gaze heatmap)
* The gaze_positions.csv file
* A reference image according to which's perspective the world.mp4's frames get registered 
* (see part about corner_localizer.py).

##### Potential prerequisite

* corner_localizer.py (to assert that all markers and plate corners were successfully detected) and its 
prerequisites.
* create_imgs_from_vid.py (to extract frames of world.mp4)

#### Purpose

Registers norm_pos_x and norm_pos_y according to the homographies extracted from the images of raw_imgs_load_dir
     and reference image from ref_img_path and adds them (as well as the homographies) to the csv given by
     gaze_pos_csv_path.
