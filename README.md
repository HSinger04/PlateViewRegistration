# TODO

* Make sure that not fully recorded plates get thrown out (mit return von localize_aruco bestimmbar)
* When a video gets loaded as input, make sure to specify whether it's the one with or without gaze heatmaps
* Documentation
  * Create a full guide on how to use this
    * How to get a reference image
      * Take a picture and make sure that ALL ArUco markers get detected (run the function and check output).
    * Examples includes examples
      * Add default arguments or sth for telling what to do

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

(Reference) image that contains all the markers and optionally also the plate edges in full.

##### Potential prerequisite

generate_ArUco.py to generate ArUco markers

#### Purpose
Detect (ArUco) markers, their ids and top-left corners and optionally plate edges, draw them in 
and save the image. You should run this function on the image you want to use as reference 
image and check it's output to make sure that all markers and their plate edges get successfully 
detected.

### main.py

#### Required data

* world.mp4 in its raw format (i.e. without the gaze heatmap)
* world.mp4 with gaze heatmap
* The gaze_positions.csv file
* A reference image according to which's perspective the world.mp4's frames get registered.

##### Potential prerequisite

corner_localizer.py (to assert that all markers and plate edges were successfully detected) and its 
prerequisites.

#### Purpose

## Optional

### create_imgs_from_vid.py

#### Required data

A video file

### create_vid_from_imgs.py

#### Required data

Images from which to create the video

### norm_pos_registration.py

#### Required data

* Extracted frames of world.mp4 in its raw format (i.e. without gaze heatmap)
* The gaze_positions.csv file
* A reference image according to which's perspective the world.mp4's frames get registered.

##### Potential prerequisite

* corner_localizer.py (to assert that all markers and plate edges were successfully detected) and its 
prerequisites.
* create_imgs_from_vid.py (to extract frames of world.mp4)

TODO
* Extract and save frames of both world.mp4 files
* 