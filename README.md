# Child Gesture Recognition

This repository contains a test code for our CogSci-2020 paper:

    Soumitra Samanta, Colin Bannard, Julian Pine and The Language05 Team
    "Can Automated Gesture Recognition Support the Study of Child Language Development?"
    in Proc. CogSci 2020


# Overview

This is test code for child gesture recognition. Here we have considered 12 different child gestures: grasp object (GO), give (GV), hold out (HO), lower object (LO), object manipulation (OM), other (OT), point-declarative (PD), point-imperative (PI), reaches-imperative (RI), reaches-declarative (RD), retract object (RO), and share orientation (SO).

# Prerequisites
- python 3.7 or might work for python2.x (not tested)
- numpy
- [PyTorch](https://pytorch.org/) (1.4.0)
- [pyflow](https://github.com/pathak22/pyflow)
- scikit-learn interface based [thundersvm](https://github.com/Xtra-Computing/thundersvm/tree/master/python)

Please save the installation of [pyflow](https://github.com/pathak22/pyflow) under `"third_party_libs/optical_flow/pyflow/"` and [thundersvm](https://github.com/Xtra-Computing/thundersvm/tree/master/python) under `"third_party_libs/svm/thundersvm/thundersvm_cpu/"` (if you are using cpu version of [thundersvm](https://github.com/Xtra-Computing/thundersvm/tree/master/python)) or  `"third_party_libs/svm/thundersvm/thundersvm_gpu/"` (if you are using gpu version of [thundersvm](https://github.com/Xtra-Computing/thundersvm/tree/master/python)). 

**Special note** for [pyflow](https://github.com/pathak22/pyflow) installation: you can off some default print statements (on the screen) by editing `"bool OpticalFlow::IsDisplay=false;"` at **line no 13** in [OpticalFlow.cpp](https://github.com/pathak22/pyflow/blob/master/src/OpticalFlow.cpp) during the pyflow installation. 

# Description

This code has various dependencies, some are listed in **Prerequisites** section. Code is a bit slow due to some data preprocessing steps, mainly optical flow calculation ([pyflow](https://github.com/pathak22/pyflow) is slow but gives better accuracy compared to others). You can use any other package to calculate the optical flow (which might be faster but keep in mind the accuracy may vary due to nose in optical flow calculation!) and save the optical flow as a grayscale image in a specific format described in the **Example** section.

We have used three fold cross-validation for our evaluation. There are six **cnn (three spatial and three motion net)** models in `"model/cnn_net/"` folder and for each **spatial & motion net** there are two **svm (for two diffrent feature pooling: avg & max)** models in `"model/svm/"`. The test script use default **cnn & svm** model trained on 1st fold (in three fold) data. To change the model, please see the different parameters option in the **Example** section.    

# Examples:

To see different parameters option, from a shell (Bash, Bourne, or else) run:

```bash
python ss_segment_child_gesture.py -h
```

To test on a particular video file (say `"my_video.mp4"`), from a shell (Bash, Bourne, or else) run the following script (slow due to optical flow calculation):

```bash
python ss_segment_child_gesture.py --input_video_filename my_video.mp4
```

To test on multiple video files, put all the video files in a folder (say all the files are in `"data/"` folder). From a shell (Bash, Bourne, or else) run the following script (slow due to optical flow calculation):

```bash
python ss_segment_child_gesture.py --input_video_folder data/
```

If you have preprocessed data ( the video file in converted into images and calculated the optical flow), then please save the preprocessed data according to the following convention (for simplicity let your video file name is `"my_video.mp4"`) :

- keep all the images under `"data/images/my_video/"`
- keep all the horizontal (`u`) and vertical (`v`)optical flow vectors as a grayscale image under `"data/optical_flow/u/my_video/"` and `"data/optical_flow/v/my_video/"`. 

Then from a shell (Bash, Bourne, or else) run the following script:

```bash
python ss_segment_child_gesture.py --input_video_filename my_video.mp4 --input_images_path data/images/ --input_images_path data/optical_flow/
```

If you find the code useful for your research, please cite our paper:

        @inproceedings{soumitrasamantacogsci20,
          title={Can Automated Gesture Recognition Support the Study of Child Language Development?},
          author={Soumitra Samanta and Colin Bannard and Julian Pine and The Language05 Team},
          booktitle={42nd Annual Virtual Meeting of the Cognitive Science Society (CogSci)},
          year={2020}
        }

