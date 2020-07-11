##############################################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 06-06-2020
#% For bug and others mail me at soumitramath39@gmail.com
#%--------------------------------------------------------------------------------------------
#% INPUT:
#% OUTPUT:
#%--------------------------------------------------------------------------------------------
#% EXAMPLE:
#%
##############################################################################################
import numpy as np
import tqdm
import argparse
import glob
import scipy.io as sio
import sys

import torch

sys.path.insert(0, 'third_party_libs/optical_flow/pyflow/')# for optical flow
sys.path.insert(0, 'third_party_libs/svm/thundersvm/python/')# for gpu based SVM

from data_utils import *
from model_utils import *


#---------------------------------------------------------------------------------------------
##############################################################################################

parser = argparse.ArgumentParser(description='********child gesture segmentation********')
parser.add_argument('--input_video_filename', default='', type=str, metavar='FILE', help='input video file (default: empty)')
parser.add_argument('--input_video_folder', default='', type=str, metavar='PATH', help='folder contains input video files  (default: data/)')
parser.add_argument('--input_video_file_format', default='mp4', type=str, metavar='FF', help='input video file format (default: mp4)')
parser.add_argument('--input_images_folder', default='', type=str, metavar='PATH', help='folder contains input images files (saved images under data/images/video_filename/)  (default: empty)')
parser.add_argument('--input_opticalflow_folder', default='', type=str, metavar='PATH', help='folder contains input optical flow (OF) files (saved OF under data/optical_flow/u/video_filename/ and data/optical_flow/v/video_filename/) (default: empty/)')
parser.add_argument('--child_gesture_duration', default=45, type=int, metavar='N', help='minimum number of frames for a gesture (default: 45)')
parser.add_argument('--feature_pool', default='max', type=str, metavar='FP', help='temporal feature pooling operator (max or avg) (default: max)')
parser.add_argument('--cnn_svm_model_id', default=1, type=int, metavar='N', help='cnn and svm model id (1, 2, 3) (default: 1)')
parser.add_argument('--batch_size', default=4, type=int, metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--num_workers', default=2, type=int, metavar='N', help='number of worker will be used during data loader(default: 2)')
parser.add_argument('--preprocessed_data_cache_path', default='temp_cache/', type=str, metavar='PATH', help='path to save the preprocessed data (default: temp_cache/)')
parser.add_argument('--save_result_path', default='result/', type=str, metavar='PATH', help='path to save the result (default: result/)')
parser.add_argument('--output_video_save_flag', default=1, type=int, metavar='flag', help='output video saving flag (1-yes, 0-no) (default: 1)')
parser.add_argument('--output_video_file_format', default='mp4', type=str, metavar='FF', help='output video file format (default: mp4)')

args = parser.parse_args()
print('########################################################################################')
print('your inputs are as follows:')
print(args)
print('########################################################################################')

def main():
    
    # different parameter
    org_modified_child_gesture_dict = {1:'grasp object', 2:'give', 3:'hold out', 4:'lower object', 5:'object manipulation', 6:'other', 
                                      7:'point-declarative', 8:'point-imperative', 9:'reaches-imperative', 10:'reaches-declarative', 
                                      11:'retract object', 12:'share orientation'}

    # parameters 
    num_flow_channel = 10
    feature_dimension = 2048
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    save_result_path = args.save_result_path
    save_result_path = folder_create(save_result_path)
    save_result_video_path = '' .join([save_result_path, 'video/'])
    save_result_video_path = folder_create(save_result_video_path)


    # load cnn and svm models
    print('----------------------------------------------------------------------------------------')
    print('loading cnn and svm model')
    print('----------------------------------------------------------------------------------------')
    spatial_cnn_model, motion_cnn_model, spatial_motion_svm_model = losd_cnn_svm_model(8, num_flow_channel, args.cnn_svm_model_id, args.feature_pool, device=device)
    print('########################################################################################')

    
    # for single file
    if len(args.input_video_filename):
        
        input_video_filename = args.input_video_filename
        
        #preprocess the video file
        print('processing "{}" ({}/{})' .format(args.input_video_filename, 1, 1))
        print('----------------------------------------------------------------------------------------')
        print('pre-processing video file: {}' .format(args.input_video_filename))
        data_info = preprocess_video(args.input_video_filename, images_data_path=args.input_images_folder, opticalflow_data_path=args.input_opticalflow_folder, cache_data_path=args.preprocessed_data_cache_path)
        
        # call segmentation script
        print('child gesture labeling for video file: {}' .format(args.input_video_filename))
        
        gesture_segmentation_results, gesture_segmentation_labels = video_segment(data_info.input_video_filename, data_info.images_data_path, data_info.opticalflow_data_path, spatial_cnn_model, motion_cnn_model, spatial_motion_svm_model, device=device, num_flow_channel=num_flow_channel, feature_dimension=feature_dimension, batch_size=args.batch_size, num_workers=args.num_workers, child_gesture_duration=args.child_gesture_duration, feature_pool=args.feature_pool)

        # save the segmentation results
        sio.savemat('' .join([save_result_path, data_info.input_video_filename, '.mat']), {'gesture_segmentation_results':gesture_segmentation_results, 'gesture_segmentation_labels':gesture_segmentation_labels})
        
        
        print('writing segmentation labels and save result as a video file in "{}"' .format(save_result_video_path))
        gesture_handcoded_labels = np.zeros(gesture_segmentation_labels.shape)
        output_video_file = '' .join([save_result_video_path, data_info.input_video_filename, '.', args.output_video_file_format])
        child_gesture_label_on_video(gesture_handcoded_labels, gesture_segmentation_labels, args.input_video_filename, output_video_file, org_modified_child_gesture_dict)
        print('########################################################################################')
        
    # for multiple files
    elif len(args.input_video_folder):
                        
        input_video_file_info = glob.glob('' .join([args.input_video_folder, '*.', args.input_video_file_format]))
        input_video_file_info = sorted(input_video_file_info)
        num_video_files = len(input_video_file_info)
        
        for i, input_video_filename in enumerate(input_video_file_info):
            
            #preprocess the video file
            print('processing "{}" ({}/{})' .format(input_video_filename, i+1, num_video_files))
            print('----------------------------------------------------------------------------------------')
            print('pre-processing video file: {}' .format(input_video_filename))
            data_info = preprocess_video(input_video_filename, images_data_path=args.input_images_folder, opticalflow_data_path=args.input_opticalflow_folder, cache_data_path=args.preprocessed_data_cache_path)

            # call segmentation script
            print('child gesture labeling for video file: {}' .format(input_video_filename))
            gesture_segmentation_results, gesture_segmentation_labels = video_segment(data_info.input_video_filename, data_info.images_data_path, data_info.opticalflow_data_path, spatial_cnn_model, motion_cnn_model, spatial_motion_svm_model, device=device, num_flow_channel=num_flow_channel, feature_dimension=feature_dimension, batch_size=args.batch_size, num_workers=args.num_workers, child_gesture_duration=args.child_gesture_duration, feature_pool=args.feature_pool)
            

            # save the segmentation results
            sio.savemat('' .join([save_result_path, data_info.input_video_filename, '.mat']), {'gesture_segmentation_results':gesture_segmentation_results, 'gesture_segmentation_labels':gesture_segmentation_labels})

            # save result video file
            if args.output_video_save_flag:
                print('writing segmentation labels and save result as a video file in "{}"' .format(save_result_video_path))
                gesture_handcoded_labels = np.zeros(gesture_segmentation_labels.shape)
                output_video_file = '' .join([save_result_video_path, data_info.input_video_filename, '.', args.output_video_file_format])
                child_gesture_label_on_video(gesture_handcoded_labels, gesture_segmentation_labels, input_video_filename, output_video_file, org_modified_child_gesture_dict)
            print('########################################################################################')
            
    else:
        raise ValueError('please give an input video file by "--input_video_filename input_video_file_name" or  "--input_video_folder folder_contains_video_files" \nOR\n')

if __name__ == "__main__":
    main()

