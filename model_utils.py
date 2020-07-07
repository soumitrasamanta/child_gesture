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
from tqdm import tqdm
import numpy as np
import glob

import torch
import torch.nn as nn

from thundersvm import *

from network import *
from data_utils import *

#---------------------------------------------------------------------------------------------
##############################################################################################

def load_resnet_cnn_net_model(input_cnn_net_model_file, num_class=8, num_channel=3, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    """
    Load pre-trained network
    """
    
    # load the base model
    cnn_model = resnet101(pretrained=True, channel=num_channel, nb_classes=8).to(device)

    # load the best trained model
#     checkpoint = torch.load(input_cnn_net_model_file, map_location=device)# python2
    checkpoint = torch.load(input_cnn_net_model_file, map_location=device, encoding='latin1')# python3
    cnn_model.load_state_dict(checkpoint['state_dict'])
    print("loaded '{}' resnet net (epoch {})".format(input_cnn_net_model_file, checkpoint['epoch']))
    
    # prepare motion_cnn_model to get last layer feature representation
    modules = list(cnn_model.children())[:-1]
    cnn_model = nn.Sequential(*modules)
    for param in cnn_model.parameters():
        param.requires_grad = False
    
    del checkpoint, modules
    
    return cnn_model

def thundersvm_load_svm_model(input_svm_model_file):

    """
    Load pre-trained SVM model
    """
    
    svm_model = SVC()
    svm_model.load_from_file(input_svm_model_file)
    
    return svm_model

def thundersvm_test_1v1_genrl(svm_model, feature):
    
    """
    Test pre-trained SVM model
    """
    
    feature = feature.T # thundersvm takes input as #samples_x_#features
    feature = feature.astype(float)
    
    predict_label = svm_model.predict(feature)
    
    return predict_label

def losd_cnn_svm_model(num_class=8, num_flow_channel=10, cnn_svm_model_id=1, feature_pool='max', device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    """
    Load pre-trained CNN (spatial and temporal) and SVM model
    ----------------------------------------------------------------------------
    
    INPUT:
    
    num_class- number of action class
    num_flow_channel- number of optical flow channel has been used for training
    cnn_svm_model_id- ids based on the data partition
    feature_pool- feature pooling ('max' or 'avg'; default- 'max')
    device- CPU or GPU (default- depends on the availability)
    
    ----------------------------------------------------------------------------

    OUTPUT:
    
    spatial_cnn_model- pre-trained cnn spatial feature model
    motion_cnn_model- pre-trained cnn motion feature model
    spatial_motion_svm_model- pre-trained svm model 
    """
#     feature_dimension = 2048 # change this according to which layer's feature you want
    num_gpu_device = torch.cuda.device_count()

    # load spatial cnn net
    input_spatial_cnn_model_file = ''.join(['model/cnn_net/spatial_lucid_08_split_no_', str(cnn_svm_model_id).zfill(2), '_nfch_10_nepochs_500_lr_0.0005_ss_model_best.pth.tar'])
    print('loading spatial cnn trained model from: {}' .format(input_spatial_cnn_model_file))
    spatial_cnn_model = load_resnet_cnn_net_model(input_spatial_cnn_model_file, num_class=num_class, num_channel=3, device=device)
    print('----------------------------------------------------------------------------------------')

    # load motion cnn net
    input_motion_cnn_model_file = ''.join(['model/cnn_net/motion_lucid_08_split_no_', str(cnn_svm_model_id).zfill(2), '_nfch_10_nepochs_500_lr_0.01_ss_model_best.pth.tar'])
    print('loading motion cnn trained model from: {}' .format(input_motion_cnn_model_file))
    motion_cnn_model = load_resnet_cnn_net_model(input_motion_cnn_model_file, num_class=num_class, num_channel=2*num_flow_channel, device=device)
    print('----------------------------------------------------------------------------------------')

    # for svm model
    input_spatial_motion_svm_model_path = ''.join(['model/svm/spatial_motion_lucid_08_split_no_', str(cnn_svm_model_id).zfill(2), '_nfch_10_nepochs_500_slr_0.0005_mlr_0.01_network_resnet101_', str(feature_pool), '_pooling_thundersvm_1v1/'])
    input_spatial_motion_svm_model_file = ''.join([input_spatial_motion_svm_model_path, 'best_model_basedon_train_val_data_nclass_12_kernel_linear_int_c_0.1_max_c_1000.0_num_div_c_46_int_g_0.1_max_g_1.0_num_div_g_10'])
    print('loading spatial and motion net svm model from: "{}"' .format(input_spatial_motion_svm_model_file))
    spatial_motion_svm_model = thundersvm_load_svm_model(input_spatial_motion_svm_model_file)
    print('----------------------------------------------------------------------------------------')
    
    # model paralization (if you have multiple gpus)
    print('model will use "{}" GPUs' .format(num_gpu_device))
    if num_gpu_device > 1:
        spatial_cnn_model = nn.DataParallel(spatial_cnn_model)
        motion_cnn_model = nn.DataParallel(motion_cnn_model)
            
    return spatial_cnn_model, motion_cnn_model, spatial_motion_svm_model

def spatial_feature_score_frames(model, num_class, dict_video_files, int_frame_id, end_frame_id, data_path, num_flow_channel=10, feature_pool='max', batch_size=4, num_workers=2, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    """
    Calculate spatil feature scores 
    """
    
    # switch to evaluate mode
    model.eval()
    
    video_filename = list(dict_video_files.keys())[0]
    
    spatial_data_loader = test_spatial_dataloader(data_path, dict_video_files, int_frame_id=int_frame_id, end_frame_id=end_frame_id, num_flow_channel=num_flow_channel, batch_size=batch_size, num_workers=num_workers, verbose=0)
    temp_data_loader = spatial_data_loader.run()
    
    num_frames = end_frame_id[video_filename] - int_frame_id[video_filename] + 1
    frame_level_scores = np.zeros((num_frames, num_class))
    frame_ids = []
    
#     progress = tqdm(temp_data_loader)
    for i, (fl_name,data,label,fm_id) in enumerate(temp_data_loader):
        
        data = data.to(device)
        
        with torch.no_grad():
                # compute output and loss
                output = model(data)
                
                temp_revised_ids = fm_id - int_frame_id[video_filename]# here -int_frame_id[video_filename] is used to access the frame_level_scores array (as frame_ids is different from frame_level_scores araay index)
                frame_level_scores[temp_revised_ids,:] = output.data.cpu().numpy().squeeze()
                frame_ids.extend(fm_id)
                
    # take entry based on frame_ids             
    temp_revised_ids = sorted(np.asarray(frame_ids)-int_frame_id[video_filename])          
    frame_level_scores = frame_level_scores[temp_revised_ids,:]# here -int_frame_id[video_filename] is used to access the frame_level_scores array (as frame_ids is different from frame_level_scores araay index)
    frame_ids = sorted(frame_ids)# index syncronization 
    
    return frame_level_scores, frame_ids

def motion_feature_score_frames(model, num_class, dict_video_files, int_frame_id, end_frame_id, data_path, num_flow_channel=10, feature_pool='avg', batch_size=4, num_workers=2, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    """
    Calculate motion feature scores 
    """
    
    # switch to evaluate mode
    model.eval()
    
    video_filename = list(dict_video_files.keys())[0]
    
    motion_data_loader = test_motion_dataloader(data_path, dict_video_files, int_frame_id=int_frame_id, end_frame_id=end_frame_id, num_flow_channel=num_flow_channel, batch_size=batch_size, num_workers=num_workers, verbose=0)
    temp_data_loader = motion_data_loader.run()
    
    num_frames = end_frame_id[video_filename] - int_frame_id[video_filename] + 1
    frame_level_scores = np.zeros((num_frames, num_class))
    frame_ids = []
    
#     progress = tqdm(temp_data_loader)
    for i, (fl_name,data,label,fm_id) in enumerate(temp_data_loader):
        
        data = data.to(device)
        
        with torch.no_grad():
                # compute output and loss
                output = model(data)
                
                temp_revised_ids = fm_id - int_frame_id[video_filename]# here -int_frame_id[video_filename] is used to access the frame_level_scores array (as frame_ids is different from frame_level_scores araay index)
                frame_level_scores[temp_revised_ids,:] = output.data.cpu().numpy().squeeze()
                frame_ids.extend(fm_id)
                
    # take entry based on frame_ids             
    temp_revised_ids = sorted(np.asarray(frame_ids)-int_frame_id[video_filename])          
    frame_level_scores = frame_level_scores[temp_revised_ids,:]# here -int_frame_id[video_filename] is used to access the frame_level_scores array (as frame_ids is different from frame_level_scores araay index)
    frame_ids = sorted(frame_ids)# index syncronization 
    
    return frame_level_scores, frame_ids

def feature_pooling(features, pooling_optr='max'):
    
    """
    Feature pooling 
    """
    
    if pooling_optr == 'max':
        features = features.max(axis=1)
    elif pooling_optr == 'avg':
        features = features.mean(axis=1)
    else:
        raise ValueError('please check the pooling operation')

    return features

def video_segment(video_filename, images_data_path, opticalflow_data_path, spatial_cnn_model, motion_cnn_model, 
                     spatial_motion_svm_model, device, num_flow_channel=10, feature_dimension=2048, child_gesture_duration=45,
                     batch_size=2, num_workers=2, image_file_format='jpg', feature_pool='max'):
    
    """
    Segment a long video with predefined child gesture 
    """
    
    num_frames = len(glob.glob('' .join([opticalflow_data_path, 'u/', video_filename, '/*.', image_file_format])))

    dict_video_file = {video_filename: 1}# here just put 1 for dummy labels as it will not used for segmentation
    min_num_extra_frames = num_frames - child_gesture_duration
    gesture_int_frame_ids = list(range(1, min_num_extra_frames, child_gesture_duration))
    gesture_int_frame_ids.append(min_num_extra_frames+1)

    gesture_segmentation_results = np.ones((3, len(gesture_int_frame_ids)))
    gesture_segmentation_labels = -np.ones(num_frames)
    for i, nf in enumerate(tqdm(gesture_int_frame_ids)):

        int_frame_id = {video_filename: nf}
        end_frame_id = {video_filename: nf+child_gesture_duration-1}
    #                     print(int_frame_id, end_frame_id)

        # calculate spatial feature
        temp_spatial_feature, _ = spatial_feature_score_frames(spatial_cnn_model, feature_dimension, dict_video_file, int_frame_id, end_frame_id, images_data_path, num_flow_channel=num_flow_channel, feature_pool=feature_pool, batch_size=batch_size, num_workers=num_workers, device=device)
        # feature pooling
        temp_spatial_feature = feature_pooling(temp_spatial_feature.T, feature_pool)
        temp_spatial_feature = temp_spatial_feature.reshape(-1, 1)
        #feature normalize
        temp_spatial_feature = lp_normalize_feature(temp_spatial_feature, 'l2')

        # calculate motion feature
        temp_motion_feature, _ = motion_feature_score_frames(motion_cnn_model, feature_dimension, dict_video_file, int_frame_id, end_frame_id, opticalflow_data_path, num_flow_channel=num_flow_channel, feature_pool=feature_pool, batch_size=batch_size, num_workers=num_workers, device=device)
        # feature pooling
        temp_motion_feature = feature_pooling(temp_motion_feature.T, feature_pool)
        temp_motion_feature = temp_motion_feature.reshape(-1, 1)
        #feature normalize
        temp_motion_feature = lp_normalize_feature(temp_motion_feature, 'l2')

        temp_combined_feature = np.concatenate((temp_spatial_feature, temp_motion_feature), axis=0)
        non_zeros_feat_ids = np.abs(temp_combined_feature).sum(axis=0)>0.0
        temp_combined_feature = temp_combined_feature[:,non_zeros_feat_ids]

        # classify gesture using SVM
        if temp_combined_feature.size:
            temp_pred_label = thundersvm_test_1v1_genrl(spatial_motion_svm_model, temp_combined_feature.reshape(-1, 1))
            gesture_segmentation_labels[int_frame_id[video_filename]:end_frame_id[video_filename]+1] = temp_pred_label
            gesture_segmentation_results[:, i] = np.array([int_frame_id[video_filename], end_frame_id[video_filename], temp_pred_label])
            
            
    return gesture_segmentation_results, gesture_segmentation_labels




