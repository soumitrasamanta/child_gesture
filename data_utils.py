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

from PIL import Image
import random
import os
import cv2
import skimage
import skimage.transform
import shutil
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch

import pyflow

#---------------------------------------------------------------------------------------------
##############################################################################################

def denominator_check(val):
    
    """
    Denominator check to hande zeros
    
    """
    
    if isinstance(val, (list, tuple, np.ndarray)):
         val[np.abs(val) == 0.0] = np.finfo(float).eps
    else:
        if val == 0:
            val = np.finfo(float).eps   
        
    return val

def lp_normalize_feature(X, l_norm): 
    
    """
    Feature normalize with l_p norm
    """
    
    lp = int(l_norm[1:])
    Temp1 = np.reshape(np.sum(X**lp, axis=0)**(1.0/lp),(1, X.shape[1]))
    normat_vector = X/denominator_check(Temp1)
    
    return normat_vector

def folder_create(folder_name):
    
    """
    Create a folder 
    """
    
    if(len(folder_name)):
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        return folder_name
    

def images_write(images, save_path = 'Temp_image_save/', file_name = 'img', file_extention = 'jpg', file_number = 0, decimal_file_number = 5):

    """
    Write images in a folder with specified format
    ----------------------------------------------------------------------------
    
    INPUT:
    
    images- a list of images
    save_path- destination folder
    file_name- starting file name
    file_extention- image file format
    file_number- file number starting index
    decimal_file_number- decimal places for file number indexing
    ----------------------------------------------------------------------------
    
    OUTPUT:
    Retun 1 , its just for confirmation
    
    """
    
    save_path = folder_create(save_path)
    num_images = len(images)
    if(num_images > 1):
        for nim in range(num_images):
            Temp_image = images[nim].astype('uint8')
            if(np.ndim(Temp_image) == 3):
                Temp_image = Temp_image[:,:,[2, 1, 0]];#21-06-17
            Temp_file_number = '%'+str(decimal_file_number)+'.'+str(decimal_file_number)+'d'
            cv2.imwrite(save_path+file_name+Temp_file_number%(nim+file_number)+'.'+file_extention, Temp_image)#21-06-17
    else:
        Temp_image = images[0].astype('uint8')
        if(np.ndim(Temp_image) == 3):
            Temp_image = Temp_image[:,:,[2, 1, 0]];#21-06-17
        Temp_file_number = '%'+str(decimal_file_number)+'.'+str(decimal_file_number)+'d'
        cv2.imwrite(save_path+file_name+Temp_file_number%file_number+'.'+file_extention, Temp_image)#21-06-17

    return(1)


def video2images_large(input_file=[], int_frame_pos_flag='index', int_frame_pos=[], end_frame_pos=[], save_path='Temp_image_save/', file_name='img', file_extention='jpg', file_number=0, decimal_file_number=5):
    
    """
    Convert a large video to image frames and save in a specified folder
    ----------------------------------------------------------------------------
    
    INPUT:
    
    input_file- input video file
    int_frame_pos_flag- index to read the video file (initial frame ids), otherwise it will read based on time 
    int_frame_pos- initial frame position
    end_frame_pos- end frame position
    save_path- image destination folder
    file_name- starting file name
    file_extention- image file format
    file_number- file number starting index
    decimal_file_number- decimal places for file number indexing
    ----------------------------------------------------------------------------
    
    OUTPUT:
    NA
    
    """
        
    save_path = folder_create(save_path)
    Temp_file_number = '%{}.{}d' .format(str(decimal_file_number), str(decimal_file_number))
    
    if os.path.isfile(input_file):
        
        # read input video informations
        video_reader = cv2.VideoCapture(input_file) # call video reader
        frame_rate = video_reader.get(5) # frame rate in the video file
        row, col = int(video_reader.get(4)), int(video_reader.get(3)) # frame size in the video file
        
        if int_frame_pos_flag=='index': # read video based on specified frame index information
            if not int_frame_pos:
                int_frame_pos = 0
            if not end_frame_pos:
                end_frame_pos = video_reader.get(7)
            num_frames = int(end_frame_pos - int_frame_pos)#.astype(int)
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, int_frame_pos)

        else: # read video based on specified time (millisecond) information
            if not int_frame_pos:
                int_frame_pos = 0
            if not end_frame_pos:
                end_frame_pos = (1000.*video_reader.get(7))/frame_rate
            num_frames = np.floor(frame_rate*((end_frame_pos-int_frame_pos)/1000.)).astype(int)
            video_reader.set(cv2.CAP_PROP_POS_MSEC, int_frame_pos)

        # read video frame by frame
        count = 0
        while((video_reader.isOpened()) and (count<num_frames)) :
            ret, frame = video_reader.read()
            if ret==True:
                Temp_image_file_name = '{}{}{}.{}' .format(save_path, file_name, Temp_file_number%(count+file_number), file_extention)
                cv2.imwrite(Temp_image_file_name, frame)
                count = count + 1
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        video_reader.release() # release the video reader
        cv2.destroyAllWindows()

    else:
        raise ValueError('No such file {} does not exits!!' .format(input_file))

    
def video_resize(input_file=[], int_frame_pos_flag='index', int_frame_pos=[], end_frame_pos=[], resize_scale=[], resize_size=[], output_file=[], output_file_format='avi', compression_method='mp4v'):
    
    """
    Resize a video frames in a long video
    ----------------------------------------------------------------------------
    
    INPUT:
    
    input_file- input video file
    int_frame_pos_flag- index to read the video file (initial frame ids), otherwise it will read based on time 
    int_frame_pos- initial frame position
    end_frame_pos- end frame position
    resize_scale- frame resize scale
    resize_size- frame resize size (if not scale used)
    output_file- output/ resized video file name
    output_file_format- resized video file format
    compression_method- resize video compression method
    ----------------------------------------------------------------------------
    
    OUTPUT:
    NA
    """
        
    if os.path.isfile(input_file):
        
        # read input video informations
        video_reader = cv2.VideoCapture(input_file) # call video reader
        frame_rate = video_reader.get(5) # frame rate in the video file
        row, col = int(video_reader.get(4)), int(video_reader.get(3)) # frame size in the video file
        
        # for resize video
        if resize_scale:
            resize_size =(np.floor(row*resize_scale), np.floor(col*resize_scale))
        #fourcc = cv2.cv.CV_FOURCC(*compression_method)#FOR OLDER VERSION OF OPENCV
        fourcc = cv2.VideoWriter_fourcc(*compression_method) #FOR NEW VERSION OF OPENCV
        if not output_file:
            output_file = input_file
            output_file = '{}resize_{}.{}' .format(output_file[:-len(output_file.split('/')[-1])], output_file.split('/')[-1].split('.')[0], output_file_format)
        folder_create(output_file[:-len(output_file.split('/')[-1])])
            
        video_writer = cv2.VideoWriter(output_file, fourcc, frame_rate, (resize_size[1], resize_size[0]))
        
        if int_frame_pos_flag=='index': # read video based on specified frame index information
            if not int_frame_pos:
                int_frame_pos = 0
            if not end_frame_pos:
                end_frame_pos = video_reader.get(7)
            num_frames = int(end_frame_pos - int_frame_pos)#.astype(int)
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, int_frame_pos)

        else: # read video based on specified time (millisecond) information
            if not int_frame_pos:
                int_frame_pos = 0
            if not end_frame_pos:
                end_frame_pos = (1000.*video_reader.get(7))/frame_rate
            num_frames = np.floor(frame_rate*((end_frame_pos-int_frame_pos)/1000.)).astype(int)
            video_reader.set(cv2.CAP_PROP_POS_MSEC, int_frame_pos)

        count = 0
        while((video_reader.isOpened()) and (count<num_frames)) :
            ret, frame = video_reader.read()
            if ret==True:
                image = frame[:,:,[2, 1, 0]].copy() # convert into RGB form 21-06-17
                image = (255*skimage.transform.resize(image, resize_size, mode='constant')).astype('uint8')
                image = image[:,:,[2, 1, 0]]#21-06-17
                
                video_writer.write(image)
                count = count + 1
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        video_reader.release() # release the video reader
        video_writer.release() # release the video writer 
        cv2.destroyAllWindows()

    else:
        raise ValueError('No such file {} does not exits!!' .format(input_file))

    
def optical_flow_large(input_video_file, int_frame_pos_flag='index', int_frame_pos=[], end_frame_pos=[], save_path='optical_flow/', output_optical_flow_file=[], optical_flow_cap_x=[-20., 20.], optical_flow_cap_y=[-20., 20.], resize_factor=[], colType=0):
    
    """
    Calculate optical flow in long video and svaed the flow as image files
    ----------------------------------------------------------------------------
    
    INPUT:
    
    input_video_file- input video file
    int_frame_pos_flag- index to read the video file (initial frame ids), otherwise it will read based on time 
    int_frame_pos- initial frame position
    end_frame_pos- end frame position
    save_path- optical flow image destination folder
    output_optical_flow_file- optical flow file name
    optical_flow_cap_x- flow (x) cap to reduce noise
    optical_flow_cap_y- flow (y) cap to reduce noise
    resize_factor- flow image rezise factor
    colType- 
    ----------------------------------------------------------------------------
    
    OUTPUT:
    NA
    """
    
    #---------------------------------------------------------------------------------------------
    # Flow parameters:(default)
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    #---------------------------------------------------------------------------------------------
    if os.path.isfile(input_video_file):
        
        # read input video informations
        video_reader = cv2.VideoCapture(input_video_file) # call video reader
        frame_rate = video_reader.get(5) # frame rate in the video file
        row, col = int(video_reader.get(4)), int(video_reader.get(3)) # frame size in the video file
        
        if int_frame_pos_flag=='index': # read video based on specified frame index information
            if not int_frame_pos:
                int_frame_pos = 0
            if not end_frame_pos:
                end_frame_pos = video_reader.get(7)
            num_frames = int(end_frame_pos - int_frame_pos)#.astype(int)
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, int_frame_pos)

        else: # read video based on specified time (millisecond) information
            if not int_frame_pos:
                int_frame_pos = 0
            if not end_frame_pos:
                end_frame_pos = (1000.*video_reader.get(7))/frame_rate
            num_frames = np.floor(frame_rate*((end_frame_pos-int_frame_pos)/1000.)).astype(int)
            video_reader.set(cv2.CAP_PROP_POS_MSEC, int_frame_pos)
        #--------------------------------------------------------------------------------------------- 
        
        # for optical flow save path
        if not output_optical_flow_file:
            output_optical_flow_file = input_video_file
            output_optical_flow_file = output_optical_flow_file.split('/')[-1].split('.')[0]
        save_path_x_flow = '{}u/{}/'.format(save_path, output_optical_flow_file)
        save_path_y_flow = '{}v/{}/'.format(save_path, output_optical_flow_file)
        #---------------------------------------------------------------------------------------------
        
        # read video frame by frame
        count = 1
        ret_1, frame_1 = video_reader.read()
        if len(resize_factor):
            frame_1 = frame_1[:,:-1,:]
        frame_1 = frame_1.astype(float) / 255.
        while((video_reader.isOpened()) and (count<num_frames)) :
            ret_2, frame_2 = video_reader.read()
            if ret_2==True:
                if len(resize_factor):
                    frame_2 = frame_2[:,:-1,:]
                frame_2 = frame_2.astype(float) / 255.
                #---------------------------------------------------------------------------------------------
                
                # call optical flow API
                flow_x, flow_y, im2W = pyflow.coarse2fine_flow(frame_1, frame_2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
    nSORIterations, colType)
                #---------------------------------------------------------------------------------------------
                
                # cap the x-flow (change cap param optical_flow_cap_x accordingly)
                flow_x[flow_x<optical_flow_cap_x[0]] = optical_flow_cap_x[0]# for minimum
                flow_x[flow_x>optical_flow_cap_x[1]] = optical_flow_cap_x[1]# for maximum
                # rescale the flow to save as image form
                flow_x = (255*((flow_x-optical_flow_cap_x[0])/denominator_check(optical_flow_cap_x[1]-optical_flow_cap_x[0]))).astype('uint8')
                # save the cap flow as images
                images_write([flow_x], save_path=save_path_x_flow, file_name='frame', file_number=count, decimal_file_number=6)
                #---------------------------------------------------------------------------------------------
                
                # cap the y-flow (change cap param optical_flow_cap_y accordingly)
                flow_y[flow_y<optical_flow_cap_y[0]] = optical_flow_cap_y[0]# for minimum
                flow_y[flow_y>optical_flow_cap_y[1]] = optical_flow_cap_y[1]# for maximum
                # rescale the flow to save as image form
                flow_y = (255*((flow_y-optical_flow_cap_y[0])/denominator_check(optical_flow_cap_y[1]-optical_flow_cap_y[0]))).astype('uint8')
                # save the cap flow as images
                images_write([flow_y], save_path=save_path_y_flow, file_name='frame', file_number=count, decimal_file_number=6)
                #---------------------------------------------------------------------------------------------
                
                frame_1 = frame_2.copy()
                count = count + 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
            
                
        video_reader.release() # release the video reader
        cv2.destroyAllWindows()

    else:
        raise ValueError('No such file {} does not exits!!' .format(input_video_file))
        

class preprocess_video():
    
    """
    
    """
    
    def __init__(self, input_video_filename, images_data_path='', opticalflow_data_path='', cache_data_path='temp_cache/'):
        
        self.video_data_path = input_video_filename[:-len(input_video_filename.split('/')[-1])]
        self.input_video_file_format = input_video_filename.split('/')[-1].split('.')[-1]
        self.input_video_filename = input_video_filename.split('/')[-1].split('.')[0]
        
        self.images_data_path = images_data_path
        self.opticalflow_data_path = opticalflow_data_path
        
        self.revised_video_size = (256, 342)
        self.decimal_file_number = 6
        self.output_image_file_format = 'jpg'
        self.output_video_file_format = 'mp4'
        self.optical_flow_cap_x = [-20., 20.]
        self.optical_flow_cap_y = [-20., 20.]
        
        self.cache_data_path = folder_create(cache_data_path)
        self.output_video_filename = '' .join([self.cache_data_path, 'video/', self.input_video_filename, '.', self.output_video_file_format])
        self.output_image_path = '' .join([self.cache_data_path, 'images/', self.input_video_filename, '/'])
        self.output_optical_flow_path = '' .join([self.cache_data_path, 'optical_flow/'])
        
        self.video_properties()
        self.video_scaling()
        self.video2images()
        self.video_opticalflow()
                
        
    def video_properties(self):
        
        self.video_prop = {}
        
        temp_video_file = '' .join([self.video_data_path, self.input_video_filename, '.', self.input_video_file_format]) 
        if os.path.isfile(temp_video_file):
            
            video_reader = cv2.VideoCapture(temp_video_file) # call video reader

            self.video_prop['row'], self.video_prop['col'] = int(video_reader.get(4)), int(video_reader.get(3))# video frame size
            self.video_prop['frame_rate'] = video_reader.get(5) # frame rate in the video file
            self.video_prop['num_frames'] = int(video_reader.get(7))# number of frames in the video

            video_reader.release() # release the video reader

        else:
            raise ValueError('No such file {} exits!!' .format(temp_video_file))

    
    def video_scaling(self):
        
        # scaling video file
        temp_video_file = '' .join([self.video_data_path, self.input_video_filename, '.', self.input_video_file_format])
        
        if((self.video_prop['row']>self.revised_video_size[0]) and (self.video_prop['col']>self.revised_video_size[1])):
            print('video scaling file : "{}" with resize factor: "{}"' .format(temp_video_file, self.revised_video_size))
            if os.path.isfile(self.output_video_filename):
                print('scaling video "{}" already exits' .format(self.output_video_filename))
            else:
                video_resize(input_file=temp_video_file, resize_size=self.revised_video_size, output_file=self.output_video_filename)
        else:
            print('GOOD video file: "{}" is already scaled' .format(self.output_video_filename))
            shutil.copy2(temp_video_file, self.output_video_filename)
            self.video_data_path = self.output_video_filename[:-len(self.output_video_filename.split('/')[-1])]
            
        print('----------------------------------------------------------------------------------------')
        
        
    def video2images(self):
        
        # transform video to images
        if len(self.images_data_path):
            print('GOOD: you have given images stored at "{}" converted from the input video "{}"' .format(self.images_data_path, self.output_video_filename))
        else:
            print('transform video : "{}" to images will be stored at : "{}"' .format(self.output_video_filename, self.output_image_path))
            video2images_large(input_file=self.output_video_filename, save_path=self.output_image_path, file_name='frame', file_extention=self.output_image_file_format, file_number=1, decimal_file_number=self.decimal_file_number)
            self.images_data_path = self.output_image_path[:-(len(self.input_video_filename)+1)]
        print('----------------------------------------------------------------------------------------')

        
    def video_opticalflow(self):
        
        # calculate optical flow
        if len(self.opticalflow_data_path):
            print('GOOD: you have given optical flow for the input video')
        else:
            print('optical flow calculation for video : "{}" ' .format(self.output_video_filename))
            optical_flow_large(self.output_video_filename, save_path=self.output_optical_flow_path, output_optical_flow_file=self.input_video_filename.split('/')[-1].split('.')[0], optical_flow_cap_x=self.optical_flow_cap_x, optical_flow_cap_y=self.optical_flow_cap_y, resize_factor=[1])
            self.opticalflow_data_path = self.output_optical_flow_path
        print('----------------------------------------------------------------------------------------')

        
def child_gesture_label_on_video(gesture_handcoded_labels, gesture_segmentation_labels, input_video_file, output_video_file, org_modified_child_gesture_dict={}):
    
    """
    Labels child gestures in a video 
    """

    if not org_modified_child_gesture_dict:
        org_modified_child_gesture_dict = {1:'grasp object', 2:'give', 3:'hold out', 4:'lower object', 5:'object manipulation', 6:'other', 
                                  7:'point-declarative', 8:'point-imperative', 9:'reaches-imperative', 10:'reaches-declarative', 
                                  11:'retract object', 12:'share orientation'}

    # convert the label file to actual gesture name
    num_label_frames = gesture_segmentation_labels.shape[0]
    text = [None]*num_label_frames
    text_pos = [None]*num_label_frames
    max_text_length = 0
    for i in range(num_label_frames):

        org_label = int(gesture_handcoded_labels[i])
        pred_label = int(gesture_segmentation_labels[i])

        org_seg_text = 'h-coded: '

        if org_label>0:
            org_seg_text = '' .join(['h-coded: ', '(',str(org_label),')', org_modified_child_gesture_dict[org_label]])
        else:
            org_seg_text = 'h-coded: '
        if pred_label>0:
            pred_seg_tex = '' .join(['m-coded: ', '(',str(pred_label),')', org_modified_child_gesture_dict[pred_label]])
        else:
            pred_seg_tex = 'm-coded: '

        if(max_text_length < max(len(org_seg_text), len(pred_seg_tex))):
            max_text_length = max(len(org_seg_text), len(pred_seg_tex))

        text[i] = ['f#: {}' .format(i), org_seg_text, pred_seg_tex]
        text_pos[i] = np.array([[10, 10, 10],[10, 30, 50]])

    # call text writing api
    text_color = np.array([[255, 20, 20],[20, 255, 20], [20, 20, 255]])
    draw_textOnVideo(input_file=input_video_file, text=text, text_pos=text_pos, max_text_length=max_text_length, text_color=text_color, output_file=output_video_file, output_file_format='flv')

def images_concatenate(images, axis=0, border_handle_color=255):
    
    """
    Concatenate multiple images along any axis
    ----------------------------------------------------------------------------
    
    INPUT:
    
    images- a list of images which will be concatenate
    axis- axis to be concatenate
    border_handle_color- border pixel values
    ----------------------------------------------------------------------------
    
    OUTPUT:
    images- concatenated image
    """
    
    images = list(images)
    num_images = len(images)
    
    # concatenate along rows
    if axis==0: 
        max_col_size = images[0].shape[1]
        rgb_flag = np.ndim(images[0])
        for i in range(num_images):
            if max_col_size < images[i].shape[1]:
                max_col_size = images[i].shape[1]
            if rgb_flag < np.ndim(images[i]):
                rgb_flag = np.ndim(images[i])
            
        for i in range(num_images):
            T_image = (border_handle_color*np.ones((images[i].shape[axis], max_col_size-images[i].shape[1], rgb_flag))).astype('uint8')  
            images[i] = np.concatenate((images[i], T_image), axis=1)
    
    # concatenate along columns
    elif axis==1: 
        
        max_row_size = images[0].shape[0]
        rgb_flag = np.ndim(images[0])
        for i in range(num_images):
            if max_row_size < images[i].shape[0]:
                max_row_size = images[i].shape[0]
            if rgb_flag < np.ndim(images[i]):
                rgb_flag = np.ndim(images[i])
            
        for i in range(num_images):
            # handle border 
            T_image = (border_handle_color*np.ones((max_row_size - images[i].shape[0], images[i].shape[axis], rgb_flag))).astype('uint8')  
            # handle gray scale images
            if(np.ndim(images[i])!=rgb_flag):
                images[i] = np.reshape(images[i], (images[i].shape[0], images[i].shape[1], 1))
                images[i] = np.concatenate((images[i], images[i], images[i]), axis=2)
            images[i] = np.concatenate((images[i], T_image), axis=0)
        
    else:
        raise ValueError('please check the image dimension :{} (only take 0 or 1) which you want to concatenate\n' .format(axis))
    
    images = tuple(images)
    images = np.concatenate(images, axis=axis)
    
    return images 

def draw_textOnImage(image, text, text_pos, text_font_size=2, text_color=[], text_width=2, verbose_flag=0):
    
    """
    Draw text in an image
    ----------------------------------------------------------------------------
    
    INPUT:
    
    image- input image for text draw
    text- input text
    text_pos- text position in the input image
    text_font_size- text font size
    text_color- text color
    text_width- text width
    verbose_flag
    ----------------------------------------------------------------------------
    
    OUTPUT:
    image- text drawn image 
    """
    
    text_pos = text_pos.astype(int)
    num_text = np.size(text)
    if num_text > text_pos.shape[1]:
        if verbose_flag:
            print('WARNING:NUMBER OF TEXT (%d) AND ITS POSITION (%d)IS NOT SAME \n'%(num_text, text_pos.shape[1]))
            print('WARNING:TAKING (%s) FOR ALL THE POSITIONS\n'%(text[0]))
        
        text_pos = np.append(text_pos, np.reshape(text_pos[:,-1], (text_pos.shape[0], 1))*np.ones((text_pos.shape[0], num_text-text_pos.shape[1])), axis=1)
    if np.size(text_font_size) == 0:
        text_font_size = 2*np.ones((num_text), np.int)
    elif np.size(text_font_size) == 1:
        text_font_size = text_font_size*np.ones((num_text), np.int)
    else:
        if (np.size(text_font_size) != num_text):
            text_font_size = text_font_size*np.ones((num_text), np.int)
    if len(text_color) == 0:
        text_color = ss_color_generation(num_text)
        text_color = text_color.astype(int)
    else:
        if ((text_color.shape[1] == 1) & (text_color.shape[1] != num_text)):
            text_color = text_color*np.ones((3, num_text), np.int)
    if np.size(text_width) == 0:
        text_width = 2*np.ones((num_text), np.int)
    elif np.size(text_width) == 1:
        text_width = text_width*np.ones((num_text), np.int)
    else:
        if (np.size(text_width) != num_text):
            text_width = text_width*np.ones((num_text), np.int)
    for ntxt in range(num_text):
        font = cv2.FONT_HERSHEY_SIMPLEX        
        cv2.putText(image, text[ntxt], tuple(np.reshape(text_pos[:,ntxt],(2,1))), font, text_font_size[ntxt], tuple([int(x) for x in text_color[:, ntxt]]), text_width[ntxt], cv2.LINE_AA)
        
    return image

    
def create_imageWithText(text, row=[], column=[], text_pos=np.array([]), text_font_size=1, text_color=[], text_width=2, image_background=np.array([[255],[255],[255]])):
    
    """
    Create an image with specified text
    ----------------------------------------------------------------------------
    
    INPUT:
    
    text- input text
    row- image row size
    column- image column size
    text_pos- text position in the input image
    text_font_size- text font size
    text_color- text color
    text_width- text width
    image_background- image background color
    ----------------------------------------------------------------------------
    
    OUTPUT:
    image- text drawn image 
    """

    num_text = np.size(text)
    
    #find the maximum text length to esrimate the row and column of the image
    if(np.size(row)==0 or row==0):
        row = int((30*text_font_size)*(num_text+2));
    if(np.size(column)==0 or column==0):
        max_text_length = len(text[0])
        for i in range(num_text):
            if max_text_length < len(text[i]):
                max_text_length = len(text[i])
        column = int((17*text_font_size)*max_text_length)
    
    # for text position
    num_text_pos = np.size(text_pos)
    if(num_text_pos==0 or num_text > num_text_pos):
        text_pos = np.zeros((2, num_text))
        for i in range(num_text):
            text_pos[0,i] = 20*text_font_size
            text_pos[1,i] = (35*text_font_size)*(i+1)
    text_pos = text_pos.astype(int)
    
    image = np.reshape(image_background, (1, 1, 3))*np.ones((row, column, 3), np.uint8).astype('uint8')
    image = image.astype('uint8')
    image = draw_textOnImage(image, text, text_pos, text_font_size=text_font_size, text_color=text_color, text_width=text_width)
  
    return image

def draw_textOnVideo(input_file=[], int_frame_pos_flag='index', int_frame_pos=[], end_frame_pos=[], text=['test'], text_pos=[np.array([[5], [20]])], max_text_length=40, text_font_size=0.6, text_color=np.array([[255],[25],[0]]), text_width=2, output_file=[], output_file_format='mp4', compression_method='mp4v', draw_within_flag=0):
    
    """
    Draw text in a video
    ----------------------------------------------------------------------------
    
    INPUT:
    input_file- input video file
    int_frame_pos_flag- index to read the video file (initial frame ids), otherwise it will read based on time 
    int_frame_pos- initial frame position
    end_frame_pos- end frame position
    text- input text
    text_pos- text position in the input image
    max_text_length- maximum text length 
    text_font_size- text font size
    text_color- text color
    text_width- text width
    output_file- output/ resized video file name
    output_file_format- resized video file format
    compression_method- resize video compression method
    ----------------------------------------------------------------------------
    
    OUTPUT:
    
    NA
    """
        
    if os.path.isfile(input_file):
        
        # read input video informations
        video_reader = cv2.VideoCapture(input_file) # call video reader
        frame_rate = video_reader.get(5) # frame rate in the video file
        row, col = np.int(video_reader.get(4)), np.int(video_reader.get(3)) # frame size in the video file
        
        if int_frame_pos_flag=='index': # read video based on specified frame index information
            if not int_frame_pos:
                int_frame_pos = 0
            if not end_frame_pos:
                end_frame_pos = video_reader.get(7)
            num_frames = int(end_frame_pos - int_frame_pos)#.astype(int)
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, int_frame_pos)
        else: # read video based on specified time (millisecond) information
            if not int_frame_pos:
                int_frame_pos = 0
            if not end_frame_pos:
                end_frame_pos = (1000.*video_reader.get(7))/frame_rate
            num_frames = np.floor(frame_rate*((end_frame_pos-int_frame_pos)/1000.)).astype(int)
            video_reader.set(cv2.CAP_PROP_POS_MSEC, int_frame_pos)
        
        # for marker video
        if not output_file:
            output_file = input_file
            output_file = '{}marker_{}.{}' .format(output_file[:-len(output_file.split('/')[-1])], output_file.split('/')[-1].split('.')[0], output_file_format)
        folder_create(output_file[:-len(output_file.split('/')[-1])])
        
        # for marker video size
        image = np.zeros((row, col, 3))
        if draw_within_flag == 0:
            text_image = create_imageWithText(text[0], column=int((17*text_font_size)*max_text_length), text_font_size=text_font_size, text_color=text_color, text_width=text_width)
            image = images_concatenate((text_image, image), axis=0)
        marker_video_size = image.shape
        
        #fourcc = cv2.cv.CV_FOURCC(*compression_method)#FOR OLDER VERSION OF OPENCV
        fourcc = cv2.VideoWriter_fourcc(*compression_method) #FOR NEW VERSION OF OPENCV
        video_writer = cv2.VideoWriter(output_file, fourcc, frame_rate, (marker_video_size[1], marker_video_size[0]))
        
        if num_frames:
            # adjust the number of text and its position with the number of frames
            num_text = len(text)
            if num_text < num_frames:
                text.extend([text[-1]]*(num_frames-num_text))
            num_text_pos = len(text_pos)
            if num_text_pos < num_frames:
                text_pos.extend([text_pos[-1]]*(num_frames-num_text_pos))                 
                
            # main part (video read and manipulation)   
            count = 0
            while((video_reader.isOpened()) and (count<num_frames)) :
                ret, frame = video_reader.read()
                if ret==True:
                    image = frame[:,:,[2, 1, 0]].copy() # convert into RGB form 
                    if draw_within_flag:
                        image = draw_textOnImage(image, text[count], text_pos[count], text_font_size=text_font_size, text_color=text_color, text_width=text_width)
                    else:
                        text_image = create_imageWithText(text[count], column=marker_video_size[1], text_font_size=text_font_size, text_color=text_color)
                        image = images_concatenate((text_image, image), axis=0)
                    image = image[:,:,[2, 1, 0]] #convert into BGR form
                    
                    video_writer.write(image)
                    count = count + 1
                    #print(count)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break

            video_reader.release() # release the video reader
            video_writer.release() # release the video writer 
            cv2.destroyAllWindows()

    else:
        raise ValueError('No such file {} does not exits!!' .format(input_file))

        
class test_spatial_dataset(Dataset): 
    
    """
    Load spatial data (image frames) for data loader
    ----------------------------------------------------------------------------
    
    INPUT:
    
    mode- define the data loader mode (at present its only in "test" mode)
    dict_video_clips- a dictionary of video clips (where key is the video (clip) file name and value is the starting frame number)
    data_path- path to the rgb video frames (images) (same as in test_spatial_dataloader())
    img_rows- rgb video frame (image) row size (it should be 224 as during training it was used)(same as in test_spatial_dataloader())
    img_cols- rgb video frame (image) column size (it should be 224 as during training it was used)(same as in test_spatial_dataloader())
    num_flow_channel- number of flow channel will be used to test a clip (it should be 10 as during training it was used)
    data_transform- fransformation for data augmentation (like image resize, flip etc.)
    image_file_format- rgb frame (image) file format
    ----------------------------------------------------------------------------
    
    OUTPUT:
    
    video_clips- video clips name
    data- stacked rgb video frames (images)
    label- label of the video clip 
    frame_ids- starting frame number (ids) of the video clip
    
    """
    
    def __init__(self, mode, dict_video_clips, data_path, img_rows=224, img_cols=224, num_flow_channel=10, data_transform=None, image_file_format='jpg'):
        
        self.mode = mode
        self.keys = list(dict_video_clips.keys())
        self.values = list(dict_video_clips.values())
        self.data_path = data_path
        
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.num_flow_channel = num_flow_channel
        
        self.data_transform = data_transform
        self.image_file_format = image_file_format
        

    def load_image(self):
        
        
        idx = str(self.clips_idx)
        
        frame_idx = '' .join(['frame', idx.zfill(6)])
        image_filename = '' .join([self.data_path, self.video_clips, '/', frame_idx, '.', self.image_file_format])
        img = Image.open(image_filename)
        img_transform = self.data_transform(img)
        img.close()  

        return img_transform
    

    def __len__(self):
        
        return len(self.keys)
    

    def __getitem__(self, idx):
        
        if self.mode == 'test':
            self.video_clips, self.clips_idx = self.keys[idx].split('-')
            self.clips_idx = int(self.clips_idx)
#             self.clips_idx = int(self.clips_idx + random.randint(1, self.num_flow_channel))# random samples
        else:
            raise ValueError('please define your mode here')

        label = int(self.values[idx]) - 1 
        data = self.load_image()
        
        if self.mode == 'test':
            sample = (self.video_clips, data, label, self.clips_idx)
        else:
            raise ValueError('please define your mode "{}" here' .format(self.mode))
            
        return sample
    
    
class test_spatial_dataloader():
    
    """
    Spation (image frames) dataloader for test video data
    ----------------------------------------------------------------------------
    
    INPUT:
    
    data_path- path to the rgb video frames (images)
    dict_test_video_files- video files as a dictionary (where key will be file name and value will be class label, keep in mind that labels should be [1, 2, ...12] (for unknown class label just put a random label))
    int_frame_id- starting frame of the test video (as you might want to test a video from a particular frame no)
    end_frame_id- end frame of the test video (as you might want to test a video upto a particular frame no)
    img_rows-  rgb video frame (image) row size (it should be 224 as during training it was used)
    img_cols- rgb video frame (image) column size (it should be 224 as during training it was used)
    num_flow_channel- number of flow channel will be used to test a clip (it should be 10 as during training it was used)
    image_file_format- rgb frame (image) file format
    batch_size- for batch data loader 
    num_workers- numbert of workers (CPU-core) will be used during data loader
    ----------------------------------------------------------------------------

    OUTPUT:
    
    test_dataloader- test rgb video frames (images) dataloader

    """
    
    def __init__(self, data_path, dict_test_video_files, int_frame_id, end_frame_id, img_rows=224, img_cols=224, num_flow_channel=10, image_file_format='jpg', batch_size=4, num_workers=2, verbose=0):
        
        self.data_path = data_path
        self.dict_test_video_files = dict_test_video_files
        self.int_frame_id = int_frame_id
        self.end_frame_id = end_frame_id
        self.num_flow_channel = num_flow_channel
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.image_file_format = image_file_format
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.verbose = verbose
        
        
    def run(self):
        
        if self.verbose:
            print('----------------------------------------------------------------------------------------')
                
        # get the test video files
        self.get_test_files_dict()
        
        # get the test images
        self.get_test_image_data(mode='test')
        
        if self.verbose:
            print('########################################################################################')
        
        return self.test_dataloader
    
    
    def get_test_files_dict(self):
        
        self.dict_test_video_clips = {}
        
        for video in self.dict_test_video_files:
            num_frames = self.end_frame_id[video] - self.int_frame_id[video] + 1
            num_extra_frames = num_frames - self.num_flow_channel + 1
            sampling_interval = 1#int((num_frames-self.num_flow_channel+1)/min_num_extra_frames)
            
            for index in range(num_extra_frames):
                clip_idx = self.int_frame_id[video] + index*sampling_interval
                key = '' .join([video, '-', str(clip_idx)])
                self.dict_test_video_clips[key] = self.dict_test_video_files[video]


    def get_test_image_data(self, mode='test'):
        
        test_set = test_spatial_dataset(mode=mode, dict_video_clips=self.dict_test_video_clips, data_path=self.data_path, img_rows=self.img_rows, img_cols=self.img_cols, num_flow_channel=self.num_flow_channel, 
            data_transform = transforms.Compose([
            transforms.Resize([self.img_rows, self.img_cols]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ]), image_file_format=self.image_file_format)
        
        if self.verbose:
            print('#{} data: {} and one sample fixed size(image): {}' .format(mode, len(test_set), test_set[1][1].size()))

        self.test_dataloader = DataLoader(
            dataset=test_set, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers, 
            pin_memory=True)
                 

class test_motion_dataset(Dataset): 
    
    """
    Load motion data (image frames) for data loader
    ----------------------------------------------------------------------------
    
    INPUT:
    
    mode- define the data loader mode (at present its only in "test" mode)
    dict_video_clips- a dictionary of video clips (where key is the video (clip) file name and value is the starting frame number)
    data_path- path to the optical flow (give before the optical flow (u and v) path) (same as in test_motion_dataloader())
    img_rows- optical flow image frame row size (it should be 224 as during training it was used)(same as in test_motion_dataloader())
    img_cols- optical flow image frame column size (it should be 224 as during training it was used)(same as in test_motion_dataloader())
    num_flow_channel- number of flow channel will be used to test a clip (it should be 10 as during training it was used)
    data_transform- fransformation for data augmentation (like image resize, flip etc.)
    image_file_format- optical flow image file format
    ----------------------------------------------------------------------------
    
    OUTPUT:
    
    video_clips- video clips name
    data- stacked optical flow
    label- label of the video clip 
    frame_ids- starting frame number (ids) of the video clip
    
    """
    
    def __init__(self, mode, dict_video_clips, data_path, img_rows=224, img_cols=224, num_flow_channel=10, data_transform=None, image_file_format='jpg'):
        
        self.mode = mode
        self.keys = list(dict_video_clips.keys())
        self.values = list(dict_video_clips.values())
        self.data_path = data_path
        
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.num_flow_channel = num_flow_channel
        
        self.data_transform = data_transform
        self.image_file_format = image_file_format
        

    def stack_optflow(self):
        
        optflow = torch.FloatTensor(2*self.num_flow_channel, self.img_rows, self.img_cols)
            
        u = '' .join([self.data_path, 'u/', self.video_clips])
        v = '' .join([self.data_path, 'v/', self.video_clips])
        
        for j in range(self.num_flow_channel):
            idx = str(self.clips_idx + j)
            frame_idx = '' .join(['frame', idx.zfill(6)])
            h_optflow = '' .join([u, '/', frame_idx, '.', self.image_file_format])
            v_optflow = '' .join([v, '/', frame_idx, '.', self.image_file_format])
            
            h_optflow = Image.open(h_optflow)
            v_optflow = Image.open(v_optflow)
            
            optflow[2*(j-1),:,:] = self.data_transform(h_optflow)
            optflow[2*(j-1)+1,:,:] = self.data_transform(v_optflow)
                
            h_optflow.close()
            v_optflow.close()  
            
        return optflow
    

    def __len__(self):
        
        return len(self.keys)
    

    def __getitem__(self, idx):
        
        if self.mode == 'test':
            self.video_clips, self.clips_idx = self.keys[idx].split('-')
            self.clips_idx = int(self.clips_idx)
        else:
            raise ValueError('please define your mode here')

        label = int(self.values[idx]) - 1 
        data = self.stack_optflow()
        
        if self.mode == 'test':
            sample = (self.video_clips, data, label, self.clips_idx)
        else:
            raise ValueError('please define your mode "{}" here' .format(self.mode))
            
        return sample


class test_motion_dataloader():
    
    """
    Motion (optical flow images) dataloader for test video data
    ----------------------------------------------------------------------------
    
    INPUT:
    
    data_path- path to the optical flow (give before the optical flow (u and v) path)
    dict_test_video_files- video files as a dictionary (where key will be file name and value will be class label, keep in mind that labels should be [1, 2, ...12] (for unknown class label just put a random label))
    int_frame_id- starting frame of the test video (as you might want to test a video from a particular frame no)
    end_frame_id- end frame of the test video (as you might want to test a video upto a particular frame no)
    img_rows- optical flow image frame row size (it should be 224 as during training it was used)
    img_cols- optical flow image frame column size (it should be 224 as during training it was used)
    num_flow_channel- number of flow channel will be used to test a clip (it should be 10 as during training it was used)
    image_file_format- optical flow image file format
    batch_size- for batch data loader 
    num_workers- numbert of workers (CPU-core) will be used during data loader
    ----------------------------------------------------------------------------

    OUTPUT:
    
    test_dataloader- test optical flow dataloader

    """
    
    def __init__(self, data_path, dict_test_video_files, int_frame_id, end_frame_id, img_rows=224, img_cols=224, num_flow_channel=10, image_file_format='jpg', batch_size=4, num_workers=2, verbose=0):
        
        self.data_path = data_path
        self.dict_test_video_files = dict_test_video_files
        self.int_frame_id = int_frame_id
        self.end_frame_id = end_frame_id
        self.num_flow_channel = num_flow_channel
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.image_file_format = image_file_format
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.verbose = verbose
        

    def run(self):
        
        if self.verbose:
            print('----------------------------------------------------------------------------------------')
                
        # get the test video files
        self.get_test_files_dict()
        
        # get the test flow
        self.get_test_flow_data(mode='test')
        
        if self.verbose:
            print('########################################################################################')
        
        return self.test_dataloader
    
    
    def get_test_files_dict(self):
        
        self.dict_test_video_clips = {}
        
        for video in self.dict_test_video_files:
            num_frames = self.end_frame_id[video] - self.int_frame_id[video] + 1
            num_extra_frames = num_frames - self.num_flow_channel + 1
            sampling_interval = 1#int((num_frames-self.num_flow_channel+1)/min_num_extra_frames)
            
            for index in range(num_extra_frames):
                clip_idx = self.int_frame_id[video] + index*sampling_interval
                key = '' .join([video, '-', str(clip_idx)])
                self.dict_test_video_clips[key] = self.dict_test_video_files[video]
                                    
    
    def get_test_flow_data(self, mode='test'):
        
        test_set = test_motion_dataset(mode=mode, dict_video_clips=self.dict_test_video_clips, data_path=self.data_path, img_rows=self.img_rows, img_cols=self.img_cols, num_flow_channel=self.num_flow_channel, 
            data_transform = transforms.Compose([
            transforms.Resize([self.img_rows, self.img_cols]),
            transforms.ToTensor()
            ]), image_file_format=self.image_file_format)
        
        if self.verbose:
            print('#{} data: {} and one sample fixed size(video): {}' .format(mode, len(test_set), test_set[1][1].size()))

        self.test_dataloader = DataLoader(
            dataset=test_set, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers, 
            pin_memory=True)
        

if __name__ == '__main__':
    
    # data pre-processing 
    input_video_filename = '../data/class_001_043_11M_001_org.mp4'
    images_data_path = ''#'../temp_cache/images/'
    opticalflow_data_path = ''#'../temp_cache/optical_flow/'
    
    data = preprocess_video(input_video_filename)
    
    test_video_filename = data.input_video_filename
    images_data_path = data.images_data_path
    opticalflow_data_path = data.opticalflow_data_path
    
    print('test video file name: {}, image path: "{}", opticalflow path: "{}"' .format(test_video_filename, images_data_path, opticalflow_data_path))
    
    # for data loader
    int_frame_id = {test_video_filename: 1}
    end_frame_id = {test_video_filename: 30}
    
    # load spatial (images) data
    print('spatial (images) data loader')
    spatial_dataloader = test_spatial_dataloader(images_data_path, {test_video_filename:1}, int_frame_id, end_frame_id, verbose=1)
    
    test_spatial_dataloader = spatial_dataloader.run()
    for i, (fl_name, data, label, fm_id) in enumerate(test_spatial_dataloader):
        print('iteration: {}, video file name: {}, data size: {}, class label: {}, starting frame number: {}' .format(i, fl_name, data.shape, label, fm_id))
    
    # load motion (optical flow) data
    print('motion (optical flow) data loader')
    motion_dataloader = test_motion_dataloader(opticalflow_data_path, {test_video_filename:1}, int_frame_id, end_frame_id, verbose=1)
    
    test_motion_dataloader = motion_dataloader.run()
    for i, (fl_name, data, label, fm_id) in enumerate(test_motion_dataloader):
        print('iteration: {}, video file name: {}, data size: {}, class label: {}, starting frame number: {}' .format(i, fl_name, data.shape, label, fm_id))
    
    
    
        
        
        
        
    