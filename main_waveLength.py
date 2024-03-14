#%% packages
import os
import sys
import pandas as pd
from scipy import ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure
import skimage.measure
import cv2
from skimage import exposure
from skimage import color, data, filters, graph, measure, morphology
import time

#%% functions

# Function to read location files from a directory
def read_location_files(**kwargs):
    """
    Read location files (CSV) from a directory.
    
    Args:
        directory (str): Directory path containing CSV files.
    
    Returns:
        list: List of DataFrames (one for each CSV file).
    """
    directory = kwargs.get('directory', os.getcwd())

    all_xypts = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # full path to the csv file
            file_path = os.path.join(directory, filename)
            xys = pd.read_csv(file_path)
            all_xypts.append(xys)
    
    return all_xypts

# Function to list AVI video files from a directory
def list_avi_files(**kwargs):
    """
    List AVI video files from a directory.
    
    Args:
        directory (str): Directory path containing AVI files.
    
    Returns:
        list: List of OpenCV VideoCapture objects.
    """
    directory = kwargs.get('directory', os.getcwd())

    all_avi_files = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.avi'):
            # full path to the csv file
            avi_file_path = os.path.join(directory, filename)
            cap = cv2.VideoCapture(avi_file_path)
            all_avi_files.append(cap)
    
    return all_avi_files

# Function to crop a frame
def crop_this(frame, start_pos, length, width, with_plot=False, gray_scale=True):
    """
    Crop a frame.
    
    Args:
        frame (numpy.array): Input image frame.
        start_pos (tuple): Starting position for cropping (x, y).
        length (int): Length of the cropped region.
        width (int): Width of the cropped region.
        with_plot (bool): Whether to plot the cropped image.
        gray_scale (bool): Convert to grayscale if True.
    
    Returns:
        numpy.array: Cropped image.
    """
    image_shape = frame.shape
    length = abs(length)
    width = abs(width)

    top_left = (start_pos[1] - length/2, start_pos[0] - width/2)
    rect = plt.Rectangle(top_left, length, width)
    start_row =  int(rect.xy[0])
    start_column = int(rect.xy[1])
    
    end_row = length + start_row
    end_row = end_row if end_row <= image_shape[0] else image_shape[0]
    end_column = width + start_column
    end_column = end_column if end_column <= image_shape[1] else image_shape[1]

    image_cropped = frame[start_row:end_row, start_column:end_column]
    cmap_val = None if not gray_scale else 'gray'
    return image_cropped

# Function to calculate waves in an image
def calc_waves(croped_im, ip_dist):
    """
    Calculate waves in a cropped image.
    
    Args:
        croped_im (numpy.array): Cropped grayscale image.
        ip_dist (float): Pixel distance.
    
    Returns:
        float: Calculated wavelength in microns.
    """
    image =  cv2.cvtColor(croped_frame, cv2.COLOR_BGR2GRAY)
    # Masking and filtering 
    image = exposure.rescale_intensity(image)
    image = cv2.GaussianBlur(image, (7, 7), 0)
    ret, edges = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    t0, t1 = filters.threshold_multiotsu(image, classes=3)
    mask = (image > t0)
    vessels = filters.sato(image, sigmas=range(1, 10)) * mask
    thresholded = filters.apply_hysteresis_threshold(vessels, 0.01, 0.03)
    labeled = ndi.label(thresholded)[0]
    largest_nonzero_label = np.argmax(np.bincount(labeled[labeled > 0]))
    binary = labeled == largest_nonzero_label
    skeleton = morphology.skeletonize(binary)
    g, nodes = graph.pixel_graph(skeleton, connectivity=2)
    px, distances = graph.central_pixel(
        g, nodes=nodes, shape=skeleton.shape, partition_size=80
    )
    centroid = measure.centroid(labeled > 0)
    centro = np.abs((float(centroid[0])) - (float(centroid[1])))
    x_wave = float(float(centroid[0]) - float(px[0]))
    y_wave = float(float(centroid[1]) - float(px[1]))
    wavelengh = np.sqrt(x_wave ** 2 + y_wave ** 2)
    wavelengh_micron = wavelengh * ip_dist
    cv2.destroyAllWindows()
    return wavelengh_micron

#%% main code

# Read the location files and the videos 
all_xypts = read_location_files()
all_avi_files =  list_avi_files()

# Ensure the same number of CSV and video files
if len(all_xypts) != len(all_avi_files):
    print('Exiting because len(vids) != len(csv)')
    sys.exit()

# Run frame by frame, locate the waves, and analyze them
waveLength_data = []
for v in np.arange(0, len(all_avi_files)):
    current_vid = all_avi_files[v]
    frames_count = int(current_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    waveLength_list = []
    for frame_number in np.arange(0, frames_count-1):
        current_vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret , frame = current_vid.read()
        xcoordinate1, ycoordinate1 = all_xypts[0]['pt1_cam1_X'][frame_number], all_xypts[0]['pt1_cam1_Y'][frame_number]
        xcoordinate2, ycoordinate2 = all_xypts[0]['pt1_cam1_X'][frame_number+1], all_xypts[0]['pt1_cam1_Y'][frame_number+1]
        start_pos = xcoordinate1, ycoordinate1,
        croped_frame = crop_this(
            frame=frame,
            start_pos=start_pos,
            length=75,
            width=75,
            with_plot=True,
            gray_scale=True
        )
        waveLength_list.append(calc_waves(croped_frame, ip_dist=0.16))
        plt.plot(waveLength_list)
    current_vid.release()
    waveLength_data.append(np.array(waveLength_list))