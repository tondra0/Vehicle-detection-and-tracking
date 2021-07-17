from necessary_functions import get_hog_features, color_hist, convert_image, slide_window, bin_spatial, draw_boxes, show_hog
from necessary_functions_2 import heat_threshold, hog_sub_sampling
from parameters import Parameters
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import random
import os


# This function loads images as numpy array to a separate list
def get_dataset(dir):
    images = []
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            if '.DS_Store' not in file:
                images.append(os.path.join(subdir, file))
                
    return list(map(lambda img: cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB), images))


# Read vehicle and non-vehicle images from folders
vehicle_images = get_dataset("./vehicles")  # positive dataset
non_vehicle_images = get_dataset("./non-vehicles")   # negative dataset

# Extract features with given HOG parameters
def extract_features(img, params):
        file_features = []
        # Read in each one by one
        # apply color conversion if other than 'RGB'
        feature_image = convert_image(img, params.color_space)    

        if params.spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=params.spatial_size)
            file_features.append(spatial_features)
        if params.hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=params.hist_bins)
            file_features.append(hist_features)
        if params.hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if params.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        params.orient, params.pix_per_cell, params.cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,params.hog_channel], params.orient, 
                            params.pix_per_cell, params.cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
            
        # Return list of feature vectors
        return np.concatenate(file_features)

# Initialize parameters for feature extraction
params = Parameters(
            color_space = 'YCrCb',
            spatial_size = (16, 16),
            orient = 8,
            pix_per_cell = 8,
            cell_per_block = 2,
            hog_channel = 'ALL',
            hist_bins = 32,
            scale = 1.5,
            spatial_feat=True, 
            hist_feat=True, 
            hog_feat=True
        )

# Select random images
vehicle = random.choice(vehicle_images)
non_vehicle = random.choice(non_vehicle_images)

# Get spatial features
car_spatial_features = extract_features(vehicle, params)
notcar_spatial_features = extract_features(non_vehicle, params)

# Plot spatial features graph for selected vehicle and non-vehicle images
plt.figure()
plt.subplot(121)
plt.plot((car_spatial_features))
plt.xlabel("Car Spatial Features",fontsize=15)

plt.subplot(122)
plt.plot(notcar_spatial_features)
plt.xlabel("Non-Car Spatial Features",fontsize=15)
#plt.savefig('spatial_features.png')

# Show HOG features of selected vehicle and non-vehicle images
show_hog(vehicle, non_vehicle, params)

#-----------------------------------Support vector machine (svm) model-------------------------------------------------

sample_size = 8750
cars = vehicle_images[0:sample_size] # positive dataset
notcars = non_vehicle_images[0:sample_size] # negative dataset

car_features = list(map(lambda img: extract_features(img, params), cars))
notcar_features = list(map(lambda img: extract_features(img, params), notcars))

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X) # this assumes data is normally distributed within each feature

# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',params.orient,'orientations',params.pix_per_cell,
    'pixels per cell and', params.cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC 
svc = LinearSVC()

# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'seconds required to train Linear SVC.')

# Check the score of the SVC
print('Test Accuracy of Linear SVC = ', round(svc.score(X_test, y_test), 4))

# Load all video images to a list
test_images = ['./vid_images/test1.jpg','./vid_images/test2.jpg','./vid_images/test3.jpg','./vid_images/test4.jpg','./vid_images/test5.jpg','./vid_images/test6.jpg']


'''
The 'Multiple car detections and False positives' section and 'Heat maps and threshold limit' section (line 158 to line 260) is 
only run once in order to output some slide window images and heat map images for the final report.
'''

#-----------------------------------Multiple car detections and False positives-------------------------------------------------

#This function returns the windows where the cars are found on the image.
#   `y_start_stop` : Contains the Y axis range to find the cars.
#   `xy_window` : Contains the windows size.
#   `xy_overlap` : Contains the windows overlap percent.
def findCars(img, svc, scaler, params, y_start_stop=[400, 700], xy_window=(64, 64), xy_overlap=(0.85, 0.85) ):
    # Sliding window
    windows_list = []
    windows = slide_window(img, y_start_stop=y_start_stop, xy_window=xy_window, xy_overlap=xy_overlap)
    for window in windows:
        img_window = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        features = extract_features(img_window, params)
        scaled_features = scaler.transform(features.reshape(1, -1))
        pred = svc.predict(scaled_features)
        if pred == 1:
            windows_list.append(window)
    return windows_list


#Run video test images
draw_images = []
for _image in test_images:
    img = cv2.cvtColor(cv2.imread(_image), cv2.COLOR_BGR2RGB)
    windows_list = findCars(img, svc, X_scaler, params)
    out_img = draw_boxes(img, windows_list)
    draw_images.append(out_img)
    
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
ax1.set_title('test1')
ax1.imshow(draw_images[0])
plt.savefig('find_cars1.png')

ax2.set_title('test2')
ax2.imshow(draw_images[1])
plt.savefig('find_cars2.png')

ax3.set_title('test3')
ax3.imshow(draw_images[2])
plt.savefig('find_cars3.png')

f, (ax4, ax5, ax6) = plt.subplots(1, 3, figsize=(20,10))
ax4.set_title('test4')
ax4.imshow(draw_images[3])
plt.savefig('find_cars4.png')

ax5.set_title('test5')
ax5.imshow(draw_images[4])
plt.savefig('find_cars5.png')

ax6.set_title('test6')
ax6.imshow(draw_images[5])
plt.savefig('find_cars6.png')

#-----------------------------------Heat maps and threshold limit-------------------------------------------------

threshold = 4
# Get image, locate cars in the image and get heat threshold
img = cv2.cvtColor(cv2.imread(test_images[0]), cv2.COLOR_BGR2RGB)
windows_list = findCars(img, svc, X_scaler, params)
draw_img, heatmap = heat_threshold(img, threshold, windows_list)

# Show original image and heap map immage
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(draw_img)
ax1.set_title('Car Positions', fontsize=50)
ax2.imshow(heatmap, cmap='hot')
ax2.set_title('Heat Map', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.savefig('heat_map1.png')

img = cv2.cvtColor(cv2.imread(test_images[2]), cv2.COLOR_BGR2RGB)
windows_list = findCars(img, svc, X_scaler, params)
draw_img, heatmap = heat_threshold(img, threshold, windows_list)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(draw_img)
ax1.set_title('Car Positions', fontsize=50)
ax2.imshow(heatmap, cmap='hot')
ax2.set_title('Heat Map', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.savefig('heat_map2.png')

img = cv2.cvtColor(cv2.imread(test_images[3]), cv2.COLOR_BGR2RGB)
windows_list = findCars(img, svc, X_scaler, params)
draw_img, heatmap = heat_threshold(img, threshold, windows_list)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(draw_img)
ax1.set_title('Car Positions', fontsize=50)
ax2.imshow(heatmap, cmap='hot')
ax2.set_title('Heat Map', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.savefig('heat_map3.png')

img = cv2.cvtColor(cv2.imread(test_images[4]), cv2.COLOR_BGR2RGB)
windows_list = findCars(img, svc, X_scaler, params)
draw_img, heatmap = heat_threshold(img, threshold, windows_list)

# Show original image with box and heap map immage
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(draw_img)
ax1.set_title('Car Positions', fontsize=50)
ax2.imshow(heatmap, cmap='hot')
ax2.set_title('Heat Map', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.savefig('heat_map4.png')


#-----------------------------------Finding cars uisng HOG sub-sampling window search-------------------------------------------------

ystart = 350
ystop = 656

threshold = 1
# Get image, locate cars using HOG sub-sampling window search in the image and get heat threshold
img = cv2.cvtColor(cv2.imread(test_images[0]), cv2.COLOR_BGR2RGB)    
car_windows = hog_sub_sampling(img, ystart, ystop, svc, X_scaler, params)
draw_img, heat_map = heat_threshold(img, threshold, car_windows)

# Show original image with box and heap map image
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(draw_img)
ax1.set_title('Car Positions', fontsize=50)
ax2.imshow(heat_map, cmap='hot')
ax2.set_title('Heat Map', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.savefig('sub_sample1.png')

img = cv2.cvtColor(cv2.imread(test_images[1]), cv2.COLOR_BGR2RGB)    
car_windows = hog_sub_sampling(img, ystart, ystop, svc, X_scaler, params)
draw_img, heat_map = heat_threshold(img, threshold, car_windows)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(draw_img)
ax1.set_title('Car Positions', fontsize=50)
ax2.imshow(heat_map, cmap='hot')
ax2.set_title('Heat Map', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.savefig('sub_sample2.png')

img = cv2.cvtColor(cv2.imread(test_images[2]), cv2.COLOR_BGR2RGB)    
car_windows = hog_sub_sampling(img, ystart, ystop, svc, X_scaler, params)
draw_img, heat_map = heat_threshold(img, threshold, car_windows)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(draw_img)
ax1.set_title('Car Positions', fontsize=50)
ax2.imshow(heat_map, cmap='hot')
ax2.set_title('Heat Map', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.savefig('sub_sample3.png')

img = cv2.cvtColor(cv2.imread(test_images[3]), cv2.COLOR_BGR2RGB)    
car_windows = hog_sub_sampling(img, ystart, ystop, svc, X_scaler, params)
draw_img, heat_map = heat_threshold(img, threshold, car_windows)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(draw_img)
ax1.set_title('Car Positions', fontsize=50)
ax2.imshow(heat_map, cmap='hot')
ax2.set_title('Heat Map', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.savefig('sub_sample4.png')

img = cv2.cvtColor(cv2.imread(test_images[4]), cv2.COLOR_BGR2RGB)    
car_windows = hog_sub_sampling(img, ystart, ystop, svc, X_scaler, params)
draw_img, heat_map = heat_threshold(img, threshold, car_windows)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(draw_img)
ax1.set_title('Car Positions', fontsize=50)
ax2.imshow(heat_map, cmap='hot')
ax2.set_title('Heat Map', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.savefig('sub_sample5.png')

img = cv2.cvtColor(cv2.imread(test_images[5]), cv2.COLOR_BGR2RGB)    
car_windows = hog_sub_sampling(img, ystart, ystop, svc, X_scaler, params)
draw_img, heat_map = heat_threshold(img, threshold, car_windows)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(draw_img)
ax1.set_title('Car Positions', fontsize=50)
ax2.imshow(heat_map, cmap='hot')
ax2.set_title('Heat Map', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.savefig('sub_sample6.png')


#----------------------------Create the pipeline video by processing the each frame of the image-------------------------------

def pipeline(img):
    ystart = 350
    ystop = 656
    threshold = 1 
    car_windows = hog_sub_sampling(img, ystart, ystop, svc, X_scaler, params)
    draw_img, heat_map = heat_threshold(img, threshold, svc, X_scaler, car_windows, params)
    
    return draw_img


_output = 'processed_video.mp4'
clip1 = VideoFileClip("_video.mp4")
process_image = pipeline(img)
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(_output))

