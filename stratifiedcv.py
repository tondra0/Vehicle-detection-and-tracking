from necessary_functions import get_hog_features, color_hist, convert_image, bin_spatial
from parameters import Parameters
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold
import numpy as np
import cv2
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
vehicle_images = get_dataset("./vehicles")
non_vehicle_images = get_dataset("./non-vehicles")

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


#-----------------------------------Support vector machine (svm) model-------------------------------------------------

sample_size = 8750
cars = vehicle_images[0:sample_size]
notcars = non_vehicle_images[0:sample_size]

car_features = list(map(lambda img: extract_features(img, params), cars))
notcar_features = list(map(lambda img: extract_features(img, params), notcars))

# Get all features except the labels
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        

# Get labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Use a linear SVC 
svc = LinearSVC()

#------------------------------Stratified 10-fold cross-validation-----------------------------------------

stratifiedCV = StratifiedKFold(n_splits=10)

tprs1 = []
aucs1 = []
mean_fpr = np.linspace(0, 1, 100)

fig1, ax1 = plt.subplots()


for i, (train, test) in enumerate(stratifiedCV.split(X, y)):
    # Support vector machine classifier 
    svc.fit(X[train], y[train])
    viz1 = plot_roc_curve(svc, X[test], y[test],
                         name='ROC fold {}'.format(i),
                         alpha=0.3, lw=1, ax=ax1)
    interp_tpr1 = np.interp(mean_fpr, viz1.fpr, viz1.tpr)
    interp_tpr1[0] = 0.0
    tprs1.append(interp_tpr1)
    aucs1.append(viz1.roc_auc)


ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

mean_tpr1 = np.mean(tprs1, axis=0)
mean_tpr1[-1] = 1.0
mean_auc1 = auc(mean_fpr, mean_tpr1)
std_auc1 = np.std(aucs1)
ax1.plot(mean_fpr, mean_tpr1, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc1, std_auc1),
        lw=2, alpha=.8)

std_tpr1 = np.std(tprs1, axis=0)
tprs_upper1 = np.minimum(mean_tpr1 + std_tpr1, 1)
tprs_lower1 = np.maximum(mean_tpr1 - std_tpr1, 0)
ax1.fill_between(mean_fpr, tprs_lower1, tprs_upper1, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax1.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Receiver operating characteristic")
ax1.legend(loc="lower right")


plt.show()
#plt.savefig('roc.png')
print("Stratified cross-validation done!")
