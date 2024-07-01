import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from EstimateFundamentalMatrix import * 
from EssentialMatrixFromFundamentalMatrix import *
from GetInlierRANSANC import *
from EstimateCameraPose import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
from NonLinearTriangulation import *
from LinearPnP import *
from Helper import *
from NonLinearPnP import *
from PnPRansac import *
from BundleAdjustment import *

# reading images
path = ['./P3Data/1.png','./P3Data/2.png','./P3Data/3.png','./P3Data/4.png','./P3Data/5.png']

images = []
for i in range(len(path)):
    img = cv.imread(path[i])
    images.append(img)

Kcalib = np.array([[531.122155322710, 0, 407.192550839899], [0, 531.541737503901, 313.308715048366], [0, 0, 1]])

def showMatches(image_1, image_2, pts1, pts2, color, file_name):
    comb = np.concatenate((image_1, image_2), axis = 1)

    if pts1 is not None:
        corners_1_x = pts1[:,0].copy().astype(int)
        corners_1_y = pts1[:,1].copy().astype(int)
        corners_2_x = pts2[:,0].copy().astype(int)
        corners_2_y = pts2[:,1].copy().astype(int)
        corners_2_x += image_1.shape[1]

        for i in range(corners_1_x.shape[0]):
            cv.line(comb, (corners_1_x[i], corners_1_y[i]), (corners_2_x[i] ,corners_2_y[i]), color, 1)
    cv.imshow(file_name, comb)
    cv.waitKey() 
    if file_name is not None:    
        cv.imwrite(file_name, comb)
    cv.destroyAllWindows()
    return comb

total_images = 5

print('\nTotal images = ', total_images)

def extractMatchingFeaturesFromFileNew(folder_name, total_images):
   
    feat_descriptor = []
    feat_x = []
    feat_y = []
    feat_flag = []


    for n in range(1, total_images):
        matching_file_name = folder_name + "matching" + str(n) + ".txt"
        file_object = open(matching_file_name,"r")
        nFeatures = 0

        for i, line in enumerate(file_object):
            if i == 0:#nFeatures
                line_elements = line.split(':')
                nFeatures = int(line_elements[1])
            else:
                x_row = np.zeros((1, total_images))
                y_row = np.zeros((1, total_images))
                flag_row = np.zeros((1, total_images), dtype = int)

                line_elements = line.split()
                features = [float(x) for x in line_elements]
                features = np.array(features)
       
                n_matches = features[0]
                r = features[1]
                g = features[2]
                b = features[3]

                feat_descriptor.append([r,g,b])

                src_x = features[4]
                src_y = features[5]

                x_row[0, n-1] = src_x
                y_row[0, n-1] = src_y
                flag_row[0, n-1] = 1

                m = 1
                while n_matches > 1:
                    image_id = int(features[5+m])
                    image_id_x = features[6+m]
                    image_id_y = features[7+m]
                    m = m+3
                    n_matches = n_matches - 1

                    x_row[0, image_id - 1] = image_id_x
                    y_row[0, image_id - 1] = image_id_y
                    flag_row[0, image_id - 1] = 1

                feat_x.append(x_row)
                feat_y.append(y_row)
                feat_flag.append(flag_row)

    return np.array(feat_x).reshape(-1, total_images), np.array(feat_y).reshape(-1, total_images), np.array(feat_flag).reshape(-1, total_images), np.array(feat_descriptor).reshape(-1, 3)

## extracting feature correspondences:
feat_x, feat_y, feat_flag, feat_descriptor = extractMatchingFeaturesFromFileNew('./P3Data/',total_images)
# print('\nfeat_x =\n',feat_x)
# print('\nfeat_flag =\n',feat_flag)
# print('\nfeat_flag_shape =',np.shape(feat_flag))

filtered_feat_flag = np.zeros_like(feat_flag)
f_matrix = np.empty(shape=(total_images, total_images), dtype=object)

print('\nUsing RANSAC to find inliers:')
for i in range(0, total_images - 1):
    for j in range(i + 1, total_images):

        idx = np.where(feat_flag[:,i] & feat_flag[:,j])
        pts1 = np.hstack((feat_x[idx, i].reshape((-1, 1)), feat_y[idx, i].reshape((-1, 1))))
        pts2 = np.hstack((feat_x[idx, j].reshape((-1, 1)), feat_y[idx, j].reshape((-1, 1))))

        # to show matches
        # showMatches(images[i], images[j], pts1, pts2, (0,255,0), None)
        idx = np.array(idx).reshape(-1)
        if len(idx) > 8:
            F_best, chosen_idx = getInliers(pts1, pts2, idx)
            print('Inliers for image:',  i,j ,'=', len(chosen_idx), '/', len(idx) )            
            f_matrix[i, j] = F_best
            filtered_feat_flag[chosen_idx, j] = 1
            filtered_feat_flag[chosen_idx, i] = 1

# print('f_matrix=\n',f_matrix)
########################################## stored all feature points ( all correct up to this point )########################################################

################################ Estimating initial F and E values for first two images ##########################
print('\nEstimating initial F and E values for first two images...')
n,m = 0,1
F12 = f_matrix[n,m]
E12 = getEssentialMatrix(Kcalib, F12)
# print('E12=\n',E12)

Rset, Cset = ExtractCameraPose(E12)
# print('Rset=\n',Rset)
# print('Cset=\n',Cset)

idx = np.where(filtered_feat_flag[:,n] & filtered_feat_flag[:,m])
pts1 = np.hstack((feat_x[idx, n].reshape((-1, 1)), feat_y[idx, n].reshape((-1, 1))))
pts2 = np.hstack((feat_x[idx, m].reshape((-1, 1)), feat_y[idx, m].reshape((-1, 1))))

R1_ = np.identity(3)
C1_ = np.zeros((3,1))
I = np.identity(3)
Xset = []
for i in range(len(Cset)):
    pts3D = []
    x1 = pts1
    x2 = pts2
    X = LinearTriangulation(Kcalib, C1_, R1_, Cset[i], Rset[i], x1, x2)
    X = X/X[:,3].reshape(-1,1)
    Xset.append(X)

print('\nSelecting the best R and C using Depth Positivity constraint')
R_best, C_best, X = DisambiguatePose(Rset, Cset, Xset)
X = X/X[:,3].reshape(-1,1)

X_nlt = NonLinearTriangulation(Kcalib, pts1, pts2, X, R1_, C1_, R_best, C_best)
X_nlt = X_nlt / X_nlt[:,3].reshape(-1,1)

error1 = meanReprojectionError(X, pts1, pts2, R1_, C1_, R_best, C_best, Kcalib )
error2 = meanReprojectionError(X_nlt, pts1, pts2, R1_, C1_, R_best, C_best, Kcalib)
print('\nLT error: ', error1, 'NLT error:', error2)

############################ Plotting for img1 and img 2 ############################

x = X[:,0]
y= X[:,1]
z = X[:,2]
x1 = X_nlt[:,0]
y1= X_nlt[:,1]
z1 = X_nlt[:,2]
fig = plt.figure(figsize = (10,10))
plt.xlim(-10,12)
plt.ylim(-5,25)
plt.scatter(x,z,marker='.',linewidths=0.2, color = 'blue', label = 'LT')
plt.scatter(x1,z1,marker='.',linewidths=0.2, color = 'red', label = 'NLT')
plt.legend(loc='upper right')
plt.title(str(0)+str(1))
plt.savefig('./IntermediateOutputImages/01.png')
plt.show()

############################ Register cam 1 and cam 2 #############################

X_all = np.zeros((feat_x.shape[0], 3))
camera_indices = np.zeros((feat_x.shape[0], 1), dtype = int) 
X_found = np.zeros((feat_x.shape[0], 1), dtype = int)

X_all[idx] = X[:, :3]
X_found[idx] = 1
camera_indices[idx] = 1

X_found[np.where(X_all[:,2] < 0)] = 0

Cset_ = []
Rset_ = []

C0 = np.zeros(3)
R0 = np.identity(3)
Cset_.append(C0)
Rset_.append(R0)

Cset_.append(C_best)
Rset_.append(R_best)
print('\n #####################  Registered poses for camera 1 and camera 2 #####################' )

print('\nEstimating E value and cam poses for rest of the images...')
for i in range(2, total_images):

    print('\nComputing for Image: ', str(i+1) ,'......')
    feature_idx_i = np.where(X_found[:, 0] & filtered_feat_flag[:, i])
    if len(feature_idx_i[0]) < 8:
        print("Found ", len(feature_idx_i), "common points between X and ", i, " image")
        continue

    pts_i = np.hstack((feat_x[feature_idx_i, i].reshape(-1,1), feat_y[feature_idx_i, i].reshape(-1,1)))
    X = X_all[feature_idx_i, :].reshape(-1,3)
    
    #PnP
    R_init, C_init = PnPRANSAC(Kcalib, pts_i, X, n_iterations = 1000, error_thresh = 5)
    errorLinearPnP = reprojectionErrorPnP(X, pts_i, Kcalib, R_init, C_init)
    
    Ri, Ci = NonLinearPnP(Kcalib, pts_i, X, R_init, C_init)
    errorNonLinearPnP = reprojectionErrorPnP(X, pts_i, Kcalib, Ri, Ci)
    print("Error after linear PnP: ", errorLinearPnP, " Error after non linear PnP: ", errorNonLinearPnP)

    Cset_.append(Ci)
    Rset_.append(Ri)

    #performing trianglulation
    for j in range(0, i):
        idx_X_pts = np.where(filtered_feat_flag[:, j] & filtered_feat_flag[:, i])
        if (len(idx_X_pts[0]) < 8):
            continue

        x1 = np.hstack((feat_x[idx_X_pts, j].reshape((-1, 1)), feat_y[idx_X_pts, j].reshape((-1, 1))))
        x2 = np.hstack((feat_x[idx_X_pts, i].reshape((-1, 1)), feat_y[idx_X_pts, i].reshape((-1, 1))))

        X = LinearTriangulation(Kcalib, Cset_[j], Rset_[j], Ci, Ri, x1, x2)
        X = X/X[:,3].reshape(-1,1)
        
        LT_error = meanReprojectionError(X, x1, x2, Rset_[j], Cset_[j], Ri, Ci, Kcalib)
        
        X_nlt = NonLinearTriangulation(Kcalib, x1, x2, X, Rset_[j], Cset_[j], Ri, Ci)
        X_nlt = X_nlt/X_nlt[:,3].reshape(-1,1)
        
        print("\nAdded", len(idx_X_pts[0]), " points between image", i ," and ", j)
        NLT_error = meanReprojectionError(X_nlt, x1, x2, Rset_[j], Cset_[j], Ri, Ci, Kcalib)
        print("Error after LT: ", LT_error, " Error after NLT: ", NLT_error)
        
        X_all[idx_X_pts] = X_nlt[:,:3]
        X_found[idx_X_pts] = 1
        
############# plotting ###################

        x = X[:,0]
        y= X[:,1]
        z = X[:,2]
        x1 = X_nlt[:,0]
        y1= X_nlt[:,1]
        z1 = X_nlt[:,2]
        fig = plt.figure(figsize = (10,10))
        plt.xlim(-10,12)
        plt.ylim(-5,25)
        plt.scatter(x,z,marker='.',linewidths=0.2, color = 'blue', label = 'LT')
        plt.scatter(x1,z1,marker='.',linewidths=0.2, color = 'red', label = 'NLT')
        plt.legend(loc='upper right')
        plt.title(str(j)+str(i))
        plt.savefig('./IntermediateOutputImages/' + str(j)+str(i))
        plt.show()
############################################
    
    print('\n##################### Registered pose for camera : ', i+1, '######################')

X_found[X_all[:,2]<0] = 0    
print('##########################################################################')

feat_idx = np.where(X_found[:, 0])
X = X_all[feat_idx]
x = X[:,0]
y = X[:,1]
z = X[:,2]

fig = plt.figure(figsize = (10,10))
plt.xlim(-4,6)  
plt.ylim(-5,25)
plt.scatter(x,z,marker='.',linewidths=0.5, color = 'blue')
for i in range(0, len(Cset_)):
    R1 = getEuler(Rset_[i])
    R1 = np.rad2deg(R1)
    plt.plot(Cset_[i][0],Cset_[i][2], marker=(3,0, int(R1[1])), markersize=15, linestyle='None')

plt.savefig('./IntermediateOutputImages/All.png')
plt.show()
