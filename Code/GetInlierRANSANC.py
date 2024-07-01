import numpy as np
from EstimateFundamentalMatrix import *

def errorF(pts1, pts2, F): 
    """
    check the epipolar constraint
    """
    x1,x2 = pts1, pts2
    x1_temp=np.array([x1[0], x1[1], 1])
    x2_temp=np.array([x2[0], x2[1], 1]).T

    error = np.dot(x2_temp, np.dot(F, x1_temp))
    
    return np.abs(error)


def getInliers(pts1, pts2, idx):
    n_iterations = 2000
    error_thresh = 0.05

    inliers_thresh = 0
    chosen_indices = []
    chosen_f = None

    for i in range(0, n_iterations):
  
        #select 8 points randomly
        n_rows = pts1.shape[0]
        random_indices = np.random.choice(n_rows, size=8)
        pts1_8 = pts1[random_indices, :] 
        pts2_8 = pts2[random_indices, :] 
        f_8 = EstimateFundamentalMatrix(pts1_8, pts2_8)
        indices = []
        if f_8 is not None:
            for j in range(n_rows):

                error = errorF(pts1[j, :], pts2[j, :], f_8)
                
                if error < error_thresh:
                    indices.append(idx[j])

        if len(indices) > inliers_thresh:
            inliers_thresh = len(indices)
            best_idx = indices
            best_f = f_8

    return best_f, best_idx
