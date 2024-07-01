import numpy as np

def ExtractCameraPose(E):
    U,S,V_T = np.linalg.svd(E)
    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])

    C = [U[:,2],-U[:,2],U[:,2],-U[:,2]]
    R = [np.dot(U,np.dot(W,V_T)), np.dot(U,np.dot(W,V_T)), np.dot(U,np.dot(W.T,V_T)),np.dot(U,np.dot(W.T,V_T))]

    for i in range(len(R)):
        if np.linalg.det(R[i]) == -1:
            R[i] = -R[i]
            C[i] = -C[i]

    return R, C


