#
# from https://github.com/nghiaho12/rigid_transform_3D
#

import numpy as np

# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

#    print( "A:", A.T )
#    print( "B:", B.T )
    
    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)
#    centroid_A = np.median(A, axis=1)
#    centroid_B = np.median(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    # if linalg.matrix_rank(H) < 3:
    #     raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    print( "S:", S )

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[1,:] *= -1
        R = Vt.T @ U.T
        
    t = -R @ centroid_A + centroid_B

    print( "R: ", R )
    print( "t: ", t )    
    
    return R, t

#    print( ((-R @ A) + B).T )
#    print( "centroid_A:", centroid_A )
#    print( "centroid_B:", centroid_B )    
#    print( t )
#    t2 =  np.median( (-R @ A) + B, axis=1, keepdims=True )
#    print( t2 )
    
