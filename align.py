
import os
import cv2
import numpy as np
from PIL import Image


ratio = 0.3
iter_num = 2000
fit_pos_cnt_thresh = 30 # ratio设置0.3时设置数量，少的话不是同一物体

class Ransac():

    def __init__(self, K=3, threshold=1):
        ''' __INIT__
            Initialize the instance.
            Input argements:
            - K : the number of corresponding points,
            default is 3
            - threshold : determing which points are inliers
            by comparing residual with it
        '''

        self.K = K
        self.threshold = threshold

    def residual_lengths(self, A, t, pts_s, pts_t):
        ''' RESIDUAL_LENGTHS
            Compute residual length (Euclidean distance) between
            estimation and real target points. Estimation are
            calculated by the given source point and affine
            transformation (A & t).
            Input arguments:
            - A, t : the estimated affine transformation calculated
            by least squares method
            - pts_s : key points from source image
            - pts_t : key points from target image
            Output:
            - residual : Euclidean distance between estimated points
            and real target points
        '''

        if not(A is None) and not(t is None):
            # Calculate estimated points:
            # pts_esti = A * pts_s + t
            pts_e = np.dot(A, pts_s) + t

            # Calculate the residual length between estimated points
            # and target points
            diff_square = np.power(pts_e - pts_t, 2)
            residual = np.sqrt(np.sum(diff_square, axis=0))
        else:
            residual = None

        return residual

    def ransac_fit(self, pts_s, pts_t):
        ''' RANSAC_FIT
            Apply the method of RANSAC to obtain the estimation of
            affine transformation and inliers as well.
            Input arguments:
            - pts_s : key points from source image
            - pts_t : key points from target image
            Output:
            - A, t : estimated affine transformation
            - inliers : indices of inliers that will be applied to refine the
            affine transformation
        '''

        # Create a Affine instance to do estimation
        af = Affine()

        # Initialize the number of inliers
        inliers_num = 0

        # Initialize the affine transformation A and t,
        # and a vector that stores indices of inliers
        A = None
        t = None
        inliers = None

        for i in range(iter_num):
            # Randomly generate indices of points correspondences
            idx = np.random.randint(0, pts_s.shape[1], (self.K, 1))
            # Estimate affine transformation by these points
            A_tmp, t_tmp = af.estimate_affine(pts_s[:, idx], pts_t[:, idx])

            # Calculate the residual by applying estimated transformation
            residual = self.residual_lengths(A_tmp, t_tmp, pts_s, pts_t)

            if not(residual is None):
                # Obtain the indices of inliers
                inliers_tmp = np.where(residual < self.threshold)
                # Obtain the number of inliers
                inliers_num_tmp = len(inliers_tmp[0])

                # Set affine transformation and indices og inliers
                # in one iteration which has the most of inliers
                if inliers_num_tmp > inliers_num:
                    # Update the number of inliers
                    inliers_num = inliers_num_tmp
                    # Set returned value
                    inliers = inliers_tmp
                    A = A_tmp
                    t = t_tmp
            else:
                pass

        return A, t, inliers
    
    
class Affine():

    def create_test_case(self, outlier_rate=0):
        ''' CREATE_TEST_CASE
            Randomly generate a test case of affine transformation.
            Input arguments:
            - outlier_rate : the percentage of outliers in test case,
            default is 0
            Outputs:
            - pts : warped points
            - pts_tilde : source points that wll be transformed
            - A, t : parameters of affine transformation, A is a 2x2
            matrix, t is a 2x1 vector, both of them are created randomly
        '''

        # Randomly generate affine transformation
        # A is a 2x2 matrix, the range of each value is from -2 to 2
        A = 4 * np.random.rand(2, 2) - 2

        # % t is a 2x1 VECTOR, the range of each value is from -10 to 10
        t = 20 * np.random.rand(2, 1) - 10

        # Set the number of points in test case
        num = 1000

        # Compute the number of outliers and inliers respectively
        outliers = int(np.round(num * outlier_rate))
        inliers = int(num - outliers)

        # Gernerate source points whose scope from (0,0) to (100, 100)
        pts_s = 100 * np.random.rand(2, num)
        # Initialize warped points matrix
        pts_t = np.zeros((2, num))

        # Compute inliers in warped points matrix by applying A and t
        pts_t[:, :inliers] = np.dot(A, pts_s[:, :inliers]) + t

        # Generate outliers in warped points matrix
        pts_t[:, inliers:] = 100 * np.random.rand(2, outliers)

        # Reset the order of warped points matrix,
        # outliers and inliers will scatter randomly in test case
        rnd_idx = np.random.permutation(num)
        pts_s = pts_s[:, rnd_idx]
        pts_t = pts_t[:, rnd_idx]

        return A, t, pts_s, pts_t

    def estimate_affine(self, pts_s, pts_t):
        ''' ESTIMATE_AFFINE
            Estimate affine transformation by the given points
            correspondences.
            Input arguments:
            - pts : points in target image
            - pts_tilde : points in source image
            Outputs:
            - A, t : the affine transformation, A is a 2x2 matrix
            that indicates the rotation and scaling transformation,
            t is a 2x1 vector determines the translation
            Method:
            To estimate an affine transformation between two images,
            at least 3 corresponding points are needed.
            In this case, 6-parameter affine transformation are taken into
            consideration, which is shown as follows:
            | x' | = | a b | * | x | + | tx |
            | y' |   | c d |   | y |   | ty |
            For 3 corresponding points, 6 equations can be formed as below:
            | x1 y1 0  0  1 0 |       | a  |       | x1' |
            | 0  0  x1 y1 0 1 |       | b  |       | y1' |
            | x2 y2 0  0  1 0 |   *   | c  |   =   | x2' |
            | 0  0  x2 y2 0 1 |       | d  |       | y2' |
            | x3 y3 0  0  1 0 |       | tx |       | x3' |
            | 0  0  x3 y3 0 1 |       | ty |       | y3' |
            |------> M <------|   |-> theta <-|   |-> b <-|
            Solve the equation to compute theta by:  theta = M \ b
            Thus, affine transformation can be obtained as:
            A = | a b |     t = | tx |
                | c d |         | ty |
        '''

        # Get the number of corresponding points
        pts_num = pts_s.shape[1]

        # Initialize the matrix M,
        # M has 6 columns, since the affine transformation
        # has 6 parameters in this case
        M = np.zeros((2 * pts_num, 6))

        for i in range(pts_num):
            # Form the matrix m
            temp = [[pts_s[0, i], pts_s[1, i], 0, 0, 1, 0],
                    [0, 0, pts_s[0, i], pts_s[1, i], 0, 1]]
            M[2 * i: 2 * i + 2, :] = np.array(temp)

        # Form the matrix b,
        # b contains all known target points
        b = pts_t.T.reshape((2 * pts_num, 1))

        try:
            # Solve the linear equation
            theta = np.linalg.lstsq(M, b)[0]

            # Form the affine transformation
            A = theta[:4].reshape((2, 2))
            t = theta[4:]
        except np.linalg.linalg.LinAlgError:
            # If M is singular matrix, return None
            # print("Singular matrix.")
            A = None
            t = None

        return A, t
    
    
def extract_sift(img):
    ''' EXTRACT_SIFT
        Extract SIFT descriptors from the given image.
        Input argument:
        - img : the image to be processed
        Output:
        -kp : positions of key points where descriptors are extracted
        - desc : all SIFT descriptors of the image, its dimension
        will be n by 128 where n is the number of key points
    '''

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Extract key points and SIFT descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(img_gray, None)

    # Extract positions of key points
    kp = np.array([p.pt for p in kp]).T

    return kp, desc


def match_sift(desc_s, desc_t):
    ''' MATCH_SIFT
        Match SIFT descriptors of source image and target image.
        Obtain the index of conrresponding points to do estimation
        of affine transformation.
        Input arguments:
        - desc_s : descriptors of source image
        - desc_t : descriptors of target image
        Output:
        - fit_pos : index of corresponding points
    '''

    # Match descriptor and obtain two best matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_s, desc_t, k=2)

    # Initialize output variable
    fit_pos = np.array([], dtype=np.int32).reshape((0, 2))

    matches_num = len(matches)
    for i in range(matches_num):
        # Obtain the good match if the ration id smaller than 0.8
        if matches[i][0].distance <= ratio * matches[i][1].distance:
            temp = np.array([matches[i][0].queryIdx,
                             matches[i][0].trainIdx])
            # Put points index of good match
            fit_pos = np.vstack((fit_pos, temp))

    return fit_pos


def affine_matrix(kp_s, kp_t, fit_pos):
    ''' 
        Compute affine transformation matrix by corresponding points.
        Input arguments:
        - kp_s : key points from source image
        - kp_t : key points from target image
        - fit_pos : index of corresponding points
        Output:
        - M : the affine transformation matrix whose dimension
        is 2 by 3
    '''

    # Extract corresponding points from all key points
    kp_s = kp_s[:, fit_pos[:, 0]]
    kp_t = kp_t[:, fit_pos[:, 1]]

    # Apply RANSAC to find most inliers
    _, _, inliers = Ransac(3, 1).ransac_fit(kp_s, kp_t)

    # Extract all inliers from all key points
    kp_s = kp_s[:, inliers[0]]
    kp_t = kp_t[:, inliers[0]]

    # Use all inliers to estimate transform matrix
    A, t = Affine().estimate_affine(kp_s, kp_t)
    M = np.hstack((A, t))

    return M

def warp_image(source, target, M):
    ''' WARP_IMAGE
        Warp the source image into target with the affine
        transformation matrix.
        Input arguments:
        - source : the source image to be warped
        - target : the target image
        - M : the affine transformation matrix
    '''

    # Obtain the size of target image
    rows, cols, _ = target.shape

    # Warp the source image
    warp = cv2.warpAffine(source, M, (cols, rows))

    # Merge warped image with target image to display
    merge = np.uint8(target * 0.5 + warp * 0.5)

    # Show the result
#     cv2.imshow('img', merge)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

    return merge, warp, source
