import cv2
import numpy as np
import math


def hough_peaks(accumulator, num_peaks, nhood_size=3):
    """
    A function that returns the indices of the accumulator array H that
    correspond to a local maxima.  If threshold is active all values less
    than this value will be ignored, if neighborhood_size is greater than
    (1, 1) this number of indices around the maximum will be surpassed.
    :param H:
    :param num_peaks:
    :param nhood_size:
    :return:
    """

    # loop through number of peaks to identify
    indices = [] # contains (y, x) of each peak
    accumulator_ = np.copy(accumulator)
    for _ in range(num_peaks):
        idx = np.argmax(accumulator_)  # argmax treats accumulator as 1d array
        accumulator_idx = np.unravel_index(idx, accumulator_.shape)  # remap to shape of H
        indices.append(accumulator_idx)

        # surpass indices in neighborhood
        idx_y, idx_x = accumulator_idx  # first separate x, y indexes from argmax(H)
        # if idx_x is too close to the edges choose appropriate values
        min_x = max(idx_x - (nhood_size / 2), 0)

        if (idx_x + (nhood_size / 2) + 1) > accumulator.shape[1]:
            max_x = accumulator.shape[1]
        else:
            max_x = idx_x + (nhood_size / 2) + 1

        # if idx_y is too close to the edges choose appropriate values
        min_y = max(idx_y - (nhood_size / 2), 0)
        
        if (idx_y + (nhood_size / 2) + 1) > accumulator.shape[0]:
            max_y = accumulator.shape[0]
        else:
            max_y = idx_y + (nhood_size / 2) + 1

        # bound each index by the neighborhood size and set all values to 0
        for x in range(int(min_x), int(max_x)):
            for y in range(int(min_y), int(max_y)):
                # remove neighborhoods in accumulator
                accumulator_[y, x] = 0

                # highlight peaks in original accumulator
                if x == min_x or x == (max_x - 1):
                    accumulator[y, x] = 255
                if y == min_y or y == (max_y - 1):
                    accumulator[y, x] = 255

    # return the indices and the original accumulator space with selected points
    return indices, accumulator


def hough_lines_draw(img, indices, rhos, thetas):
    """
    A function that takes indices a rhos table and thetas table and draws
    lines on the input images that correspond to these values.
    :param img:
    :param indices:
    :param rhos:
    :param thetas:
    :return:
    """
    for i in range(len(indices)):
        # reverse engineer lines from rhos and thetas
        rho = rhos[indices[i][0]]
        theta = thetas[indices[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # these are then scaled so that the lines go off the edges of the image
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

def line_detection(source: np.ndarray):
    """

    :param source:
    :return:
    """
    img = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 1.5)
    img = cv2.Canny(img, 100, 200)

    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = img.shape
    diag_len = math.ceil(np.sqrt(np.square(width) + np.square(height)))  # max_dist
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len)
            accumulator[rho, t_idx] += 1

    return accumulator, rhos, thetas

def hough_lines(source: np.ndarray, num_peaks: int = 10) -> np.ndarray:
    """

    :param source:
    :param num_peaks:
    :return:
    """
    src = np.copy(source)
    H, rhos, thetas = line_detection(src)
    indicies, H = hough_peaks(H, num_peaks, nhood_size=20) # find peaks
    hough_lines_draw(src, indicies, rhos, thetas)

    return src


def detectCircles(img, threshold, region, radius = None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 1.5)
    img = cv2.Canny(img, 100, 200)

    (M,N) = img.shape
    if radius == None: # in case no radius is provided ==> default
        R_max = np.max((M,N))
        R_min = 3
    else:
        [R_max,R_min] = radius 

    R = R_max - R_min 
    # A:
    # The first dimension represents the radius.
    # The second and third dimensions represent the X and Y coordinates of the circle center, pad with R_max zeros.
    A = np.zeros((R_max,M+2*R_max,N+2*R_max))
    B = np.zeros((R_max,M+2*R_max,N+2*R_max)) # used for storing the final detected circles after non-maximal suppression.

    theta = np.arange(0,360)*np.pi/180 # 360 radian angles
    edges = np.argwhere(img[:,:]) # all edge coordinates where non zero canny

    ## Voting
    for val in range(R): # for each value r in the range
        r = R_min+val
        circle_template = np.zeros((2*(r+1),2*(r+1)))
        (m,n) = (r+1,r+1)     # center of the circle template

        # create the points on the circle for each theta
        for angle in theta: 
            x = int(np.round(r*np.cos(angle)))
            y = int(np.round(r*np.sin(angle)))
            circle_template[m+x,n+y] = 1


        num_pnts = np.argwhere(circle_template).shape[0]

        
        for x,y in edges:                                                       
            X = [x-m+R_max,x+m+R_max]  # account for padding and center m,n
            Y= [y-n+R_max,y+n+R_max]
            A[r,X[0]:X[1],Y[0]:Y[1]] += circle_template #Adds the circle_template to the corresponding region in the accumulator array A for the current radius r

            
        A[r][A[r]<threshold*num_pnts/r] = 0

    ## Non-Maximal Suppression
    for r,x,y in np.argwhere(A): # for each non zero value in A
        temp = A[r-region:r+region,x-region:x+region,y-region:y+region] # a (region) sized square is taken
        try:
            p,a,b = np.unravel_index(np.argmax(temp),temp.shape) # find the index of max value of the temp array wich is a region of accumulator array
            # p ==> radius dimension.
            # a ==> x within the region.
            # b ==> y within the region.
        except:
            continue
        B[r+(p-region),x+(a-region),y+(b-region)] = 1 # adjust the max values p,a,b by -region to be a global coordinate
        # and sets the detected circle in the accumulator B to 1

    return B[:,R_max:-R_max,R_max:-R_max] # final accumulator array after removing padding

def displayCircles(A, img):
    """

    :param A:
    :param img:
    :return:
    """
    circleCoordinates = np.argwhere(A)                                          #Extracting the circle information
    for r,x,y in circleCoordinates:
        cv2.circle(img,(y, x), r, color=(0,255,0), thickness=2)
    return img

def hough_circles(source: np.ndarray, min_radius: int = 20, max_radius: int = 50) -> np.ndarray:
    """
    :param source:
    :param min_radius:
    :param max_radius:
    :return:
    """

    src = np.copy(source)
    circles = detectCircles(src, threshold=8, region=15,radius=[max_radius, min_radius])
    return displayCircles(circles, src)
