#!/usr/bin/env python
# coding: utf-8

# ## Mathematical Foundations of Computer Graphics and Vision 2023
# ## Exercise 2 - Global Optimization

# In this exercise you will apply what you learned about global optimization, especially branch and bound (B&B), concave and convex envelopes, and reformulation. You will implement branch and bound for consensus set maximization.

# <b style="color:red"> TODO A: </b>:  Derivation of the problem formulation in the canonical form of Linear Programming. Please explain all your steps clearly. You may use the hints and notation from section 2 of the excercise sheet.

# Your answer here (Type *Markdown* and LaTeX: $\alpha^2$)

# #### Branch and Bound Implementation for the Consensus Set Maximization Problem

# <b style="color:red"> TODO B: </b>In the following code you are supposed to implement consensus set maximization by branch and bound in the context of stereo matching where the model is a 2D translation between two input images. Please use the imports listed in the cell down. You may also add more imports for example for visualization purposes if you find them useful.
# 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import math
import heapq
from typing import Tuple, List, Optional
from collections import namedtuple
import os

import numpy as np
from scipy.io import loadmat
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# Translation Theta
T = namedtuple("T", ["x", "y"])

# Load input points, input image and fix delta parameter
corr = loadmat('data/ListInputPoints')['ListInputPoints'] # correspondences (nx4 array) [x_i,y_i,x_i',y_i']
N = corr.shape[0]  # number of total points

left_image = plt.imread('data/InputLeftImage.png')
right_image = plt.imread('data/InputRightImage.png')
image_size = left_image.shape

h,w,_=image_size
theta_lower= T(x=-w, y=-h)
theta_upper=T(x=w, y=h)
objlower =T(x=0, y=0)
objupper=T(x=55, y=55)
delta = 3  # inlier threshold [pixels]


# We provide you some skeleton code that you are free to use as it is or where you can also add/ delete parameters.
# You may also write auxiliary functions if you feel the need to do so.

# In[2]:


def plot_matches(inliers: List, outliers: List):
    fig, ax = plt.subplots(1, figsize=(17, 7))
    
    gap = np.zeros_like(left_image[:, :20])
    stitched = np.concatenate((left_image, gap, right_image), 1)
    ax.imshow(stitched)
    
    offset = left_image.shape[1] + gap.shape[1]

    for idx, line in enumerate(corr):
        x1, y1, x2, y2 = line
        if idx in inliers:
            color = "lime"
        elif idx in outliers:
            color = "crimson"
        else:
            color = "blue"
        
        ax.plot([x1, offset + x2], [y1, y2], ".-", color=color, linewidth=0.8)

    ax.axis('off')
    plt.show(fig)
    plt.close(fig)


plot_matches(inliers=[], outliers=[])


# To help with implementing and debugging BnB, we provide a naive estimation of the inlier upper bound within box constraints of $[\underline{T_x}, \overline{T_x}]\times[\underline{T_y}, \overline{T_y}]$.

# In[3]:


def naive_upper_bound(theta_lower: T, theta_upper: T) -> Tuple[T, int]:
    """
    Naive estimation of the upper bound on objective
    Calculates number of inliers in region [Tx_lb, Tx_ub]x[Ty_lb, Ty_ub]

    Parameters:
    - theta_lower, theta_upper: Lower and upper bounds for T

    Returns:
    - Tuple (translation, objcost) representing the estimated upper bound on the model and the objective
    """

    x1, y1, x2, y2 = corr.T
    
    T_x = x2 - x1
    T_y = y2 - y1
    
    inliers = (((T_x + delta) >= theta_lower.x) *
               ((T_y + delta) >= theta_lower.y) *
               ((T_x - delta) <= theta_upper.x) *
               ((T_y - delta) <= theta_upper.y))
    
    #Set center of region as new model
    theta = T(0.5 * (theta_lower.x + theta_upper.x), 0.5 * (theta_lower.y + theta_upper.y))

    return theta, np.sum(inliers)


# In[4]:


def solve_relaxed_LP(theta_lower: T, theta_upper: T) -> Tuple[T, int]:
    # Decision variables: [T_x, T_y, z_1, ..., z_N, w_1x, ..., w_Nx, w_1y, ..., w_Ny]
#N=55 shape of x=3N+2=167
    c = np.concatenate(([0, 0], -np.ones(N), np.zeros(2 * N)))  # Coefficients for the objective function
    # # Bounds for each variable
    bounds = [(theta_lower.x, theta_upper.x), (theta_lower.y, theta_upper.y)] + \
                [(0, 1)] * N + [(None, None)] * (2 * N)  # Bounds for T_x, T_y, z_i's, w_ix's, and w_iy's

    # # Constructing the inequality constraints A_ub * x <= b_ub
    A_ub = []
    b_ub = []

    for i in range(N):
        x_i, y_i, x_i_prime, y_i_prime = corr[i]
        # print(x_i, y_i, x_i_prime, y_i_prime)
        # Constraints for x-coordinate #TxTy,z,wx,wy
        A_ub.append([0, 0] + [x_i - x_i_prime - delta if j == i else 0 for j in range(N)] + [1 if j == i else 0 for j in range(N)] + [0] * N)
        b_ub.append(0)
        
        A_ub.append([0, 0] + [-x_i + x_i_prime - delta if j == i else 0 for j in range(N)] + [-1 if j == i else 0 for j in range(N)] + [0] * N)
        b_ub.append(0)
        
        # Constraints for y-coordinate
        A_ub.append([0, 0] + [y_i - y_i_prime - delta if j == i else 0 for j in range(N)] + [0] * N + [1 if j == i else 0 for j in range(N)])
        b_ub.append(0)
        
        A_ub.append([0, 0] + [-y_i + y_i_prime - delta if j == i else 0 for j in range(N)] + [0] * N + [-1 if j == i else 0 for j in range(N)])
        b_ub.append(0)
        
        # wix
        
        # A_ub.append([0, 0] + [theta_lower.x if j == i else 0 for j in range(N)] + [-1 if j == i else 0 for j in range(N)] + [0] * N)
        # b_ub.append(0)
        
        # A_ub.append([1, 0] + [theta_upper.x if j == i else 0 for j in range(N)] + [-1 if j == i else 0 for j in range(N)] + [0] * N)
        # b_ub.append(theta_upper.x)
        
        # A_ub.append([-1, 0] + [-theta_lower.x if j == i else 0 for j in range(N)] + [1 if j == i else 0 for j in range(N)] + [0] * N)
        # b_ub.append(theta_lower.x)
        
        # A_ub.append([0, 0] + [theta_upper.x if j == i else 0 for j in range(N)] + [1 if j == i else 0 for j in range(N)] + [0] * N)
        # b_ub.append(0)

        # # #wiy
        # A_ub.append([0, 0] + [theta_lower.y if j == i else 0 for j in range(N)] + [0] * N + [-1 if j == i else 0 for j in range(N)])
        # b_ub.append(0)
        
        # A_ub.append([0, 1] + [theta_upper.y if j == i else 0 for j in range(N)] + [0] * N + [-1 if j == i else 0 for j in range(N)])
        # b_ub.append(theta_upper.y)        

        
        # A_ub.append([0, -1] + [-theta_lower.y if j == i else 0 for j in range(N)] + [0] * N + [1 if j == i else 0 for j in range(N)])
        # b_ub.append(theta_lower.y)
        
        # A_ub.append([0, 0] + [theta_upper.y  if j == i else 0 for j in range(N)] + [0] * N + [1 if j == i else 0 for j in range(N)])
        # b_ub.append(0)    
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        T_x, T_y = result.x[:2]
        num_inliers = -np.sum(result.x[2:2 + N])  # Negative because we minimized the negative sum of z_i's
        theta = T(T_x, T_y)
        print("nice")
    else:
        theta = T(0, 0)  # Default to (0, 0) if LP fails
        num_inliers = 0
        print("not nice")

    return theta, num_inliers


# In[5]:


def is_inlier(theta: T) -> np.ndarray:
    """
    Parameters:
    - theta - model that's being tested
    
    Returns:
    - bool vector f size N, where i-th element indicates if i-th point of `corr` is an inlier
    """

    #TODO: Compute if i-th element is an inliers
    # inliers = ...
    x, y, xprime, yprime = corr.T  # Unpack the correspondence columns into x and y coordinates
    
    # Apply the model (translation) to the first set of points
    x_translated = x + theta.x
    y_translated = y + theta.y
    
    # Compute the distances between translated points and their correspondences in the second image
    distancesx = np.sqrt(( x_translated-xprime) ** 2 )
    distancesy= np.sqrt(( y_translated-yprime) ** 2)    
    # Determine inliers: points for which the distance is less than or equal to delta
    inliers = np.logical_and(distancesx <= delta ,distancesy<=delta)
    
    return inliers


# In[6]:


def consensus_set_maximization_by_bnb(image_size: Tuple[int, int], upper_bound_fn) -> Tuple[T, List[int], List[int]]:
    """
    Parameters:
    - image_size - tuple of image size (H, W)

    Returns a tuple of
    - Best model according to the BnB algorithm
    - list of upper bounds for each iteration
    - list of lower bounds for each iteration
    """
    
    h, w, _ = image_size

    best_model = T(0, 0)
    upper_bounds = []
    lower_bounds = []

    #TODO: Implement BnB

    return best_model, lower_bounds, upper_bounds


# In[7]:


def consensus_set_maximization_by_bnb(image_size: Tuple[int, int], upper_bound_fn) -> Tuple[T, List[int], List[int]]:
    """
    Parameters:
    - image_size - tuple of image size (H, W)

    Returns a tuple of
    - Best model according to the BnB algorithm
    - list of upper bounds for each iteration
    - list of lower bounds for each iteration
    """
    
    h, w, _ = image_size

    best_model = T(0, 0)
    upper_bounds = []
    lower_bounds = []

    #TODO: Implement BnB

    return best_model, lower_bounds, upper_bounds


# In[8]:


def plot_bound_convergence(lower_bounds: List[int], upper_bounds: List[int]):
    fig, ax = plt.subplots(1, figsize=(17, 7))
    
    iterations = np.arange(len(lower_bounds)) + 1
    
    # Draw lines
    ax.plot(iterations, lower_bounds, marker="o", color="r")
    ax.plot(iterations, upper_bounds, marker="o", color="b")
    
    # Write titles
    ax.set_title("Convergence of Bounds")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Upper and Lower Bounds")
    ax.legend(["Lower Bound", "Upper Bound"], loc="upper right")
    ax.xaxis.set_major_locator(MultipleLocator(1 if len(lower_bounds) < 20 else 5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.grid(True)

    plt.show(fig)
    plt.close(fig)


# In[9]:


def perform_bnb(upper_bound_fn):
    
    #perform consensus set maximization
    best_model, lower_bounds, upper_bounds = consensus_set_maximization_by_bnb(image_size, upper_bound_fn)
    
    inliers = is_inlier(best_model)
    
    inliers_indices = np.where(inliers)[0]
    outliers_indices = np.where(~inliers)[0]
    
    print(f"Global optimal translational model (Tx, Ty) = {best_model}")
    print(f"Inlier set S_I: {inliers_indices} ({len(inliers_indices)} points)")
    print(f"Outlier indices S_O = S\S_I: {outliers_indices}")
    
    plot_matches(inliers_indices, outliers_indices)
    
    plot_bound_convergence(lower_bounds, upper_bounds)


# #### Results

# <b style="color:red"> TODO C-E: </b> Present and discuss your results. Also plot the figures described in the deliverables D and E.

# In[10]:


# Display results with the naive upper bound
perform_bnb(naive_upper_bound)


# In[11]:


# Display results with the LP upper bound
perform_bnb(solve_relaxed_LP)


# In[12]:


get_ipython().system('jupyter nbconvert --to script Exercise2.ipynb')


# <b style="color:red"> Discussion (not graded): </b> In the previous exercise, you implemented RANSAC to fit polynomials. Now, can you use RANSAC to solve the consensus set maximization problem for stereo matching? If so, please briefly describe the main steps you would take to implement RANSAC for this task, and compare it with the BnB approach. Additionally, can you identify scenarios where one approach may be more effective than the other? If not, explain why it is not feasible to adapt RANSAC for this application. Can you also combine RANSAC and BnB in one pipeline?

# _TODO_: Your answer here
