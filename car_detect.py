#!/usr/bin/env python

"""
Written by Andrew Taber for earthmine inc.

Copyright (c) 2012 earthmine, inc. All rights reserved.

The purpose of this algorithm is to take a depth panorama
and extract features (in particular cars) in order to speed
up and make more accurate the license plate detection process 
for later blurring. This is a prototype.

So far this program is able to produce a segmented panorama that
is pretty accurate in terms of detecting cars. The next step
would be to use the Hough transform on each car to find the 
lines that correspond to faces of the cars - and then use
those lines to project the rgb image onto those lines and
so produce "rotated" car faces (the idea being that license 
plates are much easier to detect when orthogonal to the
camera), however this step is not yet completed. 
"""

from PIL import Image
import numpy as np
from scipy.misc import imsave
from math import *
from matplotlib import pyplot as plt
from scipy import ndimage
import pymeanshift as pms
from pylab import cm

def get_depth_map(name):
# Get resized depth panorama
    I = Image.open(name)
    I = I.resize((1000,500))
    A = np.asarray(I, dtype=np.uint16)
    return np.array(A, dtype=float)/100.0

def get_plane_map(name):
# Get resized plane panorama
    I = Image.open(name)
    I = I.resize((1000,500))
    A = np.asarray(I, dtype=np.uint8)
    return A

def get_plane_equations(name):
# Parse plane panorama, extract equations
    f = open(name, "r")
    f.readline() #skip first line
    eqn_line = f.readline()
    equations = []
    while (eqn_line):
        eqn_list = eqn_line.split(" ")
        equations.append((eqn_list[1],eqn_list[2],eqn_list[3],eqn_list[4]))
        eqn_line = f.readline()
    return equations

def process_depth_map(in_f, in_f_plane, in_pln, out_f):

    A = get_depth_map(in_f) 
    equations = get_plane_equations(in_pln)
    plane = get_plane_map(in_f_plane)
    H,buff = get_height_map(equations,plane,A)
    delta = get_delta_map(H)
    delta,labels = segment_image(delta)
    pano = transform_labels(delta,buff)
    box_pano = create_boxes(pano,labels)
    box = create_boxes(delta,labels)
    line_param = get_angles(delta,box,labels)
    lines = get_lines(box,line_param)

    # TODO: Put this block of code somewhere else
    # Fill in raster image to check accuracy of 
    # segmented pano.
    I = Image.open(path.join(sys.argv[1],"raster.jpg"))
    (h,w) = box_pano.shape
    I = I.resize((w,h))
    candidates = Image.new("RGBA",(w,h))
    for y in range(h):
        for x in range(w):
            if box_pano[y,x] != 0:
                pix = I.getpixel((x,y))
                candidates.putpixel((x,h-y),pix)
    
    RGB = I.transpose(Image.FLIP_TOP_BOTTOM) 

    plt.subplot(221)
    plt.imshow(pano)
    plt.subplot(222)
    plt.imshow(candidates)
    plt.subplot(223)
    plt.imshow(delta)
    plt.subplot(224)
    plt.imshow(lines)
    plt.show()
    #imsave(delta,out_f)

def get_lines(box_map,line_params):
    """
    Go from the line parameters we found for each segmented box 
    via the Hough transform to lines on each box. This is largely
    meant to test that the right lines are drawn in the right places.

    Input:
        box_map - segmented height image
        line_params - list of tuples of the form (label,angle,distance from origin)
    Output:
        An array where in each segmentation box of box_map, the most popular line is
        drawn (basically a linear approximation to the mode of collinear points)
    """
    result = np.zeros(box_map.shape,dtype=int)
    for line_param in line_params:
        label = line_param[0]
        angle = np.radians(line_param[1])
        dist = line_param[2]
        m = -np.cos(angle)/np.sin(angle)
        b = dist/np.sin(angle)
        x = np.arange(0,box_map.shape[1])
        y = m*x + b 
        for index,x_pt in enumerate(x):
            y_pt = y[index]
            if y_pt < 0 or y_pt >= 1000:
                continue
            # We only want lines to be drawn inside their respective boxes
            if box_map[int(x_pt),int(y_pt)] != label:
                continue
            result[int(x_pt),int(y_pt)] = label
            # the order of the x and y coordinates confused me for quite a while,
            # but this seems to give good answers
    return result 

def get_angles(delta,box_map,labels):
    """
    Given a segmentation of the height map (box_map), 
    and the height map itself (delta), we compute
    the Hough transform of the lines in each box,
    take the most popular line in each box and add to our list.

    Input:
        delta - filtered height map
        box_map - segmented height map (each object has its own box)
        labels - list of labels of each box in box_map
    Output:
        A list of tuples of the form (label, angle, distance from origin)
        representing parameterizations of the most highly voted line
        in the *label* box of box_map
    """
    result = []
    for label in labels:
        transform, angles, bins = houghtf(np.where(box_map==label,delta,np.zeros(delta.shape)))
        angle_index = np.argmax(transform)
        angle = angles[(angle_index%transform.shape[1])]
        opposite = angle - 90
        if opposite < -90:
            opposite += 180
        dist = bins[(angle_index/transform.shape[1])]
        result.append((label,angle,dist))
    return result

def houghtf(img, angles=None):
    """Perform the straight line Hough transform.

    Input:
      img - a boolean array
      angles - in degrees

    Output:
      H - the Hough transform coefficients
      distances
      angles
    
    """
    if img.ndim != 2:
        raise ValueError("Input must be a two-dimensional array")

    img = img.astype(bool)
    
    if not angles:
        angles = np.linspace(-90,90,180)

    theta = angles / 180. * np.pi
    d = np.ceil(np.hypot(*img.shape))
    nr_bins = 2*d - 1
    bins = np.linspace(-d,d,nr_bins)
    out = np.zeros((nr_bins,len(theta)),dtype=np.uint16)

    rows,cols = img.shape
    x,y = np.mgrid[:rows,:cols]

    for i,(cT,sT) in enumerate(zip(np.cos(theta),np.sin(theta))):
        rho = np.round_(cT*x[img] + sT*y[img]) - bins[0] + 1
        rho = rho.astype(np.uint16)
        rho[(rho < 0) | (rho > nr_bins)] = 0
        bc = np.bincount(rho.flat)[1:]
        out[:len(bc),i] = bc

    return out,angles,bins

def create_boxes(pano,labels):
    """
    Given a labeled image (either height map or panorama, both cases occur in this code),
    draw a box around each labeled pixel group

    Input:
        pano - segmented (labeled) image, ideally sparsely populated
        labels - list of labels for said image
    Output:
        An array representing the coarser segmentation of boxes rather than classifying
        point by point
    """
    result = np.zeros(pano.shape,dtype=int)
    for label in labels:
        if len(pano[np.where(pano==label)])==0:
            continue
        upper_left = pano.shape 
        lower_right = (0,0)
        for multi_index,value in np.ndenumerate(pano):
            if value != label:
                continue
            else:
                (y,x) = multi_index
                if y < upper_left[0]: 
                    upper_left = y,upper_left[1]
                if y > lower_right[0]:
                    lower_right = y,lower_right[1]
                if x < upper_left[1]:
                    upper_left = upper_left[0],x
                if x > lower_right[1]:
                    lower_right = lower_right[0],x
        result[upper_left[0]:lower_right[0],upper_left[1]:lower_right[1]] = label
    return result

def transform_labels(labeled,buff):
    """
    Take a labeled height map, and the buffer containing references describing, 
    for each nonzero pixel in the height map, the pixel in the panorama it corresponds to.

    Input:
        labeled - an x-y height map that has been previously labeled
        buff - list containing coords of the form (pixel coord in pano, pixel coord in height map)

    Output:
        An array representing a panorama with the x-y labeled pixels transformed into the panorama
    """
    result = np.zeros((500,1000),dtype=int)
    for y in range(labeled.shape[0]):
        for x in range(labeled.shape[1]):
            if labeled[y,x] > 0 and not buff[y,x][0] == 0:
                result[buff[y,x][0],buff[y,x][1]] = labeled[y,x]
    return result

def get_delta_map(H):
    """
    Filter a height map by replacing each pixel with its local standard deviation,
    and then providing some threshold to remove noisy areas that don't correspond to
    objects

    Input:
        H - height map
    Output:
        Array of filtered height
    """

    h = H.shape[0]
    w = H.shape[1]

    delta = np.zeros((h,w), dtype=float)
    for y in range(5,h-5,1):
        for x in range(5,w-5,1):
            std = np.std(H[y-2:y+2,x-2:x+2])
            if (std < 0.3):
                continue
            delta[y,x] = std
    return delta

def get_ground_plane(equations):
    ground = (0,0,0,1)
    for equation in equations:
        if float(equation[2]) > float(ground[2]):
              # we define ground plane to be plane with largest z coord
              ground = equation
    return ground

def segment_image(H):
    """
    First, we compensate for sparse depth data by dilating and then closing the image.
    Then, we label the result (a simple segmentation technique)
    Note that we are using a binary dilation and closing, even though the input image
    is in general not binary (not for any particular reason, will probably change).

    Input:
        H - filtered or unfiltered height map
    Output:
        Segmented image 
    """

    H_dil = ndimage.morphology.binary_dilation(H,[[1,1,1],[1,1,1],[1,1,1]],iterations=2)
    H_close = ndimage.morphology.binary_closing(H_dil,[[1,1,1],[1,1,1],[1,1,1]],iterations=2)
    H_return, num = ndimage.label(H_close,[[1,1,1], [1,1,1], [1,1,1]])
    valid_labels = []
    for label in range(1,num):
        if len(H_return[np.where(H_return==label)]) < 300:
            H_return[np.where(H_return==label)] = 0
        else:
            valid_labels.append(label)
    return H_return, valid_labels

def get_height_map(equations,plane,A):
    """
    Sample a depth panorama and transform each pixel into an x-y coordinate system,
    where the value of the array at any pixel is its distance from the ground plane.
    While we do so, we note, for each pixel in the panorama, its corresponding x-y
    coordinates, for easy retrieval later.
    Note: we only care about points that are within 10 meters of camera, because
    anything farther than that will not be recognizable, and so needs no blurring.

    Input:
        equations - list of tuples that contain parametrizations of each plane in plane pano.
        plane - plane panorama
        A - depth panorama
    Output:
        An array representing a top-down view of the scene 10 meters immediately around 
        the camera, a list of coords of the form (pano pixel coord, x-y pixel coord).

    Note: Presumably, height corresponds to distance from ground plane, but this is not
    always true. TODO: Make more robust to ground plane errors.
    
    """
    w = A.shape[1]
    h = A.shape[0]

    ground = get_ground_plane(equations)
    H = np.zeros((1000,1000),dtype=float)
    buff = np.zeros((1000,1000,2),dtype=int)
    for y in range(h):
        for x in range(w):
            if (plane[y,x] == 255) or (equations[plane[y,x]] != ground):
                continue

            r = float(A[y,x])
            if (r > 10): 
                continue

            yaw = (2*pi * x/w) - pi
            pitch = (-pi * y/h) + pi/2.0

            theta = -pitch + pi/2.0
            phi = -yaw + pi/2.0

            X = sin(theta) * cos(phi)
            Y = sin(theta) * sin(phi)
            Z = cos(theta)

            height = (float(ground[0])*r*X + float(ground[1])*r*Y + float(ground[2])*r*Z) + float(ground[3])
            coord_y = int(r*Y*50 + H.shape[0]/2) 
            coord_x = int(r*X*50 + H.shape[1]/2)
            # center coordinates (note the factor of 50 in the coordinate - that's effectively "zooming in"
            # in picture, giving higher resolution)
            if (height > 1 or height < .2):
                continue
            #print "H[y,x] = ", coord_y, coord_x, height
            H[coord_y,coord_x] = float(abs(height))
            buff[coord_y,coord_x,0] = y
            buff[coord_y,coord_x,1] = x

    return H, buff

if __name__ == '__main__':
    import sys
    from os import path

    if len(sys.argv) < 2:
        print "Usage: one argument - directory containing depth and plane panoramas and plane palette files"
        sys.exit(1)
    in_f = path.join(sys.argv[1], 'depth.png')
    in_f_plane = path.join(sys.argv[1], 'plane_pano.png')
    in_pln = path.join(sys.argv[1], 'plane_palette.txt')
    out_f = path.join(sys.argv[1],'heightmap.png')

    process_depth_map(in_f, in_f_plane, in_pln, out_f)
