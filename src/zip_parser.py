# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 22:50:08 2025
@author: oscar
"""
from pathlib import Path
import numpy as np
import cv2

from sklearn.neighbors import KNeighborsClassifier

def _detect_and_crop_grid(img): 
    '''Crops a raw image of a grid and counts its number of rows 
    and columns.'''
    
    # Convert image to grayscale. Then invert, threshold and dilate it
    # to highlight vertical and horizontal lines. 
    img = img.copy()
    gray = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th_50 = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    th_50 = cv2.dilate(th_50, np.ones((5, 5)))
    
    # Detect vertical lines using a morphological opening with a tall kernel. 
    v_kernel = np.ones((th_50.shape[0]//2, 1))
    th_v = cv2.morphologyEx(th_50, cv2.MORPH_OPEN, v_kernel)
    _, _, _, x = cv2.connectedComponentsWithStats(th_v)
    x = np.sort([xj[0] for xj in x[1:]])
    
    # Detect horizontal lines using a morphological opening with a wide kernel. 
    h_kernel = np.ones((1, th_50.shape[1]//2))
    th_h = cv2.morphologyEx(th_50, cv2.MORPH_OPEN, h_kernel)
    _, _, _, y = cv2.connectedComponentsWithStats(th_h)
    y = np.sort([yi[1] for yi in y[1:]])
    
    assert x.size == y.size, 'Could not detect square grid.'
    assert 0.99 < x.ptp() / y.ptp() < 1.01 , 'Could not detect square grid.'
    
    # Crop the image and return the number of rows and columns in the grid. 
    img = img[round(y.min()):round(y.max())+1, \
              round(x.min()):round(x.max())+1]  
    n = x.size - 1
    return img, n
    
def _preprocess_checkpoint(s):
    '''Preprocesses an image of a checkpoint number into a list of 
    images of its digits.'''
    
    # Black out the borders of the image, to highlight the white number
    # in the middle. 
    s = s.copy()
    sy, sx = s.shape
    cv2.floodFill(s, None, (0,0), 0)
    cv2.floodFill(s, None, (0,sy-1), 0)
    cv2.floodFill(s, None, (sx-1,0), 0)
    cv2.floodFill(s, None, (sx-1,sy-1), 0)
    
    # Detect the contours corresponding to digits, and sort them from 
    # left to right. 
    digits = []
    contours, _ = cv2.findContours(s, cv2.RETR_EXTERNAL, 2)
    contours = [cnt for cnt in contours if cv2.boundingRect(cnt)[3] > 0.25*s.shape[0]]
    contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[0])    
    
    for cnt in contours:
        # For each contour, scale it evenly so that its height is 24px. Add 
        # black borders to make the image 32x32.
        x, y, w, h = cv2.boundingRect(cnt)
        d = s[y:y+h, x:x+w]
        d = cv2.resize(d, (round(24*w/h), 24))
        d = cv2.copyMakeBorder(d, 0, 8, 0, 32-d.shape[1], cv2.BORDER_CONSTANT, 0)
        
        # Shift the center of mass of the digit to the center of the image. 
        M = cv2.moments(d)
        cx = round(M['m10']/M['m00'])
        cy = round(M['m01']/M['m00'])
        T = np.float32([[1,0,16-cx], [0,1,16-cy]])
        d = cv2.warpAffine(d, T, d.shape)
        digits.append(d)
    
    return digits

def _classify_digit(d):
    '''Classifies a digit image.'''
    
    # Read the training set of labeled images. 
    files = list(Path('datasets/neighbors').glob('*'))
    X = [cv2.imread(f, cv2.IMREAD_GRAYSCALE).flatten() for f in files]
    y = [f.stem[-1] for f in files]
    
    # Train a KNN classifier with K=1. Since digits are read from a 
    # screenshot and have a constant font, this model is preferred 
    # over a more complex one, like a neural network. 
    clf = KNeighborsClassifier(n_neighbors=1).fit(X, y)
    return clf.predict([d.flatten()])[0]
            
def parse_zip_img(img):
    '''Parses a raw image of a grid and indicates the 
    locations of its checkpoints and walls.''' 
    
    # Crop the image and count its number of rows and columns.
    img = img.copy()
    img, n = _detect_and_crop_grid(img)
    L = img.shape[0] / n
    
    # Convert image to grayscale. Then invert and threshold it to highlight
    # checkpoints and walls in the grid. 
    gray = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th_100 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    checkpoints = []
    for i in range(n):
        for j in range(n):
            # For each square cell in the grid, check if it contains a
            # checkpoint. If it does, preprocess the region of the image 
            # with the checkpoint and detect the number in it. 
            y1, y2 = round((i+0.2)*L), round((i+0.8)*L)
            x1, x2 = round((j+0.2)*L), round((j+0.8)*L)
            s = th_100[y1:y2+1, x1:x2+1]   
            
            if np.count_nonzero(s) > 0.33*s.size:
                digits = _preprocess_checkpoint(255 - s)
                labels = [_classify_digit(d) for d in digits]
                label = int(''.join(labels))
                checkpoints.append((label, i, j))
    
    # Verify that the numbers found in checkpoints form a set of consecutive 
    # integers starting at 1. Then sort the coordinates of these checkpoints
    # by its number. 
    labels_found = {c[0] for c in checkpoints}
    labels_true = set(range(1, len(checkpoints)+1))
    assert labels_found == labels_true, 'Could not find checkpoints.'
    checkpoints = [(i,j) for label, i, j in sorted(checkpoints)]
    
    walls = []
    for i in range(n):
        for j in range(1, n):
            # For each pair of cells with a vertical wall between them, 
            # check if they are divided by a wall. 
            y1, y2 = round((i+0.2)*L), round((i+0.8)*L)
            x1, x2 = round((j-0.1)*L), round((j+0.1)*L)
            w = th_100[y1:y2+1, x1:x2+1]
            if np.count_nonzero(w) > 0.66*w.size: 
                walls.append((i,j-1,i,j))
            
            # For each pair of cells with a horizontal wall between them, 
            # check if they are divided by a wall. 
            y1, y2 = round((j-0.1)*L), round((j+0.1)*L)
            x1, x2 = round((i+0.2)*L), round((i+0.8)*L)
            w = th_100[y1:y2+1, x1:x2+1]
            if np.count_nonzero(w) > 0.66*w.size: 
                walls.append((j-1,i,j,i))
                
    return n, checkpoints, walls, img

def check_parse_zip_img(img, n, checkpoints, walls):
    '''Draws the elements found by the parse_zip_img function.'''
    
    img = img.copy()
    L = img.shape[0] / n
    
    # Write the checkpoint numbers. 
    for k, (i, j) in enumerate(checkpoints):
        i1, i2 = round((j+0.5)*L), round((i+0.5)*L)
        cv2.putText(img, str(k+1), (i1, i2), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255,0,0), 2, cv2.LINE_AA)
    
    # Draw vertical and horizontal walls between cells. 
    for i1, j1, i2, j2 in walls:
        if i1 == i2: 
            y1, y2 = (round((i2+0.2)*L), round((i2+0.8)*L))
            x1, x2 = (round((j2-0.1)*L), round((j2+0.1)*L))
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2, cv2.LINE_AA)
            
        if j1 == j2: 
            y1, y2 = round((i2-0.1)*L), round((i2+0.1)*L)
            x1, x2 = round((j2+0.2)*L), round((j2+0.8)*L)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2, cv2.LINE_AA)
        
    return img
    
def draw_zip_board(img, path):
    '''Draws the solution of the puzzle.''' 
    
    i, j = path[0]
    L = img.shape[0] / (max(path)[0] + 1)
    
    # Select a pretty color to draw to the path. 
    hue = img[round((i+0.2)*L), round((j+0.2)*L)]
    hsv = cv2.cvtColor(np.uint8([[hue]]), cv2.COLOR_BGR2HSV)[0,0]
    hsv[1] = 200
    hue = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0,0]

    # Draw the path as an open polyline.
    img2 = img.copy()
    lines = np.array([(round((j+0.5)*L), round((i+0.5)*L)) for i,j in path])
    cv2.polylines(img2, [lines], False, hue.tolist(), round(0.33*L), cv2.LINE_AA)
    img = cv2.addWeighted(img, 0.3, img2, 0.7, 0)
    
    return img
    