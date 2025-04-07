# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 22:54:00 2025
@author: oscar
"""
from pathlib import Path
import cv2 

from zip_solver import ZipSolver
from zip_parser import parse_zip_img, draw_zip_board, check_parse_zip_img

for f in Path('examples/inputs').glob('*'):
    img = cv2.imread(f)
    
    n, checkpoints, walls, img = parse_zip_img(img)
    path = ZipSolver(depth=n**2).solve(n, checkpoints, walls)
    img = draw_zip_board(img, path)
    
    img = check_parse_zip_img(img, n, checkpoints, walls)
    cv2.imwrite(f'examples/outputs/solution_{f.name}', img)
