<p align="center">
  <img src="assets/portada.png" width=100% />
</p>

![License](https://img.shields.io/github/license/Oscar-sandbox/linkedin-zip-solver)
![Python version](https://img.shields.io/badge/python-3.11-blue.svg) 
![OpenCV version](https://img.shields.io/badge/opencv-4.10-blue.svg)
![Networkx version](https://img.shields.io/badge/networkx-3.4-blue.svg)
![Sklearn version](https://img.shields.io/badge/sklearn-1.6-blue.svg)

# LinkedIn's Zip Puzzle Solver

## About the puzzle 
This project provides an algorithm for solving LinkedIn's Zip puzzle game automatically. The puzzle is a visual logic game 
where you need to plot a path through a grid while passing through numbered checkpoints in order. 
A tutorial on how to play can be found [here](https://www.linkedin.com/games/zip/). 

## Preprocessing and Computer Vision
First, the grid and its elements must be located in a raw image. The puzzle elements to identify are checkpoint numbers and walls 
between cells. To this end, several image processing techniques are used, like thresholding, morphological transformations and 
optical character recognition. This step is done using the popular computer vision library OpenCV, as seen in [zip_parser.py](src/zip_parser.py). 

| <img src="assets/original.png" align="middle"/> | <img src="assets/crop.png" align="middle"/> |  <img src="assets/check_parse.png" align="middle"/> | <img src="assets/solution_b.png" align="middle"/> |  
| :---: | :---: | :---: | :---: | 
| Original image | Detect and crop grid | Detect puzzle elements | Solution | 

## Backtracking and Graph Theory

<p align="center">
  <img src="assets/graph.png" width=60% />
</p>

## Results
The solver was tested on 20 puzzles, ranging from puzzle No. 68 to 88 (except No.79, because it is Christmas themed). With backtracking at depth 2, every puzzle was solved in under a third of a second, with an average solve time of 0.04s. Full results, with images, are available at [examples](examples). 
