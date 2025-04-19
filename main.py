import numpy as np
from skimage.measure import label
from skimage.morphology import binary_opening

star = np.load('stars.npy')

view_cross = np.array([[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[0,1,0,1,0],[1,0,0,0,1]])
view_plus = np.array([[0,0,1,0,0],[0,0,1,0,0],[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0]])
image_cross = binary_opening(star, view_cross)
image_plus = binary_opening(star, view_plus)
labeled_cross = label(image_cross)
labeled_plus = label(image_plus)

print(f"number of stars: {np.max(labeled_cross) + np.max(labeled_plus)}")
