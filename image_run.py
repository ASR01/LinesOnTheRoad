import cv2
import numpy as np
import image_processing as ip


img = cv2.imread('./test_images/solidYellowCurve.jpg')


ip.process_img(img, 0)