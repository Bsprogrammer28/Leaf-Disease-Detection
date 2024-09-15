import os
import cv2
import numpy as np
from scipy import ndimage

class LeafSegmentation:
    filter_size = (11, 11)
    filter_sigma = 5
    otsu_threshold_min = 0
    otsu_threshold_max = 255
    structuring_element = cv2.MORPH_RECT
    structuring_element_size = (15, 15)
    
    def __init__(self, rgb_folder_or_image):
        self.m_rgb_folder_or_image = str(rgb_folder_or_image)
        
    def segment(self):
        if os.path.isdir(self.m_rgb_folder_or_image):
            # If it's a folder, process all images
            for filename in os.listdir(self.m_rgb_folder_or_image):
                if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                    rgb_img_path = os.path.join(self.m_rgb_folder_or_image, filename)
                    rgb_img = cv2.imread(rgb_img_path)
                    self.process_and_show(rgb_img, filename)
        else:
            # If it's a single image
            rgb_img = cv2.imread(self.m_rgb_folder_or_image)
            if rgb_img is not None:
                self.process_and_show(rgb_img, os.path.basename(self.m_rgb_folder_or_image))
            else:
                print("Error: Could not read the image.")
    
    def process_and_show(self, rgb_img, image_name):
        start_time = cv2.getTickCount()
        resulted_img = self.perform_segmentation(rgb_img)
        end_time = cv2.getTickCount()
        elapsed_time = (end_time - start_time) / cv2.getTickFrequency()
        
        cv2.imshow(f'Original Image - {image_name}', rgb_img)
        cv2.imshow(f'Segmented Image - {image_name}', resulted_img)
        print(f'Elapsed time for {image_name}: {elapsed_time:.2f} s')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def histogram_equalization(self, img):
        return cv2.equalizeHist(img)

    def gaussian_filter(self, img):
        filtered = np.copy(img)
        filtered = cv2.GaussianBlur(img, self.filter_size, self.filter_sigma)
        return filtered
    
    def create_mask_using_otsu(self, img):
        _, mask = cv2.threshold(img, 
                                        self.otsu_threshold_min, 
                                        self.otsu_threshold_max,
                                        cv2.THRESH_OTSU)
        x, y = mask.shape
        for i in range(0, x, 1):
            for j in range(0, y, 1):
                if mask[i, j] == self.otsu_threshold_max:
                    mask[i, j] = self.otsu_threshold_min
                else:
                    mask[i, j] = self.otsu_threshold_max 
        return mask

    def improve_mask(self, mask):
        SE = cv2.getStructuringElement(self.structuring_element,
                                       self.structuring_element_size)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, SE)
        return ndimage.binary_fill_holes(mask) 

    def perform_segmentation(self, img):
        img_copy = np.copy(img)
        blue, _, _ = cv2.split(img)
        img_blue = self.histogram_equalization(blue)
        img_blue = self.gaussian_filter(img_blue)
        mask_blue = self.create_mask_using_otsu(img_blue)
        mask_blue = self.improve_mask(mask_blue)
        
        x, y = mask_blue.shape
        for i in range(0, x, 1):
            for j in range(0, y, 1):
                if mask_blue[i, j] == 0:
                    img_copy[i, j, :] = 0
        return img_copy