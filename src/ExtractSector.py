import numpy as np
import image_slicer
from PIL import Image
import warnings


def segmentation():

    try:
        # To make things cleaner:
        warnings.simplefilter(action='ignore', category=FutureWarning)

        # Left eye segmentation process:

        # Obtain eye sections
        image_slicer.slice('ProcessingImages/LeftSeg.png', 4)

        # Load eye sections previously obtained
        list_im_left = ['ProcessingImages/LeftSeg_01_01.png', 'ProcessingImages/LeftSeg_01_02.png',
                      'ProcessingImages/LeftSeg_02_01.png', 'ProcessingImages/LeftSeg_02_02.png']
        images_left = [Image.open(i).convert('L') for i in list_im_left]

        # Select the smallest image, then resize all the others so that the dimensions will match one another
        min_shape = sorted([(np.sum(i.size), i.size) for i in images_left])[0][1]
        images_comb_left = np.hstack((np.asarray(i.resize(min_shape)) for i in images_left))

        # Save the image containing the left eye segmentation
        images_comb = Image.fromarray(images_comb_left)
        images_comb.save('SegmentationOutput/Left_Segmentation.jpg')

        # Right eye segmentation process:

        # Obtain eye sections
        image_slicer.slice('ProcessingImages/RightSeg.png', 4)

        # Load eye sections previously obtained
        list_im_right = ['ProcessingImages/RightSeg_01_01.png', 'ProcessingImages/RightSeg_01_02.png',
                   'ProcessingImages/RightSeg_02_01.png', 'ProcessingImages/RightSeg_02_02.png']
        images_right = [Image.open(i).convert('L') for i in list_im_right]

        # Select the smallest image, then resize all the others so that the dimensions will match one another
        min_shape = sorted([(np.sum(i.size), i.size) for i in images_right])[0][1]
        images_comb_right = np.hstack((np.asarray(i.resize(min_shape)) for i in images_right))

        # Save the image containing the right eye segmentation
        images_comb = Image.fromarray(images_comb_right)
        images_comb.save('SegmentationOutput/Right_Segmentation.jpg')
    except FileNotFoundError:
        print("Error during segmentation phase. Try again")