# Import libraries
from skimage import io
from skimage import exposure
import skimage
from skimage.color import rgb2gray
from skimage import transform
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform,measure,color
from skimage.color import rgb2gray
from skimage import transform


def part4():
    ''' Stitch two images together '''
    image0 = io.imread('im1.jpg', True)
    image1 = io.imread('im2.jpg', True)
    image00 = io.imread('im1.jpg')
    image11 = io.imread('im2.jpg')


    plt.figure(figsize=(8,12))
    plt.subplot(221),plt.imshow(image0,cmap='gray'),plt.title("First Image")
    plt.subplot(222),plt.imshow(image1,cmap='gray'),plt.title("Second Image")


    # -------- Feature detection and matching -----

    # TODO: Initiate ORB detector
    #reference:https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_orb.html
    from skimage.feature import ORB, match_descriptors
    # ...
    descriptor_extractor = ORB(n_keypoints=500)

    # TODO: Find the keypoints and descriptors
    # ...
    descriptor_extractor.detect_and_extract(image0)
    key1 = descriptor_extractor.keypoints
    des1 = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(image1)
    key2 = descriptor_extractor.keypoints
    des2 = descriptor_extractor.descriptors

    # TODO: initialize Brute-Force matcher and exclude outliers. See match descriptor function.
    # ...
    match = match_descriptors(des1, des2, cross_check = True)

    # -------- Transform estimation -------

    # TODO: Compute homography matrix using ransac and ProjectiveTransform
    from skimage.measure import ransac
    from skimage.transform import ProjectiveTransform
    # ...
    #reference:https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.ransac
    # model_robust, inliers = ransac ...
    src = key2[match[:, 1]][:, ::-1]#we need to reverse the cols
    dst = key1[match[:, 0]][:, ::-1]
    model_robust, inliers = ransac((src, dst),ProjectiveTransform, min_samples=5, residual_threshold=1.1,max_trials=1030)
    


    # ------------- Warping ----------------
    #Next, we produce the panorama itself. The first step is to find the shape of the output image by considering the extents of all warped images.

    r, c = image1.shape[:2]

    # Note that transformations take coordinates in
    # (x, y) format, not (row, column), in order to be
    # consistent with most literature.
    corners = np.array([[0, 0],
                        [0, r],
                        [c, 0],
                        [c, r]])

    # Warp the image corners to their new positions.
    warped_corners = model_robust(corners)

    # Find the extents of both the reference image and
    # the warped target image.
    all_corners = np.vstack((warped_corners, corners))

    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)

    output_shape = (corner_max - corner_min)
    output_shape = np.ceil(output_shape[::-1])

    # ----- Note: The images are now warped according to the estimated transformation model.

    # A shift is added to ensure that both images are visible in their entirety. Note that warp takes the inverse mapping as input.
    from skimage.color import gray2rgb
    from skimage.exposure import rescale_intensity
    from skimage.transform import warp
    from skimage.transform import SimilarityTransform

    offset = SimilarityTransform(translation=-corner_min)

    image0_ = warp(image0, offset.inverse,
                output_shape=output_shape)

    image1_ = warp(image1, (model_robust + offset).inverse,
                output_shape=output_shape)

    plt.subplot(223),plt.imshow(image0_, cmap="gray"),plt.title("Warped first image")
    plt.subplot(224),plt.imshow(image1_, cmap="gray"),plt.title("Warped second image")
    plt.show()

    #An alpha channel is added to the warped images before merging them into a single image:

    def add_alpha(image, background=-1):
        """Add an alpha layer to the image.

        The alpha layer is set to 1 for foreground
        and 0 for background.
        """
        rgb = gray2rgb(image)
        alpha = (image != background)
        return np.dstack((rgb, alpha))





    # TODO: add alpha to the image0 and image1
    # ...
    img0 = add_alpha(image0_)
    img1 = add_alpha(image1_)

    # TODO: merge the alpha added image (only change the next line)
    # merged = ...
    
    merged = img0 + img1

    alpha = merged[..., 3]
    merged /= np.maximum(alpha, 1)[..., np.newaxis]

    # The summed alpha layers give us an indication of
    # how many images were combined to make up each
    # pixel.  Divide by the number of images to get
    # an average.
    plt.figure(figsize=(10,8))
    plt.imshow(merged, cmap="gray")
    plt.show()


    
    
    from skimage.feature import plot_matches
    # TODO: randomly select 10 inlier matches and show them using plot_matches
    # ...
    inlier_idxs = np.where(inliers)[0]  # Get the indices of inliers
    if len(inlier_idxs) > 10:
        selected_idxs = np.random.choice(inlier_idxs, 10, replace=False)
    else:
        selected_idxs = inlier_idxs  # Use all inliers if there are 10 or less

    # Plot the selected matches
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plot_matches(ax, image00, image11, key1, key2, match[selected_idxs],keypoints_color=None)
    
    ax.set_title("Randomly Selected Inlier Matches")
    plt.show()
    


if __name__ == "__main__":
    part4()
