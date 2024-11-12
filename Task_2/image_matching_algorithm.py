import rasterio
from rasterio.plot import reshape_as_image
import matplotlib.pyplot as plt
import cv2
import os
import argparse

extract_dir = "Task_2/dataset"

def read_image(image_path):
    with rasterio.open(os.path.join(extract_dir, image_path), "r", driver='JP2OpenJPEG') as src:
        raster_img = src.read()

    grayscale_img = cv2.cvtColor(reshape_as_image(raster_img), cv2.COLOR_RGB2GRAY)
    return grayscale_img

def resize_with_padding(image, target_size=(1024, 1024)):
    h, w = image.shape
    new_h, new_w = target_size
    
    scale = min(new_h / h, new_w / w)
    resized_img = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    return cv2.copyMakeBorder(resized_img, 0, new_h - resized_img.shape[0], 0, new_w - resized_img.shape[1], cv2.BORDER_CONSTANT, value=0)

def feature_extraction(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def feature_matching(descriptors1, descriptors2):
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []

    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return good_matches

def draw_matches(image1, image2, keypoints1, keypoints2, matches):
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(20, 20))
    plt.imshow(matched_image)
    plt.axis("off")
    plt.savefig("matched_image.png")
    plt.show()

def image_matching_algorithm(image1_path, image2_path):
    image1 = read_image(image1_path)
    image2 = read_image(image2_path)

    image1 = resize_with_padding(image1)
    image2 = resize_with_padding(image2)

    keypoints1, descriptors1 = feature_extraction(image1)
    keypoints2, descriptors2 = feature_extraction(image2)

    matches = feature_matching(descriptors1, descriptors2)

    image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)

    draw_matches(image1, image2, keypoints1, keypoints2, matches)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--first_image_path", type=str, help="Path to the first image")
    parser.add_argument("--second_image_path", type=str, help="Path to the second image")
    args = parser.parse_args()

    image_matching_algorithm(args.first_image_path, args.second_image_path) 
    