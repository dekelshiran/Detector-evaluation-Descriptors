import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import part1


def sift_descriptor(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def orb_descriptor(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors


def akaze_descriptor(image):
    akaze = cv2.AKAZE_create()
    keypoints, descriptors = akaze.detectAndCompute(image, None)
    return keypoints, descriptors


def compute_matching_accuracy(descriptors1, descriptors2, keypoints1, keypoints2):
    if descriptors1.size == 0 or descriptors2.size == 0:
        return 0, 0
    # שימוש ב- BFMatcher (Brute Force Matcher) לצורך חיפוש התאמות בין descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # חישוב דיוק ההתאמה - יחס ההתאמות הנכונות
    correct_matches = 0
    treshold = 5
    for match in matches:
        idx1 = match.queryIdx
        idx2 = match.trainIdx
        if np.linalg.norm(np.array(keypoints1[idx1].pt) - np.array(keypoints2[idx2].pt)) < treshold:
            correct_matches += 1

    accuracy = correct_matches / len(matches)
    return 100 * accuracy, len(matches)


def algo_detector(algoName, image):
    if algoName == 'ORB':
        return orb_descriptor(image)
    elif algoName == 'AKAZE':
        return akaze_descriptor(image)
    elif algoName == 'SIFT':
        return sift_descriptor(image)
    else:
        raise ValueError("Error: algorithm does'nt exist")


def eval_time_and_data(image1, image2, algorithm):
    start_time = time.time()
    keypoints1, descriptors1 = algo_detector(algorithm, image1)
    keypoints2, descriptors2 = algo_detector(algorithm, image2)
    if descriptors1 is None:
        descriptors1 = np.array([])
    if descriptors2 is None:
        descriptors2 = np.array([])
    end_time = time.time()
    delta_time = end_time - start_time

    return delta_time, keypoints1, descriptors1, keypoints2, descriptors2


if __name__ == '__main__':
    distortion_types = [
        'rotate_30', 'rotate_70',
        'scale_2X', 'scale_5X',
        'gaussian_filter',
        'low_gaussian_noise', 'high_gaussian_noise'
    ]
    images_path = 'images'
    all_images = os.listdir(images_path)

    delta_time_dic = {}
    accuracy_dic = {}
    matches_size_dic = {}

    all_delta_time = {}
    all_accuracy = {}
    all_matches_size = {}
    all_algorithms = ['SIFT', 'ORB', 'AKAZE']

    for algo in all_algorithms:
        print('STARTING ALGORITHM: ' + algo)
        for image in all_images:
            img = cv2.imread(os.path.join('images', image), cv2.COLOR_BGR2GRAY)
            all_disorted_images = part1.run_all_operations(img)
            for transform in distortion_types:

                delta_time, og_keypoints, og_descriptors, keypoints, descriptors = eval_time_and_data(img,
                                                                                                      all_disorted_images[
                                                                                                          transform],
                                                                                                      algo)
                accuracy, matches_size = compute_matching_accuracy(og_descriptors, descriptors, og_keypoints, keypoints)

                if transform in delta_time_dic:
                    delta_time_dic[transform] += delta_time
                else:
                    delta_time_dic[transform] = delta_time
                if transform in accuracy_dic:
                    accuracy_dic[transform] += accuracy
                else:
                    accuracy_dic[transform] = accuracy
                if transform in matches_size_dic:
                    matches_size_dic[transform] += matches_size
                else:
                    matches_size_dic[transform] = matches_size

            delta_time_dic = {key: value / len(all_images) for key, value in delta_time_dic.items()}
            accuracy_dic = {key: value / len(all_images) for key, value in accuracy_dic.items()}
            matches_size_dic = {key: value / len(all_images) for key, value in matches_size_dic.items()}

            all_delta_time[algo] = delta_time_dic
            all_accuracy[algo] = accuracy_dic
            all_matches_size[algo] = matches_size_dic
            delta_time_dic = {}
            accuracy_dic = {}
            matches_size_dic = {}

    # Extract data
    algorithms = list(all_algorithms)  # ['SIFT', 'FAST', 'ORB', 'AKAZE']
    distortion_types = list(distortion_types)  # ['rotate_30', 'rotate_70', ...]
    num_algorithms = len(algorithms)
    num_distortions = len(distortion_types)

    # Organize the data
    values = [[all_delta_time[algo][dist] for dist in distortion_types] for algo in algorithms]

    # Plotting
    x = np.arange(num_distortions)  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each algorithm's bars
    for i, algo in enumerate(algorithms):
        ax.bar(x + i * width, values[i], width, label=algo)

    # Add labels, title, and legend
    ax.set_xlabel('Distortion Type')
    ax.set_ylabel('Detection time AVG')
    ax.set_title('Comparison of Detection time Across Algorithms')
    ax.set_xticks(x + width * (num_algorithms - 1) / 2)  # Center tick labels
    ax.set_xticklabels(distortion_types, rotation=45, ha='right')  # Rotate for clarity
    ax.legend()

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

    ###

    # Extract data
    algorithms = list(all_algorithms)  # ['SIFT', 'FAST', 'ORB', 'AKAZE']
    distortion_types = list(distortion_types)  # ['rotate_30', 'rotate_70', ...]
    num_algorithms = len(algorithms)
    num_distortions = len(distortion_types)

    # Organize the data
    values = [[all_accuracy[algo][dist] for dist in distortion_types] for algo in algorithms]

    # Plotting
    x = np.arange(num_distortions)  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each algorithm's bars
    for i, algo in enumerate(algorithms):
        ax.bar(x + i * width, values[i], width, label=algo)

    # Add labels, title, and legend
    ax.set_xlabel('Distortion Type')
    ax.set_ylabel('Accuracy AVG')
    ax.set_title('Comparison of Matching Accuracy Across Algorithms')
    ax.set_xticks(x + width * (num_algorithms - 1) / 2)  # Center tick labels
    ax.set_xticklabels(distortion_types, rotation=45, ha='right')  # Rotate for clarity
    ax.legend()

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()
