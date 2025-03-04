import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Harris Corner Detector
def harris_corner_detector_key_point(image):
    dst = cv2.cornerHarris(np.float32(image), 2, 3, 0.04)
    # Dilate to enhance the corner points
    dst = cv2.dilate(dst, None)

    # Apply threshold to identify the corners
    corners = dst > 0.05 * dst.max()

    # Get the coordinates of the corners
    corner_points = np.column_stack(np.where(corners))

    # Convert to cv2.KeyPoint objects (like FAST detector)
    # keypoints = [cv2.KeyPoint(pt)) for pt in corner_points]
    # print(keypoints)
    return corner_points
# FAST Detector
def fast_detector(image):
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(image, None)
    return keypoints

def sift_detector(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints

# AKAZE Detector
def akaze_detector(image):
    akaze = cv2.AKAZE_create()
    keypoints, descriptors = akaze.detectAndCompute(image, None)
    return keypoints

# ORB Detector
def orb_detector(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints

def choose_algo_detector(algoName, image):
    if algoName == 'FAST':
       return fast_detector(image)
    elif algoName == 'ORB':
        return orb_detector(image)
    elif algoName == 'AKAZE':
        return akaze_detector(image)
    elif algoName == 'SIFT':
        return sift_detector(image)
    elif algoName == 'HARRIS_CORNER':
        return harris_corner_detector_key_point(image)
    else:
        raise ValueError("Error: algorithm does'nt exist")

def rotate(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    return cv2.warpAffine(image, rotation_matrix, center), rotation_matrix

def scale(image, scale):
    height, width = image.shape[:2]
    scaled = cv2.resize(image, (height * scale, width * scale), interpolation=cv2.INTER_LINEAR)
    return scaled

def gaussian_filter(image, kernel_size, sigma=1):
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return np.array(blurred_image)

def gaussian_noise(image, mean=0, stddev=25):
    # Ensure the image is in float format for calculations
    image = image.astype(np.float32)

    # Generate Gaussian noise
    noise = np.random.normal(mean, stddev, image.shape)

    # Add noise to the image
    noisy_image = image + noise

    # Clip the values to [0, 255] and convert back to uint8
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return np.array(noisy_image)

def rotation_repeatability(image, angle, algorithm):
    # Detect keypoints in the original image
    keypoints_original = choose_algo_detector(algorithm, image)
    keypoints_original = np.array([kp.pt for kp in keypoints_original])  # Extract coordinates

    rotated_image, rotation_matrix = rotate(image, angle)

    # Detect keypoints in the rotated image
    keypoints_rotated = choose_algo_detector(algorithm, rotated_image)
    keypoints_rotated = np.array([kp.pt for kp in keypoints_rotated])  # Extract coordinates

    # Transform original keypoints using the rotation matrix
    original_points_homogeneous = np.hstack((keypoints_original, np.ones((keypoints_original.shape[0], 1))))
    transformed_points = np.dot(rotation_matrix, original_points_homogeneous.T).T

    return transformed_points, keypoints_rotated


def gaussian_filter_transform_keypoints(keypoints, kernel_size=(3, 3), sigma=3.0):
    # Extract the x, y coordinates of the keypoints
    keypoints_coords = np.array([kp.pt for kp in keypoints], dtype=np.float32)

    # Create a Gaussian kernel (this will act like the filter)
    kernel = cv2.getGaussianKernel(kernel_size[0], sigma)  # 1D Gaussian kernel
    kernel_2d = kernel @ kernel.T  # 2D Gaussian kernel

    # Apply the Gaussian filter to the keypoint coordinates
    filtered_coords = cv2.filter2D(keypoints_coords, -1, kernel_2d)

    return filtered_coords

def compute_repeatability(keypoints1, keypoints2, threshold=5):
    valid_matches = 0
    keypoints2 = keypoints2.tolist()
    for kp1 in keypoints1:
        if len(keypoints2) == 0:
            break

        # Compute distances to all remaining points in keypoints2
        distances = np.linalg.norm(keypoints2 - kp1, axis=1)

        # Find the closest point in keypoints2
        min_index = np.argmin(distances)
        min_distance = distances[min_index]

        # Check if the closest point is within the threshold
        if min_distance <= threshold:
            valid_matches += 1
            # Remove the matched point from keypoints2
            keypoints2.pop(min_index)

    min_size = min(len(keypoints1), len(keypoints2) + valid_matches)  # Adjust for removed matches
    avg_valid_count = valid_matches / min_size if min_size > 0 else 0

    return round(avg_valid_count * 100)

# compute_kps: get the type of tranformation, image and wanted algorithm
# return 2 arrays with the keypoints after tranformation- kp_og the wanted value, kp_trans the value of the algorithm
def compute_kps(tranformation, data=[]):
    if tranformation == 'rotate_30':
        kp_og, kp_trans = rotation_repeatability(data['image'], 30, data['algorithm'])
    elif tranformation == 'rotate_70':
        kp_og, kp_trans = rotation_repeatability(data['image'], 70, data['algorithm'])
    elif tranformation == 'scale_2X':
        scaled_img = scale(data['image'], 2)
        kp_trans = choose_algo_detector(data['algorithm'], scaled_img)
        kp_trans = np.array([(kp.pt[0], kp.pt[1]) for kp in kp_trans])
        kp_og = choose_algo_detector(data['algorithm'], data['image'])
        kp_og = np.array([(kp.pt[0] * 2, kp.pt[1] * 2) for kp in kp_og])
    elif tranformation == 'scale_5X':
        scaled_img = scale(data['image'], 5)
        kp_trans = choose_algo_detector(data['algorithm'], scaled_img)
        kp_trans = np.array([(kp.pt[0], kp.pt[1]) for kp in kp_trans])
        kp_og = choose_algo_detector(data['algorithm'], data['image'])
        kp_og = np.array([(kp.pt[0] * 5, kp.pt[1] * 5) for kp in kp_og])
    elif tranformation == 'gaussian_filter':
        kp_trans = choose_algo_detector(data['algorithm'], gaussian_filter(data['image'], 5))
        kp_trans = np.array([(kp.pt[0], kp.pt[1]) for kp in kp_trans])
        kp_og = choose_algo_detector(data['algorithm'], data['image'])
        kp_og = gaussian_filter_transform_keypoints(kp_og)
    elif tranformation == 'low_gaussian_noise':
        kp_trans = choose_algo_detector(data['algorithm'], gaussian_noise(data['image'], stddev=10))
        kp_trans = np.array([(kp.pt[0], kp.pt[1]) for kp in kp_trans])
        kp_og = choose_algo_detector(data['algorithm'], data['image'])
        noisy_keypoints = []
        for kp in kp_og:
            # Generate Gaussian noise for x and y positions
            noise_x = np.random.normal(0, 10)
            noise_y = np.random.normal(0, 10)

            # Apply the noise to the keypoint's position
            noisy_kp = cv2.KeyPoint(kp.pt[0] + noise_x, kp.pt[1] + noise_y, kp.size)
            noisy_keypoints.append(noisy_kp)
        kp_og = np.array([(kp.pt[0], kp.pt[1]) for kp in noisy_keypoints])
    elif tranformation == 'high_gaussian_noise':
        kp_trans = choose_algo_detector(data['algorithm'], gaussian_noise(data['image'], stddev=25))
        kp_trans = np.array([(kp.pt[0], kp.pt[1]) for kp in kp_trans])
        kp_og = choose_algo_detector(data['algorithm'], data['image'])
        noisy_keypoints = []
        for kp in kp_og:
            # Generate Gaussian noise for x and y positions
            noise_x = np.random.normal(0, 25)
            noise_y = np.random.normal(0, 25)

            # Apply the noise to the keypoint's position
            noisy_kp = cv2.KeyPoint(kp.pt[0] + noise_x, kp.pt[1] + noise_y, kp.size)
            noisy_keypoints.append(noisy_kp)
        kp_og = np.array([(kp.pt[0], kp.pt[1]) for kp in noisy_keypoints])
    else:
        raise ValueError("Error: Transformation doesn't exist")

    return kp_og, kp_trans

# returns the avg of the location error
def compute_location_error(keypoints1, keypoints2):
    closest_distances = []
    keypoints2 = keypoints2.tolist()
    for kp1 in keypoints1:
        if len(keypoints2) == 0:
            break
        # Compute distances to all remaining points in keypoints2
        distances = np.linalg.norm(keypoints2 - kp1, axis=1)
        # Find the closest point in keypoints2
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        closest_distances.append(min_distance)
        keypoints2.pop(min_index)

    if len(closest_distances) == 0:
        return 0
    return sum(closest_distances) / len(closest_distances)

def run_all_operations(img, all_operations={}):
    all_operations['rotate_30'] = rotate(img, 30)[0]
    all_operations['rotate_70'] = rotate(img, 70)[0]
    all_operations['scale_2X'] = scale(img, 2)
    all_operations['scale_5X'] = scale(img, 5)
    all_operations['gaussian_filter'] = gaussian_filter(img, 3)
    all_operations['low_gaussian_noise'] = gaussian_noise(img, 0, 5)
    all_operations['high_gaussian_noise'] = gaussian_noise(img)

    return all_operations


# run the test
if __name__ == '__main__':

    all_trasformations_names = [
        'rotate_30', 'rotate_70',
        'scale_2X', 'scale_5X',
        'gaussian_filter',
        'low_gaussian_noise', 'high_gaussian_noise'
    ]

    all_images = os.listdir('images')
    repeat = {}
    loc_error = {}
    data = {}
    all_repeat = {}
    all_loc_error = {}
    all_algorithms = ['SIFT', 'FAST', 'ORB', 'AKAZE']
    for algo in all_algorithms:
        data['algorithm'] = algo
        print('STARTING ALGORITHM: ' + algo)
        for image in all_images:
            img = cv2.imread(os.path.join('images', image), cv2.COLOR_BGR2GRAY)
            data['image'] = img
            for transform in all_trasformations_names:
                kp_og, kp_trans = compute_kps(transform, data)
                if transform in repeat:
                    repeat[transform] += compute_repeatability(kp_og, kp_trans)
                else:
                    repeat[transform] = compute_repeatability(kp_og, kp_trans)

                if transform in loc_error:
                    loc_error[transform] += compute_location_error(kp_og, kp_trans)
                else:
                    loc_error[transform] = compute_location_error(kp_og, kp_trans)
            repeat = {key: value / len(all_images) for key, value in repeat.items()}
            loc_error = {key: value / len(all_images) for key, value in loc_error.items()}
            all_repeat[algo] = repeat
            all_loc_error[algo] = loc_error
            repeat = {}
            loc_error = {}

    # Extract data
    algorithms = list(all_repeat.keys())  # ['SIFT', FAST', 'ORB', 'AKAZE']
    distortion_types = list(all_repeat['SIFT'].keys())  # ['rotate_30', 'rotate_70', ...]
    num_algorithms = len(algorithms)
    num_distortions = len(distortion_types)

    # Organize the data
    values = [[all_repeat[algo][dist] for dist in distortion_types] for algo in algorithms]

    # Plotting
    x = np.arange(num_distortions)  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each algorithm's bars
    for i, algo in enumerate(algorithms):
        ax.bar(x + i * width, values[i], width, label=algo)

    # Add labels, title, and legend
    ax.set_xlabel('Distortion Type')
    ax.set_ylabel('Repeatability Score')
    ax.set_title('Comparison of Repeatability Scores Across Algorithms')
    ax.set_xticks(x + width * (num_algorithms - 1) / 2)  # Center tick labels
    ax.set_xticklabels(distortion_types, rotation=45, ha='right')  # Rotate for clarity
    ax.legend()

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

#################################

 # Extract data
    algorithms = list(all_loc_error.keys())  # ['SIFT', 'FAST', 'ORB', 'AKAZE']
    distortion_types = list(all_loc_error['SIFT'].keys())  # ['rotate_30', 'rotate_70', ...]
    num_algorithms = len(algorithms)
    num_distortions = len(distortion_types)

    # Organize the data
    values = [[all_loc_error[algo][dist] for dist in distortion_types] for algo in algorithms]

    # Plotting
    x = np.arange(num_distortions)  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each algorithm's bars
    for i, algo in enumerate(algorithms):
        ax.bar(x + i * width, values[i], width, label=algo)

    # Add labels, title, and legend
    ax.set_xlabel('Distortion Type')
    ax.set_ylabel('Location error AVG')
    ax.set_title('Comparison of Location Error Across Algorithms')
    ax.set_xticks(x + width * (num_algorithms - 1) / 2)  # Center tick labels
    ax.set_xticklabels(distortion_types, rotation=45, ha='right')  # Rotate for clarity
    ax.legend()

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()