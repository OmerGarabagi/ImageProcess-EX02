# Omer Garabagi, 322471145
# Omer Chernia, 318678620

import cv2
import numpy as np
import os
import shutil

# matches is of (3|4 X 2 X 2) size. Each row is a match - pair of (kp1, kp2) where kpi = (x, y)
def get_transform(matches, is_affine):
    src_points, dst_points = matches[:, 0], matches[:, 1]
    print("Source Points:", src_points)
    print("Destination Points:", dst_points)
    
    if is_affine:
        # Compute affine transformation matrix
        T, _ = cv2.estimateAffine2D(src_points, dst_points)
    else:
        # Compute homography transformation matrix
        T, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
    print("Transformation Matrix:", T)
    
    return T

def stitch(img1, img2):
    # Ensure both images have the same size
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions for stitching.")
    
    # Combine images using maximum intensity blending
    result = np.maximum(img1, img2)
    return result

# Output size is (w, h)
def inverse_transform_target_image(target_img, original_transform, output_size):
    # Determine if the transformation is affine or projective
    is_affine = original_transform.shape == (2, 3)
    
    if is_affine:
        # Compute the inverse affine transformation matrix
        inverse_transform = cv2.invertAffineTransform(original_transform)
        inverse_transform = inverse_transform.astype(np.float32)
        # Perform the inverse affine transformation
        transformed_img = cv2.warpAffine(target_img, inverse_transform, output_size, flags=cv2.INTER_LINEAR)
    else:
        # Compute the inverse homography matrix
        inverse_transform = np.linalg.inv(original_transform)
        inverse_transform = inverse_transform.astype(np.float32)
        # Perform the inverse projective transformation
        transformed_img = cv2.warpPerspective(target_img, inverse_transform, output_size, flags=cv2.INTER_LINEAR)
    
    return transformed_img

# Returns list of pieces file names
def prepare_puzzle(puzzle_dir):
    edited = os.path.join(puzzle_dir, 'abs_pieces')
    if os.path.exists(edited):
        shutil.rmtree(edited)
    os.mkdir(edited)
    
    affine = 4 - int("affine" in puzzle_dir)
    
    matches_data = os.path.join(puzzle_dir, 'matches.txt')
    n_images = len(os.listdir(os.path.join(puzzle_dir, 'pieces')))

    matches = np.loadtxt(matches_data, dtype=np.int64).reshape(n_images - 1, affine, 2, 2)
    
    return matches, affine == 3, n_images


if __name__ == '__main__':
    lst = ['puzzle_affine_1', 'puzzle_affine_2', 'puzzle_homography_1']
    
    for puzzle_dir in lst:
        print(f'Starting {puzzle_dir}')
        
        puzzle = os.path.join('puzzles', puzzle_dir)
        
        final_puzzle = cv2.imread(os.path.join(puzzle, 'pieces', 'piece_1.jpg'))

        pieces_pth = os.path.join(puzzle, 'pieces')
        edited = os.path.join(puzzle, 'abs_pieces')
        
        matches, is_affine, n_images = prepare_puzzle(puzzle)

        for idx in range(1, n_images):
            piece = cv2.imread(os.path.join(pieces_pth, f'piece_{idx + 1}.jpg'))
            
            M = get_transform(matches=matches[idx - 1], is_affine=is_affine)
            M = M.astype(np.float32)
            transformed_img = inverse_transform_target_image(piece, M, final_puzzle.shape[:2][::-1])
            cv2.imwrite(os.path.join(edited, f'piece_absolute_{idx + 1}.jpg'), transformed_img)
            final_puzzle = stitch(final_puzzle, transformed_img)
            
        sol_file = 'solution.jpg'
        cv2.imwrite(os.path.join(puzzle, sol_file), final_puzzle)
