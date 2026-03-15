import cv2
import dlib
import numpy as np
import random
from scipy.interpolate import RegularGridInterpolator
from smooth import LandmarkSmoother


# 1. Load the detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_landmarks(image):
    # dlib works best on grayscale images
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = detector(gray)
    
    for face in faces:
        # Predict landmarks for the detected face region
        landmarks = predictor(gray, face)
        
        # Convert landmarks to a simple numpy array of (x, y) coordinates
        points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points.append((x, y))
            
        return np.array(points)
    return None

# # --- FOR A VIDEO (.mp4) ---
# cap = cv2.VideoCapture('video.mp4')
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret: break
    
#     landmarks = get_landmarks(frame)
#     if landmarks is not None:
#         # Now Person A can use these for Triangulation 
#         # and Person B can use them for TPS!
#         pass 

# cap.release()

# --- Delaunay Triangulation ---
def get_delaunay_triangles(rect, points):
    """
    Calculates Delaunay triangles for a given set of points within a rectangle.
    Returns a list of triangles, where each triangle is a list of 3 (x, y) tuples.
    """
    subdiv = cv2.Subdiv2D(rect)
    
    # Insert points into subdiv
    for p in points:
        subdiv.insert((int(p[0]), int(p[1])))

    triangle_list = subdiv.getTriangleList()
    delaunay_triangles = []

    for t in triangle_list:
        # Convert triangle array [x1, y1, x2, y2, x3, y3] into 3 points
        pts = [
            (int(t[0]), int(t[1])),
            (int(t[2]), int(t[3])),
            (int(t[4]), int(t[5]))
        ]

        # Only include triangles where all vertices are within the image boundaries
        # Subdiv2D uses large 'dummy' points outside the rect which we must filter
        is_inside = True
        for p in pts:
            if not (rect[0] <= p[0] < rect[2] and rect[1] <= p[1] < rect[3]):
                is_inside = False
                break
        
        if is_inside:
            delaunay_triangles.append(pts)
            
    return delaunay_triangles

# --- Morphing ---

def get_barycentric_matrix(pts):
    """Constructs the matrix [[ax, ay, 1], [bx, by, 1], [cx, cy, 1]].T"""
    return np.array([
        [pts[0][0], pts[1][0], pts[2][0]],
        [pts[0][1], pts[1][1], pts[2][1]],
        [1, 1, 1]
    ])

def morph_triangle(img_src, img_dest, tri_src, tri_dest):
    """Warps a single triangle from src to dest using Barycentric coordinates."""
    
    # Get bounding box of the destination triangle to limit computation
    bbox = cv2.boundingRect(np.float32([tri_dest]))
    x_range = np.arange(bbox[0], bbox[0] + bbox[2])
    y_range = np.arange(bbox[1], bbox[1] + bbox[3])
    
    # Create a grid of points (x, y) within the bounding box
    xv, yv = np.meshgrid(x_range, y_range)
    pixel_coords = np.vstack([xv.ravel(), yv.ravel(), np.ones(xv.size)])

    # Step 1: Compute Barycentric coordinates (alpha, beta, gamma)
    # B_inv * [x, y, 1]^T = [alpha, beta, gamma]^T
    B_mat = get_barycentric_matrix(tri_dest)
    try:
        B_inv = np.linalg.inv(B_mat)
    except np.linalg.LinAlgError:
        return # Skip degenerate triangles

    bary_coords = B_inv @ pixel_coords
    
    # Check if point is inside triangle: alpha, beta, gamma >= 0 and sum approx 1
    # We use a small epsilon for floating point stability
    eps = 1e-5
    is_inside = np.all(bary_coords >= -eps, axis=0) & (np.sum(bary_coords, axis=0) <= 1 + eps)
    
    if not np.any(is_inside):
        return

    # Filter only pixels inside the triangle
    valid_bary = bary_coords[:, is_inside]
    dest_pixels_x = xv.ravel()[is_inside]
    dest_pixels_y = yv.ravel()[is_inside]

    # Step 2: Compute corresponding positions in Source Image A
    A_mat = get_barycentric_matrix(tri_src)
    src_coords = A_mat @ valid_bary
    
    # Convert from homogeneous (not strictly necessary here as z=1, but following instructions)
    src_x = src_coords[0, :] / src_coords[2, :]
    src_y = src_coords[1, :] / src_coords[2, :]

    # Step 3: Interpolate and Copy
    # We setup interpolators for each color channel
    h, w = img_src.shape[:2]
    # RegularGridInterpolator expects (y_coords, x_coords)
    fn_r = RegularGridInterpolator((np.arange(h), np.arange(w)), img_src[:,:,0], bounds_error=False, fill_value=0)
    fn_g = RegularGridInterpolator((np.arange(h), np.arange(w)), img_src[:,:,1], bounds_error=False, fill_value=0)
    fn_b = RegularGridInterpolator((np.arange(h), np.arange(w)), img_src[:,:,2], bounds_error=False, fill_value=0)

    # Note: src_y is 'row', src_x is 'col'
    query_pts = np.stack([src_y, src_x], axis=-1)
    
    img_dest[dest_pixels_y, dest_pixels_x, 0] = fn_r(query_pts)
    img_dest[dest_pixels_y, dest_pixels_x, 1] = fn_g(query_pts)
    img_dest[dest_pixels_y, dest_pixels_x, 2] = fn_b(query_pts)

def perform_morph(img_a, img_b, points_a, points_b, triangles):
    """
    img_a: Source image
    img_b: Target image (will be overwritten/warped into)
    points_a/b: Facial landmarks
    triangles: List of indices (i, j, k) from Delaunay
    """
    # Create an empty canvas for the output
    output_img = np.zeros_like(img_b)

    for tri_indices in triangles:
        # Get coordinates for this specific triangle
        t_a = [points_a[i] for i in tri_indices]
        t_b = [points_b[i] for i in tri_indices]
        
        morph_triangle(img_a, output_img, t_a, t_b)
        
    return output_img

# --- 1. Helper to map Subdiv2D coordinates to landmark indices ---
def get_triangle_indices(points, triangle_list):
    """
    Matches the (x,y) coordinates from getTriangleList to the indices 
    of the original landmark list.
    """
    indices = []
    for t in triangle_list:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        tri_indices = []
        for pt in [pt1, pt2, pt3]:
            # Find the index of the point in the original landmark list
            # We use a distance threshold for floating point matching
            dist = np.linalg.norm(np.array(points) - np.array(pt), axis=1)
            index = np.argmin(dist)
            tri_indices.append(index)
        
        indices.append(tri_indices)
    return indices

def replace_face(original_img, morphed_face_img):
    """
    Replaces the face in original_img with the face from morphed_face_img.
    Assumes morphed_face_img has the face in the correct position with a black background.
    """
    # 1. Create a grayscale version of the morphed image
    gray_morphed = cv2.cvtColor(morphed_face_img, cv2.COLOR_BGR2GRAY)
    
    # 2. Create a binary mask where the face is (anything not black)
    # Threshold 1 means anything above 'almost black' becomes white (255)
    _, mask = cv2.threshold(gray_morphed, 1, 255, cv2.THRESH_BINARY)
    
    # 3. Create the inverse mask (white everywhere EXCEPT the face)
    mask_inv = cv2.bitwise_not(mask)
    
    # 4. Black out the face area in the original image using the inverse mask
    # This 'clears the stage' for the new face
    bg = cv2.bitwise_and(original_img, original_img, mask=mask_inv)
    
    # 5. Extract only the face from the morphed image (it's already cropped, 
    # but this ensures we don't pick up any noise)
    fg = cv2.bitwise_and(morphed_face_img, morphed_face_img, mask=mask)
    
    # 6. Add the two images together
    combined = cv2.add(bg, fg)
    
    return combined

def replace_face_seamless(original_img, morphed_face_img, landmarks):
    """
    Blends the morphed face into the original image using Poisson cloning.
    'landmarks' should be the list of (x,y) points for the destination face.
    """
    # 1. Ensure dimensions match
    h, w = original_img.shape[:2]
    morphed_face_img = cv2.resize(morphed_face_img, (w, h))

    # 2. Create the mask (the area to be cloned)
    gray_morphed = cv2.cvtColor(morphed_face_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_morphed, 1, 255, cv2.THRESH_BINARY)

    # 3. Calculate the center of the face for cloning
    # We find the bounding box of the landmarks to find the true face center
    convexhull = cv2.convexHull(np.array(landmarks, dtype=np.int32))
    x, y, w_face, h_face = cv2.boundingRect(convexhull)
    center = (x + w_face // 2, y + h_face // 2)

    # 4. Perform Seamless Cloning
    # cv2.NORMAL_CLONE: Keeps the lighting of the destination image
    # cv2.MIXED_CLONE: Blends textures (better if there are glasses or hair)
    output = cv2.seamlessClone(morphed_face_img, original_img, mask, center, cv2.NORMAL_CLONE)

    return output

def process_video(source_img_path, target_video_path, output_path):
    # 1. Initialize detector/predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # 2. Prepare Source Image (The face to be swapped IN)
    img_a = cv2.imread(source_img_path)
    points_a = get_landmarks(img_a) 
    if points_a is None:
        print("Could not find a face in the source image.")
        return

    # 3. Calculate Delaunay Triangulation ONCE based on source face
    rect = (0, 0, img_a.shape[1], img_a.shape[0])
    subdiv = cv2.Subdiv2D(rect)
    for p in points_a:
        subdiv.insert((int(p[0]), int(p[1])))
    triangle_indices = get_triangle_indices(points_a, subdiv.getTriangleList())

    # 4. Initialize Video Capture and Writer
    cap = cv2.VideoCapture(target_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get landmarks for the person in the current video frame
        points_b = get_landmarks(frame)

        if points_b is not None:
            # Warp source face (img_a) to target face shape (frame)
            warped_face = perform_morph(img_a, frame, points_a, points_b, triangle_indices)
            
            # Blend the warped face into the video frame
            frame = replace_face_seamless(frame, warped_face, points_b)

        out.write(frame)
        
        # Optional: Show the video while processing
        cv2.imshow("Face Swap in Progress", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to {output_path}")

def process_video_smoothed(source_img_path, target_video_path, output_path):
    # 1. Initialize detector/predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # 2. Prepare Source Image (Static)
    img_a = cv2.imread(source_img_path)
    points_a = get_landmarks(img_a) 
    if points_a is None:
        print("Error: No face found in source image.")
        return

    # Calculate Delaunay once for the source
    rect = (0, 0, img_a.shape[1], img_a.shape[0])
    subdiv = cv2.Subdiv2D(rect)
    for p in points_a:
        subdiv.insert((int(p[0]), int(p[1])))
    triangle_indices = get_triangle_indices(points_a, subdiv.getTriangleList())

    # 3. Setup Video Input/Output
    cap = cv2.VideoCapture(target_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 4. Initialize the Smoother from smooth.py
    # This stays OUTSIDE the loop to maintain state
    kalman_smoother = LandmarkSmoother(num_landmarks=68)

    print("Processing video frames with Kalman smoothing...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect raw landmarks in current frame
        raw_points_b = get_landmarks(frame)

        if raw_points_b is not None:
            # APPLY SMOOTHING HERE
            # We pass raw detections to Kalman; it returns smoothed coordinates
            points_b = kalman_smoother.update(raw_points_b)
            
            # Warp and Swap
            warped_face = perform_morph(img_a, frame, points_a, points_b, triangle_indices)
            frame = replace_face_seamless(frame, warped_face, points_b)

        out.write(frame)
        
        # Show progress
        cv2.imshow("Smoothed Face Swap", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done!")

def main():
    # Load Images
    img_a = cv2.imread("./Scarlett.jpg")
    img_b = cv2.imread("./Rambo.jpg")
    
    # Get Landmarks (Assuming you have this function defined)
    # Both must have the SAME number of points (e.g., 68 from dlib)
    points_a = get_landmarks(img_a) 
    points_b = get_landmarks(img_b)

    # Calculate Delaunay Triangulation for Image A
    size_a = img_a.shape
    rect = (0, 0, size_a[1], size_a[0])
    subdiv = cv2.Subdiv2D(rect)
    
    for p in points_a:
        subdiv.insert((int(p[0]), int(p[1])))
    
    # This gives us raw coordinates [[x1, y1, x2, y2, x3, y3], ...]
    raw_triangles = subdiv.getTriangleList()
    
    # Convert coordinates to indices [ [index1, index2, index3], ... ]
    # We do this once so we can use the same connectivity for both faces
    triangle_indices = get_triangle_indices(points_a, raw_triangles)

    # Perform the Morph! 
    # This warps Image A into the shape of Image B
    warped_image = perform_morph(img_a, img_b, points_a, points_b, triangle_indices)

    # Save 
    cv2.imwrite("Morphed_Result.jpg", warped_image)

    img_orig = cv2.imread("./Rambo.jpg")
    img_morphed = cv2.imread("./Morphed_Result.jpg")
    final_result = replace_face(img_orig, img_morphed)
    final_result_seamless = replace_face_seamless(img_orig, img_morphed, points_b)
    cv2.imwrite("Final_Face_Swap.jpg", final_result)
    cv2.imwrite("Final_Face_Swap_Seamless.jpg", final_result_seamless)

if __name__ == "__main__":
    main()
    # process_video("./Scarlett.jpg", "./Test1.mp4", "./Output1.mp4")
    process_video_smoothed("./Scarlett.jpg", "./Test1.mp4", "./Output1_smoothed.mp4")