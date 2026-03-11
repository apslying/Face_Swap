import cv2
import dlib
import numpy as np
import random
from scipy.interpolate import RegularGridInterpolator


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


def draw_delaunay(img, subdiv, color):
    # 1. Get the list of triangles
    # Each triangle is stored as [x1, y1, x2, y2, x3, y3]
    triangles = subdiv.getTriangleList()
    
    h, w = img.shape[:2]
    rect = (0, 0, w, h)

    for t in triangles:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        # 2. Check if the points are inside the image boundary
        # getTriangleList can return points outside the initial rect
        if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
            cv2.line(img, pt1, pt2, color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, color, 1, cv2.LINE_AA, 0)

def rect_contains(rect, point):
    """Checks if a point is inside a rectangle."""
    return rect[0] <= point[0] < rect[2] and rect[1] <= point[1] < rect[3]

# --- Main Implementation ---

# Create a blank image
img = np.zeros((600, 600, 3), dtype=np.uint8)
rect = (0, 0, 600, 600)

# Initialize Subdiv2D with the bounding rectangle
subdiv = cv2.Subdiv2D(rect)

# Generate some random points or use your own
points = []
for i in range(25):
    points.append((random.randint(50, 550), random.randint(50, 550)))

# Insert points into subdiv
for p in points:
    subdiv.insert(p)
    # Optional: Draw points
    cv2.circle(img, p, 3, (0, 0, 255), -1)

# Retrieve and draw the triangles
draw_delaunay(img, subdiv, (255, 255, 255))

cv2.imshow("Delaunay Triangulation", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

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

def main():
    img_path = "C:/Users/hengyiwu/Desktop/Research/vision_robotics/Face_Swap/Rambo.jpg"
    out_path = "C:/Users/hengyiwu/Desktop/Research/vision_robotics/Face_Swap/Rambo_landmarks.txt"
    save_path = "C:/Users/hengyiwu/Desktop/Research/vision_robotics/Face_Swap/Rambo_with_triangles.jpg"
    img = cv2.imread(img_path)
    size = img.shape
    rect = (0, 0, size[1], size[0])
    landmarks = get_landmarks(img)
    # print(landmarks)
    # 1. Draw the landmarks on the image
    for (x, y) in landmarks:
        # cv2.circle(image, center_coordinates, radius, color, thickness)
        # Color is BGR, so (0, 0, 255) is Bright Red
        cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

    with open(out_path, "w") as f:
        for landmark in landmarks:
            f.write(f"{landmark[0]} {landmark[1]}\n")

    # --- Delaunay Triangulation ---
    # 1. Initialize Subdiv2D
    subdiv = cv2.Subdiv2D(rect)

    # 2. Insert points into subdiv
    for p in landmarks:
        subdiv.insert((int(p[0]), int(p[1])))

    # 3. Get the list of triangles
    # Each triangle is returned as a list of 6 numbers: [x1, y1, x2, y2, x3, y3]
    triangle_list = subdiv.getTriangleList()

    # --- Draw Triangles and Dots ---
    for t in triangle_list:
        # Extract coordinates for the 3 vertices
        pts = [
            (int(t[0]), int(t[1])),
            (int(t[2]), int(t[3])),
            (int(t[4]), int(t[5]))
        ]
        
        # Check if the triangle vertices are within the image boundaries
        # (Subdiv2D sometimes creates triangles using "virtual" outer points)
        is_inside = True
        for p in pts:
            if not (0 <= p[0] < size[1] and 0 <= p[1] < size[0]):
                is_inside = False
                break
                
        if is_inside:
            # Draw the triangle edges (Green)
            cv2.line(img, pts[0], pts[1], (0, 255, 0), 1, cv2.LINE_AA)
            cv2.line(img, pts[1], pts[2], (0, 255, 0), 1, cv2.LINE_AA)
            cv2.line(img, pts[2], pts[0], (0, 255, 0), 1, cv2.LINE_AA)

    # Draw the landmarks as red dots on top
    for p in landmarks:
        cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
    # 3. Save the new image

    cv2.imwrite(save_path, img)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()