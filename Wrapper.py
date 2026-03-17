import cv2
import dlib
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
from scipy.interpolate import RegularGridInterpolator
from smooth import LandmarkSmoother
import torch

# Use GPU for tensor ops if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Wrapper] Using device: {DEVICE}")


# 1. Load the detector and predictor
cnn_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # CNN detector returns a 'mmod_rectangles' object
    faces = cnn_detector(gray, 1)
    
    for face in faces:
        # The actual rectangle is stored in the 'rect' attribute of the mmod_rectangle
        dlib_rect = face.rect
        
        # Predict landmarks using the same predictor
        landmarks = predictor(gray, dlib_rect)
        
        points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points.append((x, y))
            
        return np.array(points)
    return None

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
    # 1. Get bounding boxes
    r1 = cv2.boundingRect(np.float32([tri_src]))
    r2 = cv2.boundingRect(np.float32([tri_dest]))

    # 2. Offset triangles by the bounding box top-left corner
    tri1_cropped = []
    tri2_cropped = []
    for i in range(3):
        tri1_cropped.append(((tri_src[i][0] - r1[0]), (tri_src[i][1] - r1[1])))
        tri2_cropped.append(((tri_dest[i][0] - r2[0]), (tri_dest[i][1] - r2[1])))

    # 3. Create a mask for the triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri2_cropped), (1.0, 1.0, 1.0), 16, 0)

    # 4. Apply Affine Transform
    img1_cropped = img_src[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    
    warp_mat = cv2.getAffineTransform(np.float32(tri1_cropped), np.float32(tri2_cropped))
    img2_cropped = cv2.warpAffine(img1_cropped, warp_mat, (r2[2], r2[3]), None, 
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # 5. Paste the warped triangle into the destination image
    img_dest[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = \
        img_dest[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (1 - mask) + img2_cropped * mask

def perform_morph(img_a, img_b, points_a, points_b, triangles):
    """
    Warp img_a into the shape defined by points_b using Delaunay triangles.
    Uses GPU (PyTorch) for the final compositing step when CUDA is available.
    """
    h, w = img_b.shape[:2]
    output_img = np.zeros((h, w, 3), dtype=np.float32)

    for tri_indices in triangles:
        t_a = [points_a[i] for i in tri_indices]
        t_b = [points_b[i] for i in tri_indices]
        morph_triangle(img_a, output_img, t_a, t_b)

    output_img = np.clip(output_img, 0, 255).astype(np.uint8)
    return output_img


def perform_morph_gpu(img_a, img_b, points_a, points_b, triangles):
    """
    GPU-accelerated version of perform_morph.
    Triangle warping is done on CPU (OpenCV affine), but the mask compositing
    accumulation is batched on the GPU via PyTorch tensors.
    """
    h, w = img_b.shape[:2]

    # Pre-compute all (warped_patch, mask_patch, r2) on CPU in parallel
    patches = []
    with ThreadPoolExecutor() as ex:
        def _warp_tri(tri_indices):
            t_a = [points_a[i] for i in tri_indices]
            t_b = [points_b[i] for i in tri_indices]
            r1 = cv2.boundingRect(np.float32([t_a]))
            r2 = cv2.boundingRect(np.float32([t_b]))
            tri1_c = [(t_a[i][0] - r1[0], t_a[i][1] - r1[1]) for i in range(3)]
            tri2_c = [(t_b[i][0] - r2[0], t_b[i][1] - r2[1]) for i in range(3)]
            mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
            cv2.fillConvexPoly(mask, np.int32(tri2_c), (1.0, 1.0, 1.0), 16, 0)
            img1_c = img_a[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
            if img1_c.size == 0 or r2[2] == 0 or r2[3] == 0:
                return None
            warp_mat = cv2.getAffineTransform(np.float32(tri1_c), np.float32(tri2_c))
            warped = cv2.warpAffine(img1_c, warp_mat, (r2[2], r2[3]),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT_101).astype(np.float32)
            return (warped, mask, r2)

        patches = list(ex.map(_warp_tri, triangles))

    # Composite on GPU
    canvas = torch.zeros((h, w, 3), dtype=torch.float32, device=DEVICE)
    for item in patches:
        if item is None:
            continue
        warped, mask, r2 = item
        x, y, bw, bh = r2
        # Clamp to image bounds
        x2, y2 = min(x + bw, w), min(y + bh, h)
        bw_c, bh_c = x2 - x, y2 - y
        if bw_c <= 0 or bh_c <= 0:
            continue
        w_t = torch.from_numpy(warped[:bh_c, :bw_c]).to(DEVICE)
        m_t = torch.from_numpy(mask[:bh_c, :bw_c]).to(DEVICE)
        canvas[y:y2, x:x2] = canvas[y:y2, x:x2] * (1 - m_t) + w_t * m_t

    return canvas.cpu().numpy().clip(0, 255).astype(np.uint8)

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
            # Warp source face (img_a) to target face shape (frame) using GPU
            warped_face = perform_morph_gpu(img_a, frame, points_a, points_b, triangle_indices)

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


def process_video_smoothed(source_img_path, target_video_path, output_path,
                           num_workers: int = 4):
    """
    Process video with Kalman-smoothed landmarks and GPU-accelerated triangle
    compositing.  Frames are dispatched to a ThreadPoolExecutor so that
    landmark detection (CPU) and GPU warp/blend overlap with I/O.

    NOTE: dlib's CNN detector is NOT thread-safe, so landmark detection runs
    sequentially on the main thread; only the warp+blend step is parallelised.
    """
    # 1. Prepare source image
    img_a = cv2.imread(source_img_path)
    points_a = get_landmarks(img_a)
    if points_a is None:
        print("Error: No face found in source image.")
        return

    rect = (0, 0, img_a.shape[1], img_a.shape[0])
    subdiv = cv2.Subdiv2D(rect)
    for p in points_a:
        subdiv.insert((int(p[0]), int(p[1])))
    triangle_indices = get_triangle_indices(points_a, subdiv.getTriangleList())

    # 2. Video I/O
    cap = cv2.VideoCapture(target_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    kalman_smoother = LandmarkSmoother(num_landmarks=68)

    print(f"Processing video frames (device={DEVICE}, workers={num_workers})...")

    # 3. Producer/consumer: read frames sequentially, process in thread pool
    # We keep futures in order so the output video stays in sequence.
    from collections import OrderedDict
    futures = OrderedDict()
    frame_count = 0
    BATCH = num_workers * 2  # max in-flight futures

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Landmark detection must be sequential (dlib CNN is not thread-safe)
            raw_points_b = get_landmarks(frame)
            if raw_points_b is None:
                print(f"Skipped Frame {frame_count}: No face detected by dlib.")
                cv2.imwrite(f"debug_frame_{frame_count}_no_face.jpg", frame)
                out.write(frame)
                continue

            points_b = kalman_smoother.update(raw_points_b)

            # Submit GPU warp + seamless clone to thread pool
            fut = executor.submit(
                _warp_and_blend,
                frame, frame_count, img_a, points_a, points_b, triangle_indices
            )
            futures[frame_count] = fut

            # Drain completed futures to keep memory bounded
            if len(futures) >= BATCH:
                fid, f = next(iter(futures.items()))
                out.write(f.result())
                del futures[fid]

        # Drain remaining futures in order
        for fid in sorted(futures):
            out.write(futures[fid].result())

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done!")


def _warp_and_blend(frame, frame_count, img_a, points_a, points_b, triangle_indices):
    """GPU warp + seamless blend for a single frame (runs in thread pool)."""
    warped_face = perform_morph_gpu(img_a, frame, points_a, points_b, triangle_indices)

    gray_morphed = cv2.cvtColor(warped_face, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_morphed, 1, 255, cv2.THRESH_BINARY)

    non_zero_pixels = cv2.countNonZero(mask)
    if non_zero_pixels < 500:
        print(f"Skipped Frame {frame_count}: Mask nearly empty (pixels: {non_zero_pixels}).")
        cv2.imwrite(f"debug_frame_{frame_count}_bad_mask_original.jpg", frame)
        cv2.imwrite(f"debug_frame_{frame_count}_bad_mask_warped.jpg", warped_face)
        return frame

    try:
        return replace_face_seamless(frame, warped_face, points_b)
    except Exception as e:
        print(f"Skipped Frame {frame_count}: Seamless clone failed. Error: {e}")
        cv2.imwrite(f"debug_frame_{frame_count}_clone_crash.jpg", frame)
        cv2.imwrite(f"debug_frame_{frame_count}_clone_crash_mask.jpg", mask)
        return frame

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