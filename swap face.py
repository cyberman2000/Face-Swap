import cv2
import dlib
import numpy as np
import imutils
from imutils import face_utils

def face_swap(source_img, target_img, output_path):
    # Initialize dlib's face detector and facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # مدل را از اینترنت دانلود کنید
    
    # Convert images to grayscale
    gray_source = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in both images
    rects_source = detector(gray_source, 1)
    rects_target = detector(gray_target, 1)
    
    if len(rects_source) == 0 or len(rects_target) == 0:
        print("No faces detected in one or both images!")
        return
    
    # Get facial landmarks
    shape_source = predictor(gray_source, rects_source[0])
    shape_source = face_utils.shape_to_np(shape_source)
    
    shape_target = predictor(gray_target, rects_target[0])
    shape_target = face_utils.shape_to_np(shape_target)
    
    # Find convex hull for the face in source image
    hull_source = []
    hull_index = cv2.convexHull(np.array(shape_source), returnPoints=False)
    
    for i in range(0, len(hull_index)):
        hull_source.append(shape_source[int(hull_index[i])])
    
    # Find convex hull for the face in target image
    hull_target = []
    for i in range(0, len(hull_index)):
        hull_target.append(shape_target[int(hull_index[i])])
    
    # Calculate Delaunay triangulation
    rect = cv2.boundingRect(np.float32([hull_target]))
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(hull_target)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    
    # Get the face mask for the target image
    hull_target_uint8 = []
    for i in range(0, len(hull_target)):
        hull_target_uint8.append((hull_target[i][0], hull_target[i][1]))
    
    mask = np.zeros_like(gray_target)
    cv2.fillConvexPoly(mask, np.int32(hull_target_uint8), 255)
    
    # Warp the source face to match the target face
    img_source_warped = np.zeros(target_img.shape, dtype=target_img.dtype)
    
    for triangle in triangles:
        # Get points from source and target triangles
        src_tri = []
        dest_tri = []
        
        for j in range(0, 3):
            src_tri.append(hull_source[triangle[j]])
            dest_tri.append(hull_target[triangle[j]])
        
        # Warp the triangle
        warp_triangle(source_img, img_source_warped, src_tri, dest_tri)
    
    # Blend the warped face into the target image
    r = cv2.boundingRect(np.float32([hull_target]))
    center = (r[0] + int(r[2]/2), r[1] + int(r[3]/2))
    output = cv2.seamlessClone(
        img_source_warped,
        target_img,
        mask,
        center,
        cv2.NORMAL_CLONE
    )
    
    # Save the result
    cv2.imwrite(output_path, output)
    print(f"Face swap result saved to {output_path}")

def warp_triangle(src_img, dest_img, src_tri, dest_tri):
    # Calculate bounding rectangles for each triangle
    src_rect = cv2.boundingRect(np.float32([src_tri]))
    dest_rect = cv2.boundingRect(np.float32([dest_tri]))
    
    # Offset points by the bounding rectangle
    src_cropped = []
    dest_cropped = []
    
    for i in range(0, 3):
        src_cropped.append(((src_tri[i][0] - src_rect[0]), (src_tri[i][1] - src_rect[1])))
        dest_cropped.append(((dest_tri[i][0] - dest_rect[0]), (dest_tri[i][1] - dest_rect[1])))
    
    # Crop the source image
    src_cropped_img = src_img[src_rect[1]:src_rect[1] + src_rect[3], 
                            src_rect[0]:src_rect[0] + src_rect[2]]
    
    # Calculate the affine transform matrix
    warp_mat = cv2.getAffineTransform(
        np.float32(src_cropped),
        np.float32(dest_cropped)
    )
    
    # Apply the transformation
    warped = cv2.warpAffine(
        src_cropped_img,
        warp_mat,
        (dest_rect[2], dest_rect[3]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    
    # Create a mask for the triangle
    mask = np.zeros((dest_rect[3], dest_rect[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(dest_cropped), (1.0, 1.0, 1.0), 16, 0)
    
    # Copy the triangle to the destination image
    dest_img[dest_rect[1]:dest_rect[1]+dest_rect[3], 
             dest_rect[0]:dest_rect[0]+dest_rect[2]] = (
        dest_img[dest_rect[1]:dest_rect[1]+dest_rect[3], 
                 dest_rect[0]:dest_rect[0]+dest_rect[2]] * (1 - mask) + warped * mask
    )

if __name__ == "__main__":
    # Load source and target images
    source = cv2.imread("source.jpg")
    target = cv2.imread("target.jpg")
    
    if source is None or target is None:
        print("Could not load source or target image!")
        exit()
    
    # Perform face swap
    face_swap(source, target, "output.jpg")
    print("Face swap completed!")