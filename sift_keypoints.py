import cv2
import numpy as np
import matplotlib.pyplot as plt

class SIFTKeypoints:
    def __init__(self, images, indices):
        self.frames = images
        self.indices = indices
        self.gray_frames = None
        self.keypoints = None
        self.descriptors = None
        self.matches = None

    def convert_to_grayscale(self):
        """Convert RGB frames to grayscale."""
        if self.frames is None:
            raise ValueError("No frames loaded. Call load_frames() first.")
        
        gray_frames = []
        for img in self.frames:
            if img is None:
                break
            # Convert to grayscale for SIFT processing
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray_frames.append(img_gray)
        
        print(f"Converted {len(gray_frames)} frames to grayscale.")
        self.gray_frames = gray_frames
        return gray_frames
    

    def compute_sift(self):
        """Compute SIFT keypoints and descriptors for all frames."""
        if self.gray_frames is None:
            raise ValueError("No grayscale frames available. Call convert_to_grayscale() first.")
        
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        keypoints = []
        descriptors = []
        
        for i, img in enumerate(self.gray_frames):
            if img is None:
                break
            
            # Detect keypoints and compute descriptors
            kp, des = sift.detectAndCompute(img, None)
            keypoints.append(kp)
            descriptors.append(des)
            
            if (i + 1) % 10 == 0:  # Progress indicator
                print(f"Processed SIFT for {i + 1}/{len(self.gray_frames)} frames")
        
        print(f"SIFT computation complete. Found keypoints in {len(keypoints)} frames.")
        self.keypoints = keypoints
        self.descriptors = descriptors
        return keypoints, descriptors
    
    def match_keypoints(self, step=1):
        """
        Match keypoints between frames using SIFT descriptors.            
        Returns: list of tuples (frame_index, inlier_matches)
        """

        if self.keypoints is None or self.descriptors is None:
            raise ValueError("Keypoints and descriptors must be computed first.")
        

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) # Create brute force matching object

        results = []

        for i in self.indices:
            if i + step < len(self.keypoints):
                keypoint1 = self.keypoints[i]
                keypoint2 = self.keypoints[i + step]
                descriptor1 = self.descriptors[i]
                descriptor2 = self.descriptors[i + step]

                if descriptor1 is None or descriptor2 is None:
                    print(f"Warning: No descriptors found for frames {i} or {i + step}")
                    continue


                matches = bf.match(descriptor1, descriptor2)


                # Sort good matches by distance (lower is better)
                good_matches = sorted(matches, key=lambda x: x.distance)

                # Get the points in first frame, and points in second frame for each match
                src_pts = np.float32([keypoint1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoint2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Apply RANSAC to find inliers 
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 50.0)
                matchesMask = mask.ravel().tolist()
                inliers = [m for i, m in enumerate(good_matches) if matchesMask[i] == 1]

                results.append((i, inliers))
                self.matches = results

                print(f"Keypoint matching complete. Processed {len(results)} frame pairs.")

        return results
    

    def sift_visualization(self, step=1):
        """
        Draws the keypoints on the corresponding frame. 
        Returns a list of figures. 
        """
        figures = []

        rgb_frames = self.frames
        matches = self.matches
        keypoints = self.keypoints

        matches

        for index, matches in matches:

            sift_image = rgb_frames[index].copy()

            keypoint1 = keypoints[index]
            keypoint2 = keypoints[index+step]

            # Colours for convenience
            color_point1 = (0, 255, 0)  
            color_point2 = (0, 0, 255)  
            color_line = (255, 0, 0)     

            for match in matches:
                # Get the keypoints from the first frame
                x1, y1 = keypoint1[match.queryIdx].pt
                x1, y1 = int(x1), int(y1)
                
                # Get the keypoints from the second frame
                x2, y2 = keypoint2[match.trainIdx].pt
                x2, y2 = int(x2), int(y2)

                # A line that shows the keypoint movement between frames
                cv2.line(sift_image, (x1, y1), (x2, y2), color_line, 2)
                
                # Circle for keypoint in first frame
                cv2.circle(sift_image, (x1, y1), 3, color_point1, -1)

                # Circle for the position of keypoint in the second frame
                cv2.circle(sift_image, (x2, y2), 3, color_point2, -1)

            fig = plt.figure(figsize=(12, 8))
            plt.imshow(sift_image)
            plt.axis('off')
            plt.title(f"SIFT Keypoint Movement: frame_{index}.jpg to frame_{index+step}.jpg")

            figures.append(fig)

        return figures
    
    
