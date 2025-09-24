import cv2
import numpy as np
import matplotlib.pyplot as plt

class SIFTTrack:
    def __init__(self, images, keypoints, descriptors, indices):
        
        # Initial bounding box x and y coordinates, and width and height
        x, y, w, h = [20, 353, 322, 215]

        x2 = x + w
        y2 = y + h

        self.initial_bbox = [x, y, x2, y2]

        self.keypoints = keypoints
        self.frames = images
        self.descriptors = descriptors
        self.indices = indices

        self.box_keypoints = None
        self.box_descriptors = None
        self.bboxes = None


    def keypoints_in_box(self):
        """
        Finds the keypoints that sit inside the original bounding box (and their descriptors). 
        """

        x, y, x2, y2 = self.initial_bbox

        # Keypoints and descriptors in every frame
        all_keypoints = self.keypoints 
        all_descriptors = self.descriptors

        # Keypoints and descriptors in just the first frame 
        keypoints = all_keypoints[0]
        descriptors = all_descriptors[0]

        box_keypoints = []
        box_descriptors = []

        for i, kp in enumerate(keypoints):
                    kp_x, kp_y = kp.pt
                    if (x <= kp_x <= x2) and (y <= kp_y <= y2):
                        box_keypoints.append(kp)
                    
                        box_descriptors.append(descriptors[i])

        # Keypoints and descriptors inside the initial bounding box in the first frame 
        self.box_keypoints = box_keypoints
        self.box_descriptors = box_descriptors

        print(f"Found keypoints inside bounding box.")

        return box_keypoints, box_descriptors


    def track_sift(self):
        """
        Tracks the keypoints from one frame to the next by matching up keypoints in consecutive frames.
        Based on the median movement of the keypoints, adjusts the bounding box to follow. 
        """

        x, y, x2, y2 = self.initial_bbox
        keypoints = self.keypoints
        descriptors = self.descriptors 

        bboxes = [[x, y, x2, y2]]
     

        # Get the keypoints and descriptors for the first frame inside the bounding box
        box_keypoints = self.box_keypoints
        box_descriptors = self.box_descriptors

        # Convert to numpy array for faster processing
        if box_descriptors:
            box_descriptors = np.array(box_descriptors)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) # Create brute force matcher

        for frame_idx in range (1, len(keypoints)):

            current_keypoints = keypoints[frame_idx]
            current_descriptors = descriptors[frame_idx]

            matches = bf.match(box_descriptors, current_descriptors)

            good_matches = sorted(matches, key=lambda x: x.distance)

            # Second filter: remove matches where keypoints are too far apart
            distance_filtered_matches = []
            
            for m in good_matches:  
                # Get the coordinates of the matched keypoints
                src_pt = box_keypoints[m.queryIdx].pt
                dst_pt = current_keypoints[m.trainIdx].pt
                
                # Calculate spatial distance between matched points
                spatial_distance = np.sqrt((src_pt[0] - dst_pt[0])**2 + (src_pt[1] - dst_pt[1])**2)
                
                # Only keep matches where keypoints are within the maximum distance
                if spatial_distance <= 300:
                    distance_filtered_matches.append(m)

            if len(distance_filtered_matches) == 0:
                # If no keypoints in this frame, use the previous bounding box
                distance_filtered_matches = good_matches
        
            # Extract matched keypoint coordinates
            src_pts = np.float32([box_keypoints[m.queryIdx].pt for m in distance_filtered_matches[:100]]).reshape(-1, 1, 2)
            dst_pts = np.float32([current_keypoints[m.trainIdx].pt for m in distance_filtered_matches[:100]]).reshape(-1, 1, 2)

            # Calculate the median movement 
            movement = dst_pts - src_pts
            median_movement = np.median(movement.reshape(-1, 2), axis=0)

            # Update bounding box based on median movement
            prev_x, prev_y, prev_x2, prev_y2 = bboxes[-1]
            new_x = int(prev_x + median_movement[0])
            new_y = int(prev_y + median_movement[1])
            new_x2 = int(prev_x2 + median_movement[0])
            new_y2 = int(prev_y2 + median_movement[1])


            # Update the keypoints in the box to be the ones in the next frame
            box_keypoints = []
            box_descriptors = []
            
            for m in distance_filtered_matches:
                box_keypoints.append(current_keypoints[m.trainIdx])
                box_descriptors.append(current_descriptors[m.trainIdx])


            # Add the new bounding box
            bboxes.append((new_x, new_y, new_x2, new_y2))

        self.bboxes = bboxes

        print(f"Found bounding boxes in each frame.")

        # Return the list of all bounding boxes 
        return bboxes
    


    def visualize_bboxes(self):
        """
        Draws on the corresponding bounding box for each frame. 
        Returns: list of figures. 
        """

        bboxes = self.bboxes
        indices = self.indices
        rgb_frames = self.frames

        figures = []

        for index in indices:
            bbox_img = rgb_frames[index].copy()

            bbox = bboxes[index]

            x, y, x2, y2 = bbox

            # Draw the bounding box 
            cv2.rectangle(bbox_img, (x, y), (x2, y2),(25,255,55),3)
            
            fig = plt.figure(figsize=(12, 8))
            plt.imshow(bbox_img)
            plt.axis('off')
            plt.title(f'SIFT Tracking Bounding Boxes: frame_{index}.jpg')
            
            figures.append(fig)

        return figures
