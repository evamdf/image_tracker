import cv2
import os

class ImageLoader:
    def __init__(self, image_folder='data', expected_frame_count=70):
        self.image_folder = image_folder
        self.expected_frame_count = expected_frame_count
    
    def load_frames(self):
        """
        Load sequential frame images from the specified folder.
        Returns: list of RGB images as numpy arrays
        Raises: ValueError if no images found or incorrect number of images
        """
        frames = []
        frame_idx = 0
        
        while True:
            frame_path = os.path.join(self.image_folder, f"frame_{frame_idx}.jpg")
            img = cv2.imread(frame_path, 1)
            
            # If image doesn't exist, stop loading
            if img is None:
                break
                
            # Convert from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Add the image to frames
            frames.append(img)
            frame_idx += 1
        
        # Validation checks
        if len(frames) == 0:
            raise ValueError(f"Error: No images loaded. Check file paths in '{self.image_folder}' folder.")
        
        if len(frames) != self.expected_frame_count:
            raise ValueError(
                f"Error: Expected {self.expected_frame_count} images, but loaded {len(frames)}. "
                f"Check file paths in '{self.image_folder}' folder."
            )
        
        print(f"Successfully loaded {len(frames)} frames from '{self.image_folder}' folder.")
        return frames