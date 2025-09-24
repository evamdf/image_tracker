from src.image_loader import ImageLoader
from src.sift_keypoints import SIFTKeypoints
from src.sift_tracking import SIFTTrack
from src.save_figures import SaveFigures


def main():
    try:
        # Initialize image loader
        loader = ImageLoader(image_folder='images')

        # Indicies of frames to make figures of 
        indices = [x for x in range(0, 70, 20)]
        
        # Load frames
        images = loader.load_frames()
        
        # Initialize SIFT keypoint detector, and convert frames to grayscale
        sift = SIFTKeypoints(images, indices)
        gray_frames = sift.convert_to_grayscale()
        keypoints, descriptors = sift.compute_sift() 
        matches = sift.match_keypoints(step=1) # Match keypoints across frames using descriptors
        sift_figures = sift.sift_visualization() # Create visualizations 

        # Initialize SIFT tracking
        tracker = SIFTTrack(images, keypoints, descriptors, indices) 
        tracker.keypoints_in_box() # Find the keypoints inside the initial bounding box
        tracker.track_sift() # Track these keypoints across all the frames 
        tracking_figures = tracker.visualize_bboxes() # Create visualizations 

        # Save all the figures to new folders 
        saver = SaveFigures(indices)
        saver.save_figures(sift_figures, 'sift_keypoints', 'SIFT_KeyPoints_Results')
        saver.save_figures(tracking_figures, 'sift_tracking', 'SIFT_Tracking_Results')
        
        
    except ValueError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()