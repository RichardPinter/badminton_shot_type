import config as cfg
import cv2
import numpy as np
import pandas as pd
from feature_extraction import *

#Progress bar code pinched from https://stackoverflow.com/questions/6169217/replace-console-output-in-python to make output look a little better
def progress_bar(current, total, bar_length=20):
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n' if current == total else '\r'

    print(f'Progress: [{arrow}{padding}] {int(fraction*100)}%', end=ending)

# --- Jersey Color Helper Functions ---
def extract_jersey_region_from_bounding_box(video_frame, player_bounding_box):
    """
    Extract jersey region of interest from player bounding box.
    
    This function calculates a region of interest (ROI) that likely contains
    the player's jersey/shirt, focusing on the upper body area while excluding
    the head and lower body.
    
    Args:
        video_frame (np.ndarray): The video frame containing the player
        player_bounding_box (tuple): Bounding box coordinates (x1, y1, x2, y2)
        
    Returns:
        np.ndarray or None: ROI image containing the jersey area, or None if
                           the ROI is invalid or too small
    
    Notes:
        Uses global constants for ROI position calculation:
        - JERSEY_REGION_TOP_OFFSET_RATIO: Where to start ROI vertically (% of height from top)
        - JERSEY_REGION_BOTTOM_OFFSET_RATIO: Where to end ROI vertically (% of height from top)
        - JERSEY_REGION_HORIZONTAL_PADDING_RATIO: Horizontal padding from edges (% of width)
        - MINIMUM_JERSEY_REGION_AREA_PIXELS: Minimum area required for valid ROI
    """
    bbox_left, bbox_top, bbox_right, bbox_bottom = map(int, player_bounding_box)
    bounding_box_height = bbox_bottom - bbox_top
    bounding_box_width = bbox_right - bbox_left
    
    # Calculate ROI boundaries based on constants
    jersey_region_top = bbox_top + int(bounding_box_height * cfg.TrackingConfiguration.JERSEY_REGION_TOP_OFFSET_RATIO)
    jersey_region_bottom = bbox_top + int(bounding_box_height * cfg.TrackingConfiguration.JERSEY_REGION_BOTTOM_OFFSET_RATIO)
    jersey_region_left = bbox_left + int(bounding_box_width *cfg.TrackingConfiguration.JERSEY_REGION_HORIZONTAL_PADDING_RATIO)
    jersey_region_right = bbox_right - int(bounding_box_width * cfg.TrackingConfiguration.JERSEY_REGION_HORIZONTAL_PADDING_RATIO)
    
    # Ensure ROI is within frame boundaries
    jersey_region_top = max(0, jersey_region_top)
    jersey_region_bottom = min(video_frame.shape[0] - 1, jersey_region_bottom)
    jersey_region_left = max(0, jersey_region_left)
    jersey_region_right = min(video_frame.shape[1] - 1, jersey_region_right)
    
    # Check if ROI is valid
    jersey_region_area = (jersey_region_bottom - jersey_region_top) * (jersey_region_right - jersey_region_left)
    if jersey_region_bottom <= jersey_region_top or jersey_region_right <= jersey_region_left or jersey_region_area < cfg.TrackingConfiguration.MINIMUM_JERSEY_REGION_AREA_PIXELS:
        return None
        
    return video_frame[jersey_region_top:jersey_region_bottom, jersey_region_left:jersey_region_right]

def calculate_dominant_color_in_region(image_region, number_of_color_clusters=1):
    """
    Determine the dominant color in an image ROI using k-means clustering.
    
    Args:
        image_region (np.ndarray): Region of interest from which to extract dominant color
        number_of_color_clusters (int): Number of color clusters to use in k-means (default: 1)
        
    Returns:
        np.ndarray or None: BGR color array of dominant color, or None if
                           the ROI is invalid or clustering fails
    
    Notes:
        - Uses k-means clustering to find the most common color
        - Falls back to average color calculation if k-means fails
        - Returns None if the input is invalid
    """
    if image_region is None or image_region.size == 0:
        return None
        
    # Reshape image for k-means
    flattened_pixels = image_region.reshape((-1, 3))
    flattened_pixels = np.float32(flattened_pixels)
    kmeans_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    try:
        # Try k-means clustering
        _, cluster_labels, cluster_centers = cv2.kmeans(flattened_pixels, number_of_color_clusters, None, 
                                                        kmeans_criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    except cv2.error as kmeans_error:
        # Fallback: calculate average color if Kmeans fails
        if flattened_pixels.shape[0] > 0:
            average_color = np.mean(flattened_pixels, axis=0)
            return np.uint8(average_color)
        return None

    # Find the most common cluster
    cluster_centers = np.uint8(cluster_centers)
    cluster_pixel_counts = np.bincount(cluster_labels.flatten())
    dominant_cluster_index = np.argmax(cluster_pixel_counts)
    return cluster_centers[dominant_cluster_index]

def calculate_color_euclidean_distance(first_bgr_color, second_bgr_color):
    """
    Calculate the Euclidean distance between two BGR colors.
    
    Args:
        first_bgr_color (np.ndarray): First BGR color array
        second_bgr_color (np.ndarray): Second BGR color array
        
    Returns:
        float: Euclidean distance between colors, or infinity if either color is None
    
    Notes:
        - Lower values indicate more similar colors
        - Returns infinity if either color is None
    """
    if first_bgr_color is None or second_bgr_color is None:
        return float('inf')
    return np.linalg.norm(np.array(first_bgr_color, dtype=int) - np.array(second_bgr_color, dtype=int))

# --- Court Selection Functions ---
def handle_court_corner_selection_click(event, mouse_x, mouse_y, flags, callback_parameters):
    """
    Mouse callback function for selecting court corner points.
    
    Handles mouse click events and stores the selected court corner points,
    adjusting for any scaling between display and original video dimensions.
    
    Args:
        event: OpenCV mouse event type
        mouse_x, mouse_y: Mouse coordinates where the event occurred
        flags: Additional flags passed by OpenCV
        callback_parameters: Additional parameters dictionary containing:
            - frame_to_show: Frame being displayed
            - original_dims: Original video dimensions (width, height)
            - display_dims: Display dimensions (width, height) if resized
            - court_config: CourtConfiguration Object to save the points too
        
        CourtConfiguration object needs to be passed because cv2 MouseClick function doesn't return a value so the list can't be returned normally.
        Instead the points are saved by modifying the CourtConfig object
    
    Side effects:
        - Updates global BADMINTON_COURT_CORNER_POINTS list with selected points
        - Draws circles on the displayed frame to show selected points
    """
    court_points = callback_parameters.get('court_config').corner_points
    if event == cv2.EVENT_LBUTTONDOWN and len(court_points) < 4:
        original_coordinate_x, original_coordinate_y = mouse_x, mouse_y
        
        # Adjust coordinates if display is scaled
        if callback_parameters.get('display_dims') and callback_parameters.get('original_dims'):
            display_width, display_height = callback_parameters['display_dims']
            original_width, original_height = callback_parameters['original_dims']
            original_coordinate_x = int(mouse_x * (original_width / display_width))
            original_coordinate_y = int(mouse_y * (original_height / display_height))
            
        # Store point and visualize
        court_points.append((original_coordinate_x, original_coordinate_y))
        cv2.circle(callback_parameters['frame_to_show'], (mouse_x, mouse_y), 5, (0, 255, 0), -1)
        cv2.imshow("Select Court Corners", callback_parameters['frame_to_show'])

def prompt_user_to_select_court_corners(video_capture_object, frame_for_selection, court_config):
    """
    Prompt user to select four court corners using a frame from the video.
    
    Displays a frame from the video and allows the user to click on the four corners
    of the badminton court. Uses mouse callback to capture the clicks and store
    the coordinates.
    
    Args:
        video_capture_object: OpenCV video capture object
        frame_for_selection (int): Frame to jump to for court selection
        
    Returns:
        list or None: List of four (x,y) corner points, or None if selection was aborted
    
    Side effects:
        - Opens a window for court selection
        - Updates global BADMINTON_COURT_CORNER_POINTS list
    
    Notes:
        - User must press 'c' to confirm selection after picking 4 points
        - User can press 'q' to abort selection
        - Points should be selected in order: top-left, top-right, bottom-right, bottom-left
    """    
    # Seek to specified time and get frame
    video_capture_object.set(cv2.CAP_PROP_POS_FRAMES, frame_for_selection)
    frame_read_success, court_selection_frame = video_capture_object.read()
    if not frame_read_success:
        print(f"Error reading frame for court selection at {frame_for_selection}s.")
        return None
        
    # Store original dimensions
    original_frame_dimensions = (court_selection_frame.shape[1], court_selection_frame.shape[0])
    frame_to_show_selection = court_selection_frame.copy()
    scaled_display_dimensions = None
    
    # Resize for display if needed
    video_display_width, video_display_height = cfg.VideoConfiguration.DISPLAY_SIZE
    if video_display_width and video_display_height:
        frame_to_show_selection = cv2.resize(court_selection_frame.copy(), 
                                                 (video_display_width, video_display_height))
        scaled_display_dimensions = (video_display_width, video_display_height)
        
    # Set up mouse callback
    cv2.namedWindow("Select Court Corners")
    cv2.setMouseCallback("Select Court Corners", handle_court_corner_selection_click,
                         {'frame_to_show': frame_to_show_selection,
                          'original_dims': original_frame_dimensions, 
                          'display_dims': scaled_display_dimensions,
                          'court_config': court_config})
                          
    print(f"Click 4 court points (frame at {frame_for_selection}s). Press 'c' after 4, 'q' to abort.")
    court_config.corner_points = []
    while True:
        cv2.imshow("Select Court Corners", frame_to_show_selection)
        keyboard_input = cv2.waitKey(20) & 0xFF
        if keyboard_input == ord('q'):
            court_config.corner_points = []
            break
        if keyboard_input == ord('c') and len(court_config.corner_points) == 4:
            break
        if keyboard_input == ord('c') and len(court_config.corner_points) < 4:
            print("Select 4 points.")
    cv2.destroyAllWindows()
    video_capture_object.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return np.array(court_config.corner_points, dtype=np.int32) if len(court_config.corner_points) == 4 else None

def check_if_point_inside_court(test_point, court_boundary_polygon):
    return court_boundary_polygon is not None and len(court_boundary_polygon) >= 3 and \
           cv2.pointPolygonTest(court_boundary_polygon, tuple(map(int, test_point)), False) >= 0

def calculate_bounding_box_bottom_center(bounding_box_coordinates):
    left_x, top_y, right_x, bottom_y = bounding_box_coordinates
    return np.array([int((left_x + right_x) / 2), int(bottom_y)])

def determine_player_court_reference_point(player_keypoints, player_bounding_box):
    """More robust reference point for court check & matching"""
    court_position = get_player_position_on_court(player_keypoints)
    if court_position is not None: 
        return court_position
    return calculate_bounding_box_bottom_center(player_bounding_box)

# --- Position CSV Generation Function ---
def export_player_positions_to_csv(player_tracking_data, csv_output_path, video_width, video_height):
    """Generate a CSV file with player position coordinates for each frame."""
    position_records = []
    
    # Get all frame numbers from both players
    all_frame_numbers = set()
    for player_identifier, frame_data_list in player_tracking_data.items():
        for single_frame_data in frame_data_list:
            all_frame_numbers.add(single_frame_data["frame_num"])
    
    # Create a sorted list of frame numbers
    chronological_frame_numbers = sorted(all_frame_numbers)
    
    # For each frame, get the position of each player if available
    for current_frame_number in chronological_frame_numbers:
        frame_position_record = {"frame_num": current_frame_number}
        
        # Get player positions for this frame
        for player_identifier in player_tracking_data.keys():
            # Find the data for this player and frame
            matching_frame_data = next(
                (frame_data for frame_data in player_tracking_data[player_identifier] 
                 if frame_data["frame_num"] == current_frame_number), 
                None
            )
            
            if matching_frame_data and matching_frame_data.get("player_position_on_court"):
                court_position = matching_frame_data["player_position_on_court"]
                
                # Convert from normalized to actual coordinates if needed
                if cfg.CourtConfiguration.NORMALIZE_TO_UNIT_RANGE:
                    actual_x_coordinate = court_position[0] * video_width
                    actual_y_coordinate = court_position[1] * video_height
                else:
                    actual_x_coordinate = court_position[0]
                    actual_y_coordinate = court_position[1]
                
                # Add to frame entry
                frame_position_record[f"{player_identifier}_x"] = actual_x_coordinate
                frame_position_record[f"{player_identifier}_y"] = actual_y_coordinate
            else:
                # No position data for this player in this frame
                frame_position_record[f"{player_identifier}_x"] = None
                frame_position_record[f"{player_identifier}_y"] = None
        
        # Add timestamp if available from either player's data
        for player_identifier in [cfg.PlayerIdentifiers.PLAYER_IDENTIFIER_A, cfg.PlayerIdentifiers.PLAYER_IDENTIFIER_B]:
            matching_frame_data = next(
                (frame_data for frame_data in player_tracking_data[player_identifier] 
                 if frame_data["frame_num"] == current_frame_number), 
                None
            )
            if matching_frame_data and "timestamp" in matching_frame_data:
                frame_position_record["timestamp"] = matching_frame_data["timestamp"]
                break
        
        position_records.append(frame_position_record)
    
    # Create and save DataFrame
    positions_dataframe = pd.DataFrame(position_records)
    positions_dataframe.to_csv(csv_output_path, index=False)
    print(f"Player position coordinates saved to {csv_output_path}")
