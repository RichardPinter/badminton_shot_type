"""The purpose of this file is to separate out variables currently being implimented 
with globals into a central place for easier maintenance and editing"""
import torch
import os
from collections import deque, defaultdict

class PlayerIdentifiers:
    PLAYER_IDENTIFIER_A = 'Player_A'
    PLAYER_IDENTIFIER_B = 'Player_B'


class ModelConfiguration:
    POSE_ESTIMATION_MODEL_PATH = 'yolov8x-pose-p6.pt' #replaces POSE_ESTIMATION_MODEL_PATH
    PERSON_DETECTION_CONFIDENCE_THRESHOLD = 0.4 #replace PERSON_DETECTION_CONDFIDENCE_THRESHOLD
    KEYPOINT_DETECTION_CONFIDENCE_THRESHOLD = 0.3 #replaces KEYPOINT_DETECTION_CONFIDENCE_THRESHOLD
    TRACKER_CONFIGURATION_PATH = 'bytetrack.yaml' #input to pose_estimation_model.track
    USE_CUDA_WHEN_AVAILABLE = True
    COMPUTATION_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class VideoConfiguration:
    """These are the values used for image processing. Output files are always based on input filenames"""
    OUTPUT_DIRECTORY = os.path.join(os.getcwd(), "outputs")
    DISPLAY_SIZE = (1280, 720)
    DEFAULT_FRAMES_PER_SECOND = 30

    def __init__(self, input_path):
        #paths for the output
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        videos_dir = os.path.join(self.OUTPUT_DIRECTORY, "outputs", "videos")
        json_dir   = os.path.join(self.OUTPUT_DIRECTORY, "outputs", "json")
        metrics_dir= os.path.join(self.OUTPUT_DIRECTORY, "outputs", "metrics_csv")
        pos_dir    = os.path.join(self.OUTPUT_DIRECTORY, "outputs", "position_csv")

        #create the directories if they don't exist
        for d in (videos_dir, json_dir, metrics_dir, pos_dir):
            os.makedirs(d, exist_ok=True)

        self.input_video_file_path = input_path
        self.analysis_output_video_path = os.path.join(videos_dir,  f"{base_name}_output_analysis.mp4")
        self.performance_data_json_path = os.path.join(json_dir,   f"{base_name}_performance.json")
        self.detailed_metrics_csv_path  = os.path.join(metrics_dir,f"{base_name}_metrics.csv")
        self.player_positions_csv_path  = os.path.join(pos_dir,    f"{base_name}_positions.csv")
        
        self.original_video_width = 0
        self.original_video_height = 0
        self.start_frame = 0
        self.end_frame = 0
        self.total_video_frame_count = 0
        

class CourtConfiguration:
    NORMALIZE_TO_UNIT_RANGE = True
    FRAME_FOR_SELECTION = 30 #30 is 1 second in. Changed from previous timestamp based method to frame based method.
    COURT_ZONE_GRID_SIZE = 9 #Replaces COURT_ZONE_GRID_SIZE. Unsure what this is for as it's declared by never used.       

    def __init__(self):
        self.corner_points = []
        

class TrackingConfiguration:
    #Variables related to reaquiring lost players
    TRACKING_LOSS_TIMEOUT_RATIO = 0.8
    MAXIMUM_REASSIGNMENT_DISTANCE_PIXELS = 250
    POSITION_DISTANCE_WEIGHT_FOR_REACQUISITION = 0.7
    POSITION_DISTANCE_WEIGHT_FOR_NEW_ASSIGNMENT = 0.4

    #Variables related to tracking jersey color
    JERSEY_REGION_TOP_OFFSET_RATIO = 0.15
    JERSEY_REGION_BOTTOM_OFFSET_RATIO = 0.55
    JERSEY_REGION_HORIZONTAL_PADDING_RATIO = 0.20
    MINIMUM_JERSEY_REGION_AREA_PIXELS = 50
    JERSEY_COLOR_SIMILARITY_THRESHOLD_BGR = 60.0       
    JERSEY_COLOR_WEIGHT_FOR_REACQUISITION = 0.3
    JERSEY_COLOR_WEIGHT_FOR_NEW_ASSIGNMENT = 0.6
    
    def __init__(self, ident_A, ident_B, frame_rate):
        #Variables for player detection and tracking
        #Dictionaries store data on players tracked between frames
        self.max_frames_before_lost = int(self.TRACKING_LOSS_TIMEOUT_RATIO * frame_rate)
        self.slot_mapping = {ident_A: None, ident_B: None}
        self.last_known_court_positions = {ident_A: None, ident_B: None} #dict stores tuple of (numpy_coords, frame_number)
        self.last_detection_frame_numbers = {ident_A: -1, ident_B: -1}
        self.frame_by_frame_data = {ident_A: [], ident_B: []}
        self.frames_inside_court_count = {ident_A: 0, ident_B: 0}
        self.dominant_jersey_colors = {ident_A: None, ident_B: None}

        #Performance stats will need rework along with shot detection. 
        self.performance_statistics = {ident_A: {'court_zone_occupancy_counts': defaultdict(int),
                                                'stroke_type_counts': defaultdict(int),
                                                'rally_statistics': [],
                                                'shot_timestamps': [],
                                                'rest_period_durations': [],
                                                'shot_zones_distribution': defaultdict(list)}, 
                                        ident_B: {'court_zone_occupancy_counts': defaultdict(int),
                                                'stroke_type_counts': defaultdict(int),
                                                'rally_statistics': [],
                                                'shot_timestamps': [],
                                                'rest_period_durations': [],
                                                'shot_zones_distribution': defaultdict(list)}}

        
    
    def calculate_position_distance_score(self, distance):
        #Score based on distance score in existing code
        return (self.MAXIMUM_REASSIGNMENT_DISTANCE_PIXELS - distance)/self.MAXIMUM_REASSIGNMENT_DISTANCE_PIXELS

    def calculate_color_score(self, distance):
        if distance < self.JERSEY_COLOR_SIMILARITY_THRESHOLD_BGR:
            return (self.JERSEY_COLOR_SIMILARITY_THRESHOLD_BGR - distance)/self.JERSEY_COLOR_SIMILARITY_THRESHOLD_BGR
        else:
            return 0.0

class PoseDefinitions:
    """Constants related to skeleton keypoints and angle calculations"""
    # COCO 17 keypoint names in order (used by YOLOv8 pose)
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    # Dictionary mapping keypoint names to their index
    KP_N = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

    # Definitions of joint angles to calculate from keypoints
    ANGLE_DEFINITIONS = {
        "elbow_angle_L": (KP_N["left_shoulder"], KP_N["left_elbow"], KP_N["left_wrist"]),
        "elbow_angle_R": (KP_N["right_shoulder"], KP_N["right_elbow"], KP_N["right_wrist"]),
        "knee_angle_L": (KP_N["left_hip"], KP_N["left_knee"], KP_N["left_ankle"]),
        "knee_angle_R": (KP_N["right_hip"], KP_N["right_knee"], KP_N["right_ankle"]),
        "shoulder_angle_L": (KP_N["left_elbow"], KP_N["left_shoulder"], KP_N["left_hip"]),
        "shoulder_angle_R": (KP_N["right_elbow"], KP_N["right_shoulder"], KP_N["right_hip"]),
        "hip_angle_L": (KP_N["left_shoulder"], KP_N["left_hip"], KP_N["left_knee"]),
        "hip_angle_R": (KP_N["right_shoulder"], KP_N["right_hip"], KP_N["right_knee"])
    }

    # Definitions of relative distances between joints to calculate
    RELATIVE_DISTANCE_DEFINITIONS = {
        "norm_elbow_to_wrist_L": (KP_N["left_elbow"], KP_N["left_wrist"]),
        "norm_elbow_to_wrist_R": (KP_N["right_elbow"], KP_N["right_wrist"]),
        "norm_shoulder_to_hip_L": (KP_N["left_shoulder"], KP_N["left_hip"]),
        "norm_shoulder_to_hip_R": (KP_N["right_shoulder"], KP_N["right_hip"]),
        "norm_knee_to_ankle_L": (KP_N["left_knee"], KP_N["left_ankle"]),
        "norm_knee_to_ankle_R": (KP_N["right_knee"], KP_N["right_ankle"]),
    }

    # Keypoints for which to calculate velocity and acceleration
    TARGET_VEL_ACC_KEYPOINTS = {
        "wrist_L": KP_N["left_wrist"], "wrist_R": KP_N["right_wrist"],
        "elbow_L": KP_N["left_elbow"], "elbow_R": KP_N["right_elbow"],
        "ankle_L": KP_N["left_ankle"], "ankle_R": KP_N["right_ankle"],
        "knee_L": KP_N["left_knee"], "knee_R": KP_N["right_knee"],
    }

    # Connections between keypoints for skeleton visualization
    SKELETON_CONNECTIONS = [
        (KP_N["left_shoulder"], KP_N["right_shoulder"]), (KP_N["left_hip"], KP_N["right_hip"]),
        (KP_N["left_shoulder"], KP_N["left_hip"]), (KP_N["right_shoulder"], KP_N["right_hip"]),
        (KP_N["left_shoulder"], KP_N["left_elbow"]), (KP_N["left_elbow"], KP_N["left_wrist"]),
        (KP_N["right_shoulder"], KP_N["right_elbow"]), (KP_N["right_elbow"], KP_N["right_wrist"]),
        (KP_N["left_hip"], KP_N["left_knee"]), (KP_N["left_knee"], KP_N["left_ankle"]),
        (KP_N["right_hip"], KP_N["right_knee"]), (KP_N["right_knee"], KP_N["right_ankle"]),
        (KP_N["nose"], KP_N["left_eye"]), (KP_N["nose"], KP_N["right_eye"]),
        (KP_N["left_eye"], KP_N["left_ear"]), (KP_N["right_eye"], KP_N["right_ear"])
    ]

    MOVEMENT_PATTERN_JOINTS = ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]



class TemporalFeatureBuffer:
    # Window size for temporal feature calculation (number of frames)
    DEFAULT_TEMPORAL_WINDOW_SIZE = 15  
    # Stride for sliding window operations
    DEFAULT_TEMPORAL_STRIDE = 1
    # Number of Dynamic Time Warping features extracted
    NUM_DTW_FEATURES = 14
    # Scale factors applied to normalize measurements to appropriate ranges
    VELOCITY_SCALE_FACTOR = 0.01  # Scale down velocities by 100x
    ACCELERATION_SCALE_FACTOR = 0.01  # Scale down accelerations by 100x


    """
    Temporal feature buffers are used to calculate stats for the past DEFAULT_TEMPORAL_WINDOW_SIZE frames
    the states are saved in the player tracking data object
    """
    def __init__(self, ident_A, ident_B):
        self.player_temporal_feature_buffers = {
            ident_A: {
                'pose_keypoints_buffer': deque(maxlen=self.DEFAULT_TEMPORAL_WINDOW_SIZE),
                'timestamp_buffer': deque(maxlen=self.DEFAULT_TEMPORAL_WINDOW_SIZE),
                'body_center_position_buffer': deque(maxlen=self.DEFAULT_TEMPORAL_WINDOW_SIZE),
                'spatial_feature_buffer': deque(maxlen=self.DEFAULT_TEMPORAL_WINDOW_SIZE),
                'processed_frame_count': 0
            },
            ident_B: {
                'pose_keypoints_buffer': deque(maxlen=self.DEFAULT_TEMPORAL_WINDOW_SIZE),
                'timestamp_buffer': deque(maxlen=self.DEFAULT_TEMPORAL_WINDOW_SIZE),
                'body_center_position_buffer': deque(maxlen=self.DEFAULT_TEMPORAL_WINDOW_SIZE),
                'spatial_feature_buffer': deque(maxlen=self.DEFAULT_TEMPORAL_WINDOW_SIZE),
                'processed_frame_count': 0
            }
}