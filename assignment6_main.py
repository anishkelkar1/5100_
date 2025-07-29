"""
Vehicle Detection System using OpenCV
This system detects vehicles in video streams using multiple approaches:
1. Haar Cascade Classifier (Classical ML approach)
2. Background Subtraction with Contour Detection
3. HOG + SVM (Optional advanced method)
"""

import cv2
import numpy as np
import os
from datetime import datetime

class VehicleDetectionSystem:
    def __init__(self):
        """Initialize the vehicle detection system with multiple detection methods"""
        
        # Load Haar Cascade classifier for cars
        # Download from: https://github.com/opencv/opencv/tree/master/data/haarcascades
        self.car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')
        
        # Initialize background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Parameters for detection
        self.min_contour_area = 500  # Minimum area to consider as vehicle
        self.max_contour_area = 50000  # Maximum area to filter out large artifacts
        
        # Tracking parameters
        self.vehicle_count = 0
        self.tracked_vehicles = {}
        self.vehicle_id_counter = 0
        
        # Detection zones (for counting vehicles crossing a line)
        self.detection_line_y = None
        self.crossed_vehicles = set()
        
    def detect_vehicles_cascade(self, frame):
        """
        Detect vehicles using Haar Cascade classifier
        
        Args:
            frame: Input frame from video
            
        Returns:
            List of bounding boxes for detected vehicles
        """
        # Convert to grayscale for cascade detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect cars using cascade classifier
        # Parameters: scaleFactor, minNeighbors, minSize
        vehicles = self.car_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(50, 50)
        )
        
        return vehicles
    
    def detect_vehicles_motion(self, frame):
        """
        Detect moving vehicles using background subtraction
        
        Args:
            frame: Input frame from video
            
        Returns:
            List of bounding boxes for detected moving objects
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows (shadows are gray in the mask)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Dilate to fill gaps
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on area and aspect ratio
        vehicles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_contour_area < area < self.max_contour_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (vehicles typically have width > height)
                aspect_ratio = float(w) / h
                if 0.5 < aspect_ratio < 4.0:
                    vehicles.append((x, y, w, h))
        
        return vehicles, fg_mask
    
    def track_vehicles(self, vehicles, frame_height):
        """
        Simple vehicle tracking and counting
        
        Args:
            vehicles: List of detected vehicle bounding boxes
            frame_height: Height of the video frame
        """
        # Set detection line at 2/3 height of frame if not set
        if self.detection_line_y is None:
            self.detection_line_y = int(frame_height * 2 / 3)
        
        current_centroids = []
        
        # Calculate centroids of detected vehicles
        for (x, y, w, h) in vehicles:
            cx = x + w // 2
            cy = y + h // 2
            current_centroids.append((cx, cy))
            
            # Check if vehicle crosses the detection line
            if cy > self.detection_line_y - 10 and cy < self.detection_line_y + 10:
                # Simple tracking: check if this is a new vehicle
                is_new = True
                for tracked_id, (tx, ty) in self.tracked_vehicles.items():
                    if abs(cx - tx) < 50 and abs(cy - ty) < 50:
                        is_new = False
                        break
                
                if is_new and (cx, cy) not in self.crossed_vehicles:
                    self.vehicle_count += 1
                    self.crossed_vehicles.add((cx, cy))
        
        # Update tracked vehicles
        self.tracked_vehicles = {i: cent for i, cent in enumerate(current_centroids)}
    
    def process_video(self, video_path, output_path=None, show_video=True):
        """
        Main function to process video and detect vehicles
        
        Args:
            video_path: Path to input video file
            output_path: Path to save output video (optional)
            show_video: Whether to display the video while processing
        """
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer if output path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process video frame by frame
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Method 1: Cascade detection
            cascade_vehicles = self.detect_vehicles_cascade(frame)
            
            # Method 2: Motion-based detection
            motion_vehicles, fg_mask = self.detect_vehicles_motion(frame)
            
            # Combine detections (remove duplicates)
            all_vehicles = self._combine_detections(cascade_vehicles, motion_vehicles)
            
            # Track vehicles and count
            self.track_vehicles(all_vehicles, height)
            
            # Draw results on frame
            display_frame = self._draw_detections(frame.copy(), all_vehicles)
            
            # Draw detection line
            cv2.line(display_frame, (0, self.detection_line_y), 
                    (width, self.detection_line_y), (0, 255, 0), 2)
            
            # Add text information
            self._add_info_text(display_frame, frame_count, fps)
            
            # Show additional windows for debugging
            if show_video:
                cv2.imshow('Vehicle Detection', display_frame)
                cv2.imshow('Motion Mask', fg_mask)
                
                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Write frame to output video
            if output_path:
                out.write(display_frame)
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"Processing complete. Total vehicles counted: {self.vehicle_count}")
        
    def _combine_detections(self, cascade_vehicles, motion_vehicles):
        """
        Combine detections from multiple methods and remove duplicates
        
        Args:
            cascade_vehicles: Vehicles detected by cascade classifier
            motion_vehicles: Vehicles detected by motion detection
            
        Returns:
            Combined list of unique vehicle detections
        """
        all_vehicles = list(cascade_vehicles)
        
        # Add motion vehicles if they don't overlap with cascade detections
        for (mx, my, mw, mh) in motion_vehicles:
            is_duplicate = False
            
            for (cx, cy, cw, ch) in cascade_vehicles:
                # Check for overlap using IoU (Intersection over Union)
                if self._calculate_iou((mx, my, mw, mh), (cx, cy, cw, ch)) > 0.5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                all_vehicles.append((mx, my, mw, mh))
        
        return all_vehicles
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection area
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union area
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        if union_area == 0:
            return 0
        
        return intersection_area / union_area
    
    def _draw_detections(self, frame, vehicles):
        """Draw bounding boxes and labels on frame"""
        for i, (x, y, w, h) in enumerate(vehicles):
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add label
            label = f"Vehicle {i+1}"
            cv2.putText(frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw centroid
            cx = x + w // 2
            cy = y + h // 2
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        
        return frame
    
    def _add_info_text(self, frame, frame_count, fps):
        """Add information text to frame"""
        # Add vehicle count
        text = f"Vehicles Counted: {self.vehicle_count}"
        cv2.putText(frame, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Add frame info
        text = f"Frame: {frame_count} | FPS: {fps}"
        cv2.putText(frame, text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add detection method info
        text = "Methods: Cascade + Motion Detection"
        cv2.putText(frame, text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def main():
    """
    Main function to demonstrate the vehicle detection system
    """
    # Create vehicle detection system
    detector = VehicleDetectionSystem()
    
    # Example usage with different video sources
    print("Vehicle Detection System")
    print("-" * 50)
    
    # Option 1: Use webcam (0 for default camera)
    # video_source = 0
    
    # Option 2: Use video file
    # Download sample traffic videos from:
    # - https://www.pexels.com/search/videos/traffic/
    # - https://github.com/intel-iot-devkit/sample-videos
    # - https://pixabay.com/videos/search/traffic/
    # - https://www.videvo.net/stock-video-footage/traffic/
    video_source = "traffic_video.mp4"  # Replace with your video path
    
    # Option 3: Use IP camera stream
    # video_source = "http://your-ip-camera-url/video"
    
    # Process video
    output_path = "vehicle_detection_output.avi"
    
    try:
        detector.process_video(video_source, output_path, show_video=True)
    except Exception as e:
        print(f"Error processing video: {e}")
        print("Please ensure you have:")
        print("1. A valid video file or camera source")
        print("2. OpenCV installed with video codecs")
        print("3. Haar cascade XML file in the correct location")


if __name__ == "__main__":
    main()
