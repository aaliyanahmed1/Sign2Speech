#!/usr/bin/env python3
"""
DeepSORT Tracking Module
Handles object tracking to maintain consistent IDs across frames
"""

import numpy as np
from typing import List, Optional
from collections import defaultdict
import time
from .deep_sort.utils.parser import get_config
from .deep_sort.deep_sort import DeepSort
from .deep_sort.sort.tracker import Tracker

# Path to DeepSORT weights
DEEPSORT_WEIGHTS = 'components/deep_sort/deep/checkpoint/ckpt.t7'


class Track:
    """Data class for tracking results"""
    def __init__(self, track_id, bbox, confidence, class_id, class_name):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.center = self._calculate_center()
        self.last_seen = time.time()
    
    def _calculate_center(self):
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def update_position(self, bbox, confidence):
        """Update track position"""
        self.bbox = bbox
        self.confidence = confidence
        self.center = self._calculate_center()
        self.last_seen = time.time()


class SimpleTracker:
    """Simple tracking implementation when DeepSORT is not available"""
    
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_id = 0
        self.tracks = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.disappeared = defaultdict(int)
    
    def update(self, detections, frame=None):
        """Update tracker with new detections"""
        if len(detections) == 0:
            # Mark all existing tracks as disappeared
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self._deregister_track(track_id)
            return list(self.tracks.values())
        
        # If no existing tracks, register all detections as new tracks
        if len(self.tracks) == 0:
            for detection in detections:
                self._register_track(detection)
        else:
            # Match detections to existing tracks
            self._match_detections_to_tracks(detections)
        
        return list(self.tracks.values())
    
    def _register_track(self, detection):
        """Register a new track"""
        track = Track(
            track_id=self.next_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            class_id=detection.class_id,
            class_name=detection.class_name
        )
        self.tracks[self.next_id] = track
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def _deregister_track(self, track_id):
        """Remove a track"""
        if track_id in self.tracks:
            del self.tracks[track_id]
        if track_id in self.disappeared:
            del self.disappeared[track_id]
    
    def _match_detections_to_tracks(self, detections):
        """Match detections to existing tracks using distance"""
        # Calculate distances between detections and tracks
        track_ids = list(self.tracks.keys())
        
        if len(track_ids) == 0:
            # No existing tracks, register all detections
            for detection in detections:
                self._register_track(detection)
            return
        
        # Calculate distance matrix
        distances = np.zeros((len(detections), len(track_ids)))
        
        for i, detection in enumerate(detections):
            det_center = detection.center
            for j, track_id in enumerate(track_ids):
                track_center = self.tracks[track_id].center
                distance = np.sqrt(
                    (det_center[0] - track_center[0]) ** 2 + 
                    (det_center[1] - track_center[1]) ** 2
                )
                distances[i, j] = distance
        
        # Find best matches
        used_detection_indices = set()
        used_track_indices = set()
        
        # Greedy matching - find closest pairs
        for _ in range(min(len(detections), len(track_ids))):
            min_distance = np.inf
            min_i, min_j = -1, -1
            
            for i in range(len(detections)):
                if i in used_detection_indices:
                    continue
                for j in range(len(track_ids)):
                    if j in used_track_indices:
                        continue
                    if distances[i, j] < min_distance:
                        min_distance = distances[i, j]
                        min_i, min_j = i, j
            
            if min_distance < self.max_distance:
                # Update existing track
                track_id = track_ids[min_j]
                detection = detections[min_i]
                
                self.tracks[track_id].update_position(
                    detection.bbox, detection.confidence
                )
                self.tracks[track_id].class_name = detection.class_name
                self.tracks[track_id].class_id = detection.class_id
                
                self.disappeared[track_id] = 0
                
                used_detection_indices.add(min_i)
                used_track_indices.add(min_j)
            else:
                break
        
        # Register unmatched detections as new tracks
        for i, detection in enumerate(detections):
            if i not in used_detection_indices:
                self._register_track(detection)
        
        # Mark unmatched tracks as disappeared
        for j, track_id in enumerate(track_ids):
            if j not in used_track_indices:
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self._deregister_track(track_id)


class DeepSORTTracker:
    """DeepSORT tracker wrapper"""
    
    def __init__(self, max_age=70):
        """
        Initialize DeepSORT tracker
        
        Args:
            max_age: Maximum number of frames to keep alive a track without detections
        """
        self.max_age = max_age
        try:
            self.tracker = DeepSort(
                model_path=DEEPSORT_WEIGHTS,
                max_age=max_age
            )
            print("✅ DeepSORT tracker initialized")
            self.use_deepsort = True
        except Exception as e:
            print(f"❌ DeepSORT initialization failed: {e}")
            print("🔄 Falling back to simple tracker")
            self.tracker = SimpleTracker()
            self.use_deepsort = False
    
    def update(self, detections, frame=None):
        """
        Update tracker with new detections
        
        Args:
            detections: List of Detection objects
            frame: Current frame (required for DeepSORT)
            
        Returns:
            List of Track objects
        """
        if not detections:
            if self.use_deepsort:
                # Pass empty bbox_xywh and confs arrays, and frame
                tracks = self.tracker.update(np.empty((0, 4)), np.empty((0,)), frame)
                return self._convert_deepsort_tracks(tracks)
            else:
                return self.tracker.update(detections, frame)
        
        if self.use_deepsort:
            return self._update_deepsort(detections, frame)
        else:
            return self.tracker.update(detections, frame)

    def _update_deepsort(self, detections, frame):
        """Update using DeepSORT"""
        try:
            # Prepare bbox_xywh and confs arrays
            bbox_xywh = np.array([d.bbox for d in detections])
            confs = np.array([d.confidence for d in detections])
            tracks = self.tracker.update(bbox_xywh, confs, frame)
            return self._convert_deepsort_tracks(tracks, detections)
        except Exception as e:
            print(f"❌ DeepSORT update error: {e}")
            return []
    
    def _convert_deepsort_tracks(self, deepsort_tracks, detections=None):
        """Convert DeepSORT tracks to our Track format"""
        tracks = []
        
        for track in deepsort_tracks:
            bbox = track[:4]  # [left, top, right, bottom]
            track_id = track[4]
            
            # Try to get class info from detection
            class_name = "unknown"
            class_id = 0
            confidence = 0.5
            
            if detections:
                # Match with closest detection
                min_dist = float('inf')
                best_det = None
                track_center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
                
                for det in detections:
                    det_center = det.center
                    dist = np.sqrt(
                        (det_center[0] - track_center[0])**2 + 
                        (det_center[1] - track_center[1])**2
                    )
                    if dist < min_dist:
                        min_dist = dist
                        best_det = det
                
                if best_det:
                    class_name = best_det.class_name
                    class_id = best_det.class_id
                    confidence = best_det.confidence
            
            track_obj = Track(
                track_id=int(track_id),
                bbox=bbox.astype(int),
                confidence=confidence,
                class_id=class_id,
                class_name=class_name
            )
            
            tracks.append(track_obj)
        
        return tracks
    
    def get_track_count(self):
        """Get current number of active tracks"""
        if self.use_deepsort:
            return len(self.tracker.tracked_detections)
        else:
            return len(self.tracker.tracks)
    
    def reset(self):
        """Reset tracker"""
        if self.use_deepsort:
            self.tracker = DeepSort(
                model_path=DEEPSORT_WEIGHTS,
                max_age=self.max_age
            )
        else:
            self.tracker = SimpleTracker()
        print("🔄 Tracker reset")


def test_tracker():
    """Test function for tracker"""
    print("🧪 Testing DeepSORT Tracker...")
    
    # Import detection class for testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    from yolo_inference import Detection
    
    # Create tracker
    tracker = DeepSORTTracker()
    
    # Create test detections
    test_detections = [
        Detection([100, 100, 200, 200], 0.8, 0, "hello"),
        Detection([300, 150, 400, 250], 0.9, 1, "thank_you")
    ]
    
    # Create test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Update tracker
    tracks = tracker.update(test_detections, test_frame)
    
    print(f"📊 Tracked {len(tracks)} objects:")
    for track in tracks:
        print(f"  Track {track.track_id}: {track.class_name} (conf: {track.confidence:.2f})")
    
    print(f"📈 Total active tracks: {tracker.get_track_count()}")


if __name__ == "__main__":
    test_tracker()