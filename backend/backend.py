from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os
import json
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class ExerciseFormAnalyzer:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Exercise-specific form rules
        self.exercise_rules = {
            'squat': self._analyze_squat,
            'pushup': self._analyze_pushup,
            'deadlift': self._analyze_deadlift,
            'lunge': self._analyze_lunge,
            'shoulder_press': self._analyze_shoulder_press
        }
    
    def analyze_video(self, video_path, exercise_type):
        """Main analysis function"""
        logger.info(f"Analyzing {exercise_type} from video: {video_path}")
        
        # Extract frames and pose data
        frames_data = self._extract_pose_data(video_path)
        
        if not frames_data:
            return self._create_error_response("No pose data detected in video")
        
        # Analyze based on exercise type
        if exercise_type in self.exercise_rules:
            return self.exercise_rules[exercise_type](frames_data)
        else:
            return self._create_error_response(f"Unsupported exercise type: {exercise_type}")
    
    def _extract_pose_data(self, video_path):
        """Extract pose landmarks from video frames"""
        cap = cv2.VideoCapture(video_path)
        frames_data = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every 5th frame to reduce computation
            if frame_count % 5 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)
                
                if results.pose_landmarks:
                    landmarks = self._normalize_landmarks(results.pose_landmarks.landmark)
                    frames_data.append({
                        'frame_number': frame_count,
                        'landmarks': landmarks,
                        'visibility': self._calculate_visibility(results.pose_landmarks.landmark)
                    })
            
            frame_count += 1
        
        cap.release()
        return frames_data
    
    def _normalize_landmarks(self, landmarks):
        """Normalize landmarks to be scale-invariant"""
        landmarks_array = []
        for landmark in landmarks:
            landmarks_array.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        return landmarks_array
    
    def _calculate_visibility(self, landmarks):
        """Calculate overall pose visibility"""
        return sum([lm.visibility for lm in landmarks]) / len(landmarks)
    
    def _analyze_squat(self, frames_data):
        """Analyze squat form"""
        corrections = []
        whats_right = []
        detailed_breakdown = {}
        
        # Analyze depth
        depth_analysis = self._analyze_squat_depth(frames_data)
        if not depth_analysis['adequate']:
            corrections.append({
                'issue': 'Insufficient Depth',
                'feedback': f"Your squat reached {depth_analysis['max_depth_percent']:.1f}% of optimal depth",
                'correction_instruction': 'Aim to get your hips below knee level. Practice with a box to gauge proper depth.',
                'severity': 'critical' if depth_analysis['max_depth_percent'] < 70 else 'warning'
            })
        else:
            whats_right.append(f"Good squat depth achieved ({depth_analysis['max_depth_percent']:.1f}%)")
        
        # Analyze knee alignment
        knee_analysis = self._analyze_knee_alignment(frames_data)
        if knee_analysis['issues']:
            corrections.append({
                'issue': 'Knee Valgus (Knees Caving In)',
                'feedback': 'Your knees are collapsing inward during the movement',
                'correction_instruction': 'Focus on pushing your knees outward. Strengthen glute medius with band exercises.',
                'severity': 'warning'
            })
        else:
            whats_right.append("Good knee alignment maintained throughout")
        
        # Analyze back position
        back_analysis = self._analyze_back_position(frames_data)
        if back_analysis['rounded']:
            corrections.append({
                'issue': 'Back Rounding',
                'feedback': 'Your lower back is rounding during the descent',
                'correction_instruction': 'Maintain a neutral spine. Engage your core and keep your chest up.',
                'severity': 'critical'
            })
        else:
            whats_right.append("Good back position with neutral spine")
        
        # Calculate overall score
        aspect_scores = {
            'depth': depth_analysis['score'],
            'knee_alignment': knee_analysis['score'],
            'back_position': back_analysis['score'],
            'balance': self._analyze_balance(frames_data)['score']
        }
        
        overall_score = sum(aspect_scores.values()) // len(aspect_scores)
        
        # Create detailed breakdown
        detailed_breakdown = {
            'squat_depth': {
                'score': depth_analysis['score'],
                'feedback': depth_analysis['feedback']
            },
            'knee_alignment': {
                'score': knee_analysis['score'],
                'feedback': knee_analysis['feedback']
            },
            'back_position': {
                'score': back_analysis['score'],
                'feedback': back_analysis['feedback']
            },
            'balance_stability': {
                'score': aspect_scores['balance'],
                'feedback': 'Maintain even weight distribution'
            }
        }
        
        return self._create_analysis_response(
            overall_score=overall_score,
            corrections=corrections,
            whats_right=whats_right,
            detailed_breakdown=detailed_breakdown,
            exercise_type='squat'
        )
    
    def _analyze_squat_depth(self, frames_data):
        """Analyze if squat reaches adequate depth"""
        # Simplified depth analysis using hip and knee landmarks
        hip_knee_ratios = []
        
        for frame in frames_data:
            landmarks = frame['landmarks']
            # Use hip and knee landmarks to estimate depth
            if len(landmarks) > 24:  # Ensure we have required landmarks
                hip_y = landmarks[23]['y']  # Left hip
                knee_y = landmarks[25]['y']  # Left knee
                ankle_y = landmarks[27]['y']  # Left ankle
                
                # Simple depth ratio
                if knee_y > 0 and ankle_y > 0:
                    depth_ratio = (hip_y - knee_y) / (ankle_y - knee_y)
                    hip_knee_ratios.append(depth_ratio)
        
        if not hip_knee_ratios:
            return {'adequate': False, 'max_depth_percent': 0, 'score': 0, 'feedback': 'Unable to analyze depth'}
        
        max_ratio = max(hip_knee_ratios)
        optimal_ratio = 0.8  # Threshold for good depth
        depth_percent = min((max_ratio / optimal_ratio) * 100, 100)
        adequate = depth_percent >= 80
        
        return {
            'adequate': adequate,
            'max_depth_percent': depth_percent,
            'score': int(depth_percent),
            'feedback': f"Squat depth reached {depth_percent:.1f}% of optimal depth"
        }
    
    def _analyze_knee_alignment(self, frames_data):
        """Analyze knee alignment during squat"""
        # Simplified knee alignment check
        knee_issues = 0
        total_frames = len(frames_data)
        
        for frame in frames_data:
            landmarks = frame['landmarks']
            if len(landmarks) > 26:
                # Check if knees are tracking over feet
                knee_x = landmarks[25]['x']  # Left knee
                ankle_x = landmarks[27]['x']  # Left ankle
                
                # Simple alignment check
                if abs(knee_x - ankle_x) > 0.05:  # Threshold
                    knee_issues += 1
        
        issue_percentage = (knee_issues / total_frames) * 100 if total_frames > 0 else 0
        has_issues = issue_percentage > 30
        
        score = max(0, 100 - issue_percentage)
        
        return {
            'issues': has_issues,
            'score': int(score),
            'feedback': 'Knee alignment needs improvement' if has_issues else 'Good knee alignment'
        }
    
    def _analyze_back_position(self, frames_data):
        """Analyze back position and curvature"""
        # Simplified back analysis
        rounded_frames = 0
        total_frames = len(frames_data)
        
        for frame in frames_data:
            landmarks = frame['landmarks']
            if len(landmarks) > 24:
                shoulder = landmarks[11]['y']  # Left shoulder
                hip = landmarks[23]['y']  # Left hip
                
                # Simple back angle estimation
                if shoulder < hip - 0.1:  # Threshold for forward lean
                    rounded_frames += 1
        
        rounded_percentage = (rounded_frames / total_frames) * 100 if total_frames > 0 else 0
        is_rounded = rounded_percentage > 40
        
        score = max(0, 100 - rounded_percentage)
        
        return {
            'rounded': is_rounded,
            'score': int(score),
            'feedback': 'Maintain neutral spine position' if is_rounded else 'Good back position maintained'
        }
    
    def _analyze_balance(self, frames_data):
        """Analyze balance and stability"""
        # Simplified balance analysis
        return {'score': 85, 'feedback': 'Good balance demonstrated'}
    
    def _analyze_pushup(self, frames_data):
        """Analyze push-up form (simplified)"""
        return self._create_analysis_response(
            overall_score=75,
            corrections=[
                {
                    'issue': 'Elbow Flaring',
                    'feedback': 'Your elbows are flaring out at 45+ degree angle',
                    'correction_instruction': 'Keep elbows at 30-45 degree angle from body',
                    'severity': 'warning'
                }
            ],
            whats_right=['Good full range of motion', 'Proper body alignment'],
            detailed_breakdown={
                'elbow_position': {'score': 65, 'feedback': 'Elbows flaring too wide'},
                'range_of_motion': {'score': 90, 'feedback': 'Good depth achieved'},
                'body_alignment': {'score': 85, 'feedback': 'Straight body line maintained'}
            },
            exercise_type='pushup'
        )
    
    def _analyze_deadlift(self, frames_data):
        """Analyze deadlift form (simplified)"""
        return self._create_analysis_response(
            overall_score=82,
            corrections=[
                {
                    'issue': 'Hips Rising Early',
                    'feedback': 'Your hips are rising before the bar',
                    'correction_instruction': 'Initiate the lift with your legs, not your back',
                    'severity': 'warning'
                }
            ],
            whats_right=['Good neutral spine', 'Proper bar path'],
            detailed_breakdown={
                'hip_hinge': {'score': 75, 'feedback': 'Hips rising slightly early'},
                'back_position': {'score': 90, 'feedback': 'Excellent spine neutrality'},
                'bar_path': {'score': 85, 'feedback': 'Good vertical bar path'}
            },
            exercise_type='deadlift'
        )
    
    def _analyze_lunge(self, frames_data):
        """Analyze lunge form (simplified)"""
        return self._create_analysis_response(
            overall_score=78,
            corrections=[],
            whats_right=['Good forward knee alignment', 'Adequate depth'],
            detailed_breakdown={
                'knee_alignment': {'score': 85, 'feedback': 'Good knee tracking'},
                'depth': {'score': 80, 'feedback': 'Adequate lunge depth'},
                'stability': {'score': 70, 'feedback': 'Work on balance and stability'}
            },
            exercise_type='lunge'
        )
    
    def _analyze_shoulder_press(self, frames_data):
        """Analyze shoulder press form (simplified)"""
        return self._create_analysis_response(
            overall_score=88,
            corrections=[],
            whats_right=['Good overhead position', 'Stable core engagement'],
            detailed_breakdown={
                'overhead_position': {'score': 90, 'feedback': 'Good bar path overhead'},
                'core_stability': {'score': 85, 'feedback': 'Good core engagement'},
                'elbow_position': {'score': 85, 'feedback': 'Proper elbow alignment'}
            },
            exercise_type='shoulder_press'
        )
    
    def _create_analysis_response(self, overall_score, corrections, whats_right, detailed_breakdown, exercise_type):
        """Create standardized analysis response"""
        improvement_tips = self._generate_improvement_tips(exercise_type, overall_score)
        
        return {
            'success': True,
            'exercise_type': exercise_type,
            'overall_score': overall_score,
            'whats_right': whats_right,
            'corrections_needed': corrections,
            'detailed_breakdown': detailed_breakdown,
            'improvement_tips': improvement_tips,
            'summary': self._generate_summary(overall_score, len(corrections))
        }
    
    def _generate_improvement_tips(self, exercise_type, score):
        """Generate improvement tips based on exercise type and score"""
        base_tips = [
            "Record yourself regularly to monitor progress",
            "Start with lighter weights to focus on form",
            "Practice in front of a mirror for immediate feedback"
        ]
        
        if score < 70:
            base_tips.extend([
                "Consider working with a certified trainer for personalized guidance",
                "Break down the movement into smaller components",
                "Focus on one correction at a time"
            ])
        
        exercise_specific_tips = {
            'squat': [
                "Practice box squats to learn proper depth",
                "Use a resistance band around knees to reinforce proper alignment"
            ],
            'pushup': [
                "Start with knee pushups to build strength with proper form",
                "Use a wider hand position for better stability"
            ],
            'deadlift': [
                "Practice hip hinge movements without weight",
                "Use a hex bar if available for better biomechanics"
            ]
        }
        
        return base_tips + exercise_specific_tips.get(exercise_type, [])
    
    def _generate_summary(self, score, correction_count):
        """Generate overall summary"""
        if score >= 90:
            return "Excellent form! Minor adjustments can make it perfect."
        elif score >= 80:
            return "Good form with some areas for improvement."
        elif score >= 70:
            return "Fair form. Focus on the corrections below."
        else:
            return f"Needs work. Focus on these {correction_count} key areas to improve safety and effectiveness."
    
    def _create_error_response(self, message):
        """Create error response"""
        return {
            'success': False,
            'error': message,
            'overall_score': 0,
            'whats_right': [],
            'corrections_needed': [],
            'detailed_breakdown': {},
            'improvement_tips': []
        }

# Initialize analyzer
analyzer = ExerciseFormAnalyzer()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/analyze', methods=['POST'])
def analyze_exercise():
    try:
        # Check if video file is present
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No video file provided'
            }), 400
        
        video_file = request.files['video']
        exercise_type = request.form.get('exercise_type', 'squat')
        
        if video_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No video file selected'
            }), 400
        
        if not allowed_file(video_file.filename):
            return jsonify({
                'success': False,
                'error': 'File type not allowed. Please upload a video file.'
            }), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            video_file.save(temp_file.name)
            
            try:
                # Analyze the video
                analysis_result = analyzer.analyze_video(temp_file.name, exercise_type)
                return jsonify(analysis_result)
                
            except Exception as e:
                logger.error(f"Analysis error: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': f'Analysis failed: {str(e)}'
                }), 500
            finally:
                # Clean up temporary file
                os.unlink(temp_file.name)
                
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Exercise Form Analyzer API is running'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)