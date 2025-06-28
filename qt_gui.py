import sys
import os
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout,
    QTextEdit, QMessageBox, QFrame, QSizePolicy, QSlider, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QPixmap, QPalette, QColor, QFont
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from main import process_and_generate_audio
from components.yolo_inference import YOLO12Detector
from utils.draw import DrawingUtils

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign2Speech - Professional Sign Language Interpreter")
        self.setGeometry(100, 100, 1000, 600)
        self.setStyleSheet("""
            QWidget {
                background-color: #2C3E50;
                color: #ECF0F1;
            }
            QPushButton {
                background-color: #3498DB;
                border: none;
                border-radius: 5px;
                padding: 10px;
                color: white;
                font-weight: bold;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:disabled {
                background-color: #7F8C8D;
            }
            QTextEdit {
                background-color: #34495E;
                border: 1px solid #7F8C8D;
                border-radius: 5px;
                padding: 5px;
                color: #ECF0F1;
            }
        """)
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        left_panel = QVBoxLayout()
        right_panel = QVBoxLayout()

        # Left Panel - Controls
        title_label = QLabel("Sign2Speech Interpreter")
        title_label.setFont(QFont('Arial', 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(title_label)

        self.label = QLabel("Select an image or video file for inference:")
        self.label.setFont(QFont('Arial', 10))
        self.label.setWordWrap(True)
        left_panel.addWidget(self.label)

        buttons_layout = QHBoxLayout()
        self.btn_image = QPushButton("📷 Select Image")
        self.btn_image.clicked.connect(self.select_image)
        buttons_layout.addWidget(self.btn_image)

        self.btn_video = QPushButton("🎥 Select Video")
        self.btn_video.clicked.connect(self.select_video)
        buttons_layout.addWidget(self.btn_video)
        left_panel.addLayout(buttons_layout)

        self.btn_infer = QPushButton("🔍 Run Inference & Generate Voice")
        self.btn_infer.clicked.connect(self.run_inference)
        self.btn_infer.setEnabled(False)
        left_panel.addWidget(self.btn_infer)

        # Results Section
        results_label = QLabel("Results:")
        results_label.setFont(QFont('Arial', 12, QFont.Bold))
        left_panel.addWidget(results_label)

        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.result_box.setMinimumHeight(150)
        left_panel.addWidget(self.result_box)

        # Right Panel - Image Preview
        preview_frame = QFrame()
        preview_frame.setFrameStyle(QFrame.StyledPanel)
        preview_frame.setStyleSheet("background-color: #34495E; border-radius: 10px; padding: 10px;")
        preview_layout = QVBoxLayout(preview_frame)

        preview_label = QLabel("Image Preview")
        preview_label.setFont(QFont('Arial', 12, QFont.Bold))
        preview_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(preview_label)

        self.image_preview = QLabel()
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setMinimumSize(400, 400)
        self.image_preview.setStyleSheet("background-color: #2C3E50; border-radius: 5px;")
        preview_layout.addWidget(self.image_preview)

        right_panel.addWidget(preview_frame)

        # Add panels to main layout
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setMaximumWidth(500)

        right_widget = QWidget()
        right_widget.setLayout(right_panel)

        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)

        # Audio Section
        audio_frame = QFrame()
        audio_frame.setFrameStyle(QFrame.StyledPanel)
        audio_frame.setStyleSheet("background-color: #34495E; border-radius: 10px; padding: 10px;")
        audio_layout = QVBoxLayout(audio_frame)

        audio_label = QLabel("Audio Player")
        audio_label.setFont(QFont('Arial', 12, QFont.Bold))
        audio_label.setAlignment(Qt.AlignCenter)
        audio_layout.addWidget(audio_label)

        self.audio_list = QListWidget()
        self.audio_list.setMaximumHeight(100)
        self.audio_list.setStyleSheet("""
            QListWidget {
                background-color: #2C3E50;
                border: 1px solid #7F8C8D;
                border-radius: 5px;
                color: #ECF0F1;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #7F8C8D;
            }
            QListWidget::item:selected {
                background-color: #3498DB;
            }
        """)
        audio_layout.addWidget(self.audio_list)

        audio_controls = QHBoxLayout()
        self.btn_play = QPushButton("▶️ Play")
        self.btn_play.clicked.connect(self.play_audio)
        self.btn_play.setEnabled(False)
        audio_controls.addWidget(self.btn_play)

        self.btn_stop = QPushButton("⏹️ Stop")
        self.btn_stop.clicked.connect(self.stop_audio)
        self.btn_stop.setEnabled(False)
        audio_controls.addWidget(self.btn_stop)
        audio_layout.addLayout(audio_controls)

        left_panel.addWidget(audio_frame)

        self.setLayout(main_layout)
        self.selected_file = None
        self.processed_image_path = None
        
        # Initialize media player
        self.media_player = QMediaPlayer()
        
        # Initialize detector and drawing utils
        self.detector = YOLO12Detector("F:/NLPPRO/sign2speech/models/sign.pt")
        self.drawer = DrawingUtils()

    def select_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if file:
            self.selected_file = file
            self.label.setText(f"Selected Image: {file}")
            self.btn_infer.setEnabled(True)
            
            # Display image preview
            pixmap = QPixmap(file)
            scaled_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_preview.setPixmap(scaled_pixmap)
            self.image_preview.setStyleSheet("background-color: #2C3E50; border-radius: 5px; padding: 5px;")
            self.result_box.clear()

    def select_video(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Videos (*.mp4 *.avi *.mov)")
        if file:
            self.selected_file = file
            self.label.setText(f"Selected Video: {file}")
            self.btn_infer.setEnabled(True)
            
            # Clear previous preview and results
            self.image_preview.clear()
            self.image_preview.setText("Video Selected\n(Preview not available for videos)")
            self.image_preview.setStyleSheet("""
                background-color: #2C3E50;
                border-radius: 5px;
                padding: 5px;
                qproperty-alignment: AlignCenter;
                color: #ECF0F1;
            """)
            self.result_box.clear()

    def run_inference(self):
        if not self.selected_file:
            QMessageBox.warning(self, "No File Selected", "Please select an image or video file first.")
            return

        try:
            # Show processing status
            self.btn_infer.setEnabled(False)
            self.btn_infer.setText("🔄 Processing...")
            self.result_box.setText("Processing your file...\nThis may take a moment...")
            QApplication.processEvents()

            # Run inference and get detections
            frame = cv2.imread(self.selected_file)
            if frame is None:
                self.result_box.setText("❌ Error: Could not load image file.")
                return

            # Detect objects
            detections = self.detector.detect(frame)
            
            # Draw bounding boxes and labels
            annotated_frame = frame.copy()
            detected_classes = []
            
            for det in detections:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, det.bbox)
                
                # Get color based on confidence
                color = self.drawer.get_confidence_color(det.confidence)
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with confidence
                label = f"{det.class_name}: {det.confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw label background
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                detected_classes.append(det.class_name)
            
            # Save annotated image
            timestamp = int(time.time())
            self.processed_image_path = f"logs/processed_image_{timestamp}.jpg"
            os.makedirs("logs", exist_ok=True)
            cv2.imwrite(self.processed_image_path, annotated_frame)
            
            # Display annotated image in preview
            self.display_processed_image(self.processed_image_path)
            
            # Run audio generation
            result = process_and_generate_audio(self.selected_file)
            
            # Update audio list
            self.update_audio_list()
            
            # Format results nicely
            output_text = "✨ Processing Complete! ✨\n\n"
            output_text += "📝 Detection Results:\n"
            if detected_classes:
                output_text += f"Detected classes: {', '.join(set(detected_classes))}\n"
                output_text += f"Total detections: {len(detections)}\n\n"
            else:
                output_text += "No objects detected.\n\n"
            output_text += "🔊 Audio Output:\n"
            output_text += "Audio files generated and available in player below."
            
            self.result_box.setText(output_text)
            
        except Exception as e:
            self.result_box.setText(f"❌ Error:\n{str(e)}")
        finally:
            self.btn_infer.setEnabled(True)
            self.btn_infer.setText("🔍 Run Inference & Generate Voice")

    def display_processed_image(self, image_path):
        """Display the processed image with bounding boxes"""
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_preview.setPixmap(scaled_pixmap)
        self.image_preview.setStyleSheet("background-color: #2C3E50; border-radius: 5px; padding: 5px;")
    
    def update_audio_list(self):
        """Update the audio file list"""
        self.audio_list.clear()
        voices_dir = "voices"
        if os.path.exists(voices_dir):
            audio_files = [f for f in os.listdir(voices_dir) if f.endswith('.wav')]
            for audio_file in audio_files:
                item = QListWidgetItem(audio_file)
                item.setData(Qt.UserRole, os.path.join(voices_dir, audio_file))
                self.audio_list.addItem(item)
            
            if audio_files:
                self.btn_play.setEnabled(True)
    
    def play_audio(self):
        """Play selected audio file"""
        current_item = self.audio_list.currentItem()
        if current_item:
            audio_path = current_item.data(Qt.UserRole)
            if os.path.exists(audio_path):
                url = QUrl.fromLocalFile(os.path.abspath(audio_path))
                content = QMediaContent(url)
                self.media_player.setMedia(content)
                self.media_player.play()
                self.btn_stop.setEnabled(True)
                self.btn_play.setText("🔄 Playing...")
                
                # Connect finished signal to reset button
                self.media_player.mediaStatusChanged.connect(self.on_media_status_changed)
    
    def stop_audio(self):
        """Stop audio playback"""
        self.media_player.stop()
        self.btn_stop.setEnabled(False)
        self.btn_play.setText("▶️ Play")
    
    def on_media_status_changed(self, status):
        """Handle media status changes"""
        if status == QMediaPlayer.EndOfMedia:
            self.btn_stop.setEnabled(False)
            self.btn_play.setText("▶️ Play")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())