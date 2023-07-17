import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLabel, QVBoxLayout,QFileDialog
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, pyqtSlot, QThread,QUrl


class VideoThread(QThread):
    frame_processed = pyqtSignal(QImage)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    # def run(self):
    #     video_capture = cv2.VideoCapture(self.video_path)
    #     frame_interval = int(video_capture.get(cv2.CAP_PROP_FPS)) * 2  # 每两秒读取一帧
    #     count = 0
    #     while video_capture.isOpened():
    #         ret, frame = video_capture.read()
    #         if not ret:
    #             break

    #         count += 1
    #         if count % frame_interval != 0:
    #             continue

    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    #         image = QImage(gray, gray.shape[1], gray.shape[0], QImage.Format_RGB888)
    #         self.frame_processed.emit(image)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频处理")
        self.setGeometry(100, 100, 800, 600)

        self.video_widget = QVideoWidget(self)
        self.video_widget.setGeometry(50, 50, 640, 360)

        self.process_widget = QLabel(self)
        self.process_widget.setGeometry(450, 420, 320, 240)
        self.process_widget.setAlignment(Qt.AlignCenter)

        self.select_button = QPushButton("选择视频", self)
        self.select_button.setGeometry(50, 10, 100, 30)
        self.select_button.clicked.connect(self.select_video)

        self.process_button = QPushButton("开始处理", self)
        self.process_button.setGeometry(650, 10, 100, 30)
        self.process_button.clicked.connect(self.start_processing)

        self.video_path = ""
        self.video_thread = None
        self.video_player = QMediaPlayer(self)
        self.video_player.setVideoOutput(self.video_widget)

    @pyqtSlot()
    def select_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi)")
        if self.video_path:
            self.video_player.setMedia(QMediaContent(QUrl.fromLocalFile(self.video_path)))
            self.video_player.play()

    @pyqtSlot()
    def start_processing(self):
        if self.video_path and not self.video_thread:
            self.video_thread = VideoThread(self.video_path)
            self.video_thread.frame_processed.connect(self.update_image)
            self.video_thread.start()

    @pyqtSlot(QImage)
    def update_image(self, image):
        self.process_widget.setPixmap(QPixmap.fromImage(image.scaled(self.process_widget.size(), Qt.KeepAspectRatio)))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())