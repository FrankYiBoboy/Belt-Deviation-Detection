import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtMultimedia import QMediaPlayer

class VideoPlayerWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("视频播放器")
        self.setGeometry(100, 100, 800, 600)

        self.media_player = QMediaPlayer(self)
        # self.video_widget = QVideoWidget(self)
        self.media_player.setVideoOutput(self.video_widget)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)

        self.button = QPushButton("导入视频", self)
        self.button.clicked.connect(self.open_file_dialog)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        layout.addWidget(self.video_widget)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def open_file_dialog(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("视频文件 (*.mp4 *.avi *.mkv)")
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            self.media_player.setMedia(QUrl.fromLocalFile(file_path))
            self.media_player.play()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoPlayerWindow()
    window.show()
    sys.exit(app.exec_())