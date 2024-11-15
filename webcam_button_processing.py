import sys
import numpy as np
import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget

class WebcamApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # 윈도우 설정
        self.setWindowTitle("웹캠 영상 처리")
        self.setGeometry(100, 100, 640, 480)

        # 웹캠 초기화
        self.cap = cv2.VideoCapture(0)  # 기본 웹캠 (0번 카메라)

        if not self.cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            sys.exit()

        # 버튼 설정
        self.button1 = QPushButton("원본 영상", self)
        self.button1.clicked.connect(self.set_mode_original)

        self.button2 = QPushButton("왼쪽 반 화면", self)
        self.button2.clicked.connect(self.set_mode_left_half)

        self.button3 = QPushButton("사각형과 삼각형", self)
        self.button3.clicked.connect(self.set_mode_shapes)

        # 라벨 설정
        self.label = QLabel(self)

        # 레이아웃 설정
        layout = QVBoxLayout()
        layout.addWidget(self.button1)
        layout.addWidget(self.button2)
        layout.addWidget(self.button3)
        layout.addWidget(self.label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 타이머 설정 (웹캠 캡처 및 갱신 주기)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.current_mode = 'original'  # 현재 영상 처리 모드 (기본값: 원본 영상)
        self.is_video_on = False

    def set_mode_original(self):
        """원본 영상 모드"""
        self.current_mode = 'original'

    def set_mode_left_half(self):
        """왼쪽 반 화면 모드"""
        self.current_mode = 'left_half'

    def set_mode_shapes(self):
        """사각형과 삼각형 그리기 모드"""
        self.current_mode = 'shapes'

    def update_frame(self):
        # 웹캠에서 프레임을 읽음
        ret, frame = self.cap.read()

        if not ret:
            print("웹캠에서 프레임을 읽을 수 없습니다.")
            return

        # 영상 처리 모드에 따라 다른 작업을 수행
        if self.current_mode == 'left_half':
            # 왼쪽 반만 출력
            frame = frame[:, :frame.shape[1] // 2]  # 좌측 절반만 선택
        elif self.current_mode == 'shapes':
            # 원본 영상에 사각형과 삼각형을 그리기
            frame = self.draw_shapes(frame)

        # BGR에서 RGB로 변환 (OpenCV는 기본 BGR 포맷을 사용)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # QImage 객체로 변환
        height, width, channels = frame_rgb.shape
        bytes_per_line = channels * width
        q_img = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # QLabel에 QPixmap으로 설정
        pixmap = QPixmap.fromImage(q_img)
        self.label.setPixmap(pixmap.scaled(self.label.size(), aspectRatioMode=1))

    def draw_shapes(self, frame):
        """원본 영상에 사각형과 삼각형을 그려서 반환"""
        # 사각형 그리기 (임의 좌표와 크기)
        cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), 2)  # 초록색 사각형

        # 삼각형 그리기 (세 점의 좌표를 설정)
        points = [(300, 50), (400, 150), (250, 150)]
        cv2.polylines(frame, [np.array(points)], isClosed=True, color=(0, 0, 255), thickness=2)  # 빨간색 삼각형

        return frame

    def closeEvent(self, event):
        # 종료 시 웹캠 자원 해제
        if self.cap.isOpened():
            self.cap.release()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = WebcamApp()
    window.show()
    sys.exit(app.exec_())
