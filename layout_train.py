import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QStackedWidget, QGridLayout, QLabel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 메인 윈도우 설정
        self.setWindowTitle("PyQt5 QStackedWidget 내 레이아웃 전환")
        self.setGeometry(100, 100, 400, 300)

        # QGridLayout을 메인 레이아웃으로 설정
        self.main_layout = QGridLayout()

        # QStackedWidget을 그리드 레이아웃에 추가
        self.stacked_widget = QStackedWidget()

        # 레이아웃을 담을 위젯들 생성
        self.layout1 = QWidget()
        self.layout2 = QWidget()

        # 첫 번째 화면 레이아웃 (기본 화면)
        self.layout1_ui()

        # 두 번째 화면 레이아웃 (변경된 화면)
        self.layout2_ui()

        # QStackedWidget에 레이아웃 추가
        self.stacked_widget.addWidget(self.layout1)
        self.stacked_widget.addWidget(self.layout2)

        # 첫 번째 화면이 기본적으로 보이도록 설정
        self.stacked_widget.setCurrentIndex(0)

        # QGridLayout에 QStackedWidget을 배치
        self.main_layout.addWidget(self.stacked_widget, 0, 0)  # (row, column)

        btn_chk=QPushButton("그리드체크용")
        self.main_layout.addWidget(btn_chk,1,0)

        # 메인 레이아웃을 QMainWindow의 centralWidget으로 설정
        central_widget = QWidget()
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

    def layout1_ui(self):
        # 첫 번째 화면의 레이아웃 (기본 화면)
        layout = QVBoxLayout()
        label = QLabel("기본 화면", self)
        button = QPushButton("화면 변경", self)
        button.clicked.connect(self.switch_to_layout2)

        layout.addWidget(label)
        layout.addWidget(button)

        self.layout1.setLayout(layout)

    def layout2_ui(self):
        # 두 번째 화면의 레이아웃 (변경된 화면)
        layout = QVBoxLayout()
        label = QLabel("변경된 화면", self)
        button = QPushButton("원래 화면으로 돌아가기", self)
        button.clicked.connect(self.switch_to_layout1)

        layout.addWidget(label)
        layout.addWidget(button)

        self.layout2.setLayout(layout)

    def switch_to_layout1(self):
        # 첫 번째 화면으로 전환
        self.stacked_widget.setCurrentIndex(0)

    def switch_to_layout2(self):
        # 두 번째 화면으로 전환
        self.stacked_widget.setCurrentIndex(1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
