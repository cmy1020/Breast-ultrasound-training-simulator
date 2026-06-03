import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore


class USDisplayWidget(QtWidgets.QWidget):
    """
    Qt 窗口，用来显示 UltrasoundSimulator.us_image
    """
    def __init__(self, us_simulator, parent=None):
        super().__init__(parent)
        self.us_sim = us_simulator

        self.label = QtWidgets.QLabel(self)
        self.label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)

        # 定时刷新
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(50)  # 约 20fps

    def update_image(self):
        img = self.us_sim.us_image
        if img is None or img.size == 0:
            return

        # img: (H, W) 浮点 [0,1]，Y=0 在上，所以 origin='upper' 的效果在这里需要翻转
        img = np.flipud(img)  # 纵轴翻转使深度向下

        img8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        h, w = img8.shape

        # 创建 QImage（灰度图）
        qimg = QtGui.QImage(img8.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        pix = QtGui.QPixmap.fromImage(qimg)

        self.label.setPixmap(pix)
        self.label.setFixedSize(w, h)
        self.setFixedSize(w + 10, h + 10)