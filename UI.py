import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QPushButton, QVBoxLayout, QLineEdit,
    QFileDialog, QStackedWidget, QHBoxLayout, QFrame, QListWidget,
    QListWidgetItem, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QComboBox, QDateEdit
)
from PyQt6.QtCore import QDate
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import Qt

from detector import FaceDetector
from aligner import FaceAligner
from extractor import FeatureExtractor
from matcher import FaceMatcher
from database import FaceDatabase


# ============================
# 公共方法：显示图像（自动缩放以完整显示）
# ============================
def show_qimage(label: QLabel, img):
    """
    将图片自动缩放以完整显示在标签中，保持宽高比
    """
    if img is None or img.size == 0 or len(img.shape) < 2:
        label.clear()
        return
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape
    bytes_per_line = w * c
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    
    # 获取标签的可用大小
    label_size = label.size()
    available_width = label_size.width()
    available_height = label_size.height()
    
    # 如果标签大小无效，直接显示原图
    if available_width <= 0 or available_height <= 0:
        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap)
        return
    
    # 计算缩放比例，保持宽高比
    scale_w = available_width / w
    scale_h = available_height / h
    scale = min(scale_w, scale_h)  # 取较小的缩放比例，确保图片完整显示
    
    # 计算缩放后的尺寸
    new_width = int(w * scale)
    new_height = int(h * scale)
    
    # 确保缩放后的尺寸至少为1像素
    new_width = max(1, new_width)
    new_height = max(1, new_height)
    
    # 缩放图片，保持宽高比，使用平滑转换
    pixmap = QPixmap.fromImage(qimg)
    scaled_pixmap = pixmap.scaled(
        new_width, 
        new_height,
        Qt.AspectRatioMode.KeepAspectRatio,  # 保持宽高比
        Qt.TransformationMode.SmoothTransformation  # 平滑缩放
    )
    
    label.setPixmap(scaled_pixmap)


# ============================
# 注册页面
# ============================
class RegisterPage(QWidget):
    def __init__(self, db, detector, aligner, extractor):
        super().__init__()
        self.db = db
        self.detector = detector
        self.aligner = aligner
        self.extractor = extractor
        self.current_img = None

        # 标题
        title = QLabel("注册人脸")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 图片显示区域
        self.image_label = QLabel()
        self.image_label.setFixedSize(480, 360)
        self.image_label.setStyleSheet(
            "border: 2px solid #CCCCCC; border-radius: 10px; background-color: #F8F8F8;"
        )
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 姓名输入
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("请输入姓名")
        self.name_input.setFixedHeight(40)
        self.name_input.setStyleSheet("font-size:16px; padding:5px;")

        # 选择图片按钮
        self.btn_select = QPushButton("选择图片")
        self.btn_select.clicked.connect(self.select_image)
        self.btn_select.setFixedHeight(40)

        # 注册按钮
        self.btn_register = QPushButton("注册人脸")
        self.btn_register.clicked.connect(self.register_face)
        self.btn_register.setFixedHeight(40)
        self.btn_register.setStyleSheet("background-color:#0080FF;color:white;font-size:16px;")

        # 布局
        layout = QVBoxLayout()
        layout.addWidget(title)
        # 图片标签水平居中
        layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.name_input)
        layout.addWidget(self.btn_select)
        layout.addWidget(self.btn_register)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.setLayout(layout)

    # 选择图片
    def select_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.jpg *.png *.jpeg)")
        if not file:
            return
        img = cv2.imread(file)
        self.current_img = img
        show_qimage(self.image_label, img)

    # 注册人脸
    def register_face(self):
        if self.current_img is None:
            print("请先选择图片")
            return
        name = self.name_input.text().strip()
        if name == "":
            print("请输入姓名")
            return

        try:
            boxes, kps = self.detector.detect(self.current_img)
            if len(boxes) == 0:
                print("没有检测到人脸")
                return

            kp = kps[0] if kps is not None and len(kps) > 0 else None
            aligned = self.aligner.align(self.current_img, keypoints=kp, box=boxes[0])
            feature = self.extractor.extract(aligned)
            
            if feature is None:
                print("特征提取失败")
                return

            # 检查是否已存在相同名字
            if self.db.check_name_exists(name):
                print(f"警告：名字 '{name}' 已存在！")
                print(f"如需更新，请使用不同的名字或修改数据库记录")
                return

            # 检查是否已存在相同人脸（通过特征相似度）
            exists, matched_name, similarity = self.db.find_similar_face(feature, threshold=0.85)
            if exists:
                print(f"警告：检测到与已注册人员 '{matched_name}' 非常相似的人脸（相似度：{similarity:.3f}）")
                print(f"请确认这是否是同一个人，如果是，请使用名字 '{matched_name}' 或更新该记录")
                return

            # 添加新记录
            result = self.db.add(name, feature)
            if result == "inserted":
                print(f"注册成功：{name}")
            else:
                print(f"更新成功：{name}")
                
        except ValueError as e:
            print(f"注册失败: {e}")
        except Exception as e:
            print(f"注册失败: {e}")
            import traceback
            traceback.print_exc()


# ============================
# 检测页面
# ============================
class DetectPage(QWidget):
    def __init__(self, db, detector, aligner, extractor, matcher):
        super().__init__()
        self.db = db
        self.detector = detector
        self.aligner = aligner
        self.extractor = extractor
        self.matcher = matcher
        self.current_img = None

        title = QLabel("人脸检测")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.image_label = QLabel()
        self.image_label.setFixedSize(480, 360)
        self.image_label.setStyleSheet(
            "border: 2px solid #CCCCCC; border-radius: 10px; background-color: #F8F8F8;"
        )
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btn_select = QPushButton("选择图片")
        self.btn_select.clicked.connect(self.select_image)
        self.btn_select.setFixedHeight(40)

        layout = QVBoxLayout()
        layout.addWidget(title)
        # 图片标签水平居中
        layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.btn_select)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.setLayout(layout)

    def select_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.jpg *.png)")
        if not file:
            return
        img = cv2.imread(file)
        self.current_img = img
        self.process(img)

    def process(self, img):
        try:
            frame = img.copy()
            boxes, kps = self.detector.detect(frame)

            db = self.db.get_database()
            for i, box in enumerate(boxes):
                try:
                    kp = kps[i] if kps is not None and i < len(kps) else None
                    aligned = self.aligner.align(frame, keypoints=kp, box=box)
                    fea = self.extractor.extract(aligned)
                    if fea is None:
                        continue
                    name, sim = self.matcher.match(fea, db)

                    x1, y1, x2, y2 = box.astype(int)
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{name} ({sim:.2f})", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # 如果识别成功，记录考勤
                    if name != "Unknown":
                        success, message = self.db.add_attendance(name)
                        if success:
                            print(message)
                        else:
                            print(f"考勤记录: {message}")
                except Exception as e:
                    print(f"处理第 {i+1} 个人脸时出错: {e}")
                    continue

            show_qimage(self.image_label, frame)
        except Exception as e:
            print(f"图片处理失败: {e}")
            import traceback
            traceback.print_exc()


# ============================
# 删除管理页面
# ============================
class DeletePage(QWidget):
    def __init__(self, db):
        super().__init__()
        self.db = db

        title = QLabel("删除人脸信息")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 提示标签
        tip_label = QLabel("选择要删除的人脸信息：")
        tip_label.setFont(QFont("Microsoft YaHei", 12))

        # 列表显示所有已注册的人脸
        self.list_widget = QListWidget()
        self.list_widget.setFixedHeight(300)
        self.list_widget.setStyleSheet(
            "font-size:14px; padding:5px; border: 1px solid #CCCCCC; border-radius: 5px;"
        )

        # 刷新按钮
        self.btn_refresh = QPushButton("刷新列表")
        self.btn_refresh.clicked.connect(self.refresh_list)
        self.btn_refresh.setFixedHeight(40)

        # 删除按钮
        self.btn_delete = QPushButton("删除选中项")
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_delete.setFixedHeight(40)
        self.btn_delete.setStyleSheet("background-color:#FF4444;color:white;font-size:16px;")

        # 布局
        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(tip_label)
        layout.addWidget(self.list_widget)
        layout.addWidget(self.btn_refresh)
        layout.addWidget(self.btn_delete)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.setLayout(layout)
        
        # 初始化时加载列表
        self.refresh_list()

    def refresh_list(self):
        """刷新已注册人脸列表"""
        try:
            self.list_widget.clear()
            names = self.db.get_all_names()
            if len(names) == 0:
                item = QListWidgetItem("（暂无已注册的人脸）")
                item.setFlags(Qt.ItemFlag.NoItemFlags)  # 禁用选择
                self.list_widget.addItem(item)
            else:
                for name in names:
                    count = self.db.get_count_by_name(name)
                    if count > 1:
                        display_text = f"{name} ({count}条记录)"
                    else:
                        display_text = name
                    item = QListWidgetItem(display_text)
                    item.setData(Qt.ItemDataRole.UserRole, name)  # 存储原始名字
                    self.list_widget.addItem(item)
            print(f"已加载 {len(names)} 个已注册人员")
        except Exception as e:
            print(f"刷新列表失败: {e}")
            import traceback
            traceback.print_exc()

    def delete_selected(self):
        """删除选中的人脸信息"""
        current_item = self.list_widget.currentItem()
        if current_item is None:
            QMessageBox.warning(self, "提示", "请先选择要删除的人脸信息！")
            return

        name = current_item.data(Qt.ItemDataRole.UserRole)
        if name is None:
            QMessageBox.warning(self, "提示", "无效的选择！")
            return

        # 确认对话框
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除 '{name}' 的人脸信息吗？\n此操作不可恢复！",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                deleted_count = self.db.delete(name)
                if deleted_count > 0:
                    QMessageBox.information(self, "成功", f"已成功删除 '{name}' 的 {deleted_count} 条记录！")
                    self.refresh_list()  # 刷新列表
                else:
                    QMessageBox.warning(self, "失败", f"删除失败：未找到 '{name}' 的记录")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"删除失败：{str(e)}")
                print(f"删除失败: {e}")
                import traceback
                traceback.print_exc()


# ============================
# 考勤记录页面
# ============================
class AttendancePage(QWidget):
    def __init__(self, db):
        super().__init__()
        self.db = db

        title = QLabel("考勤记录")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 筛选区域
        filter_layout = QHBoxLayout()
        
        name_label = QLabel("姓名筛选:")
        self.name_filter = QComboBox()
        self.name_filter.setEditable(True)
        self.name_filter.setCurrentText("全部")
        self.name_filter.addItem("全部")
        
        date_label = QLabel("日期筛选:")
        self.date_filter = QDateEdit()
        self.date_filter.setCalendarPopup(True)
        self.date_filter.setDate(QDate.currentDate())
        self.date_filter.setDisplayFormat("yyyy-MM-dd")
        
        filter_layout.addWidget(name_label)
        filter_layout.addWidget(self.name_filter)
        filter_layout.addStretch()
        filter_layout.addWidget(date_label)
        filter_layout.addWidget(self.date_filter)
        
        # 操作按钮
        self.btn_add = QPushButton("增加记录")
        self.btn_add.clicked.connect(self.add_record)
        self.btn_add.setFixedHeight(35)
        self.btn_add.setStyleSheet("background-color:#28A745;color:white;font-size:14px;")
        
        self.btn_edit = QPushButton("修改记录")
        self.btn_edit.clicked.connect(self.edit_record)
        self.btn_edit.setFixedHeight(35)
        self.btn_edit.setStyleSheet("background-color:#FFC107;color:black;font-size:14px;")
        
        self.btn_delete = QPushButton("删除记录")
        self.btn_delete.clicked.connect(self.delete_record)
        self.btn_delete.setFixedHeight(35)
        self.btn_delete.setStyleSheet("background-color:#DC3545;color:white;font-size:14px;")
        
        # 刷新按钮
        self.btn_refresh = QPushButton("刷新记录")
        self.btn_refresh.clicked.connect(self.refresh_records)
        self.btn_refresh.setFixedHeight(35)
        
        # 清除筛选按钮
        self.btn_clear = QPushButton("清除筛选")
        self.btn_clear.clicked.connect(self.clear_filter)
        self.btn_clear.setFixedHeight(35)

        # 考勤记录表格
        self.table = QTableWidget()
        self.table.setColumnCount(5)  # ID列隐藏，但需要存储
        self.table.setHorizontalHeaderLabels(["ID", "姓名", "日期", "时间", "完整时间"])
        self.table.setColumnHidden(0, True)  # 隐藏ID列
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)  # 整行选择
        self.table.setStyleSheet("""
            QTableWidget {
                font-size: 13px;
                gridline-color: #CCCCCC;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #0080FF;
                color: white;
                padding: 8px;
                font-weight: bold;
            }
        """)
        self.table.setFixedHeight(350)

        # 统计信息标签
        self.stats_label = QLabel("统计信息：")
        self.stats_label.setFont(QFont("Microsoft YaHei", 11))
        self.stats_label.setStyleSheet("color: #666;")

        # 布局
        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addLayout(filter_layout)
        
        btn_layout1 = QHBoxLayout()
        btn_layout1.addWidget(self.btn_add)
        btn_layout1.addWidget(self.btn_edit)
        btn_layout1.addWidget(self.btn_delete)
        btn_layout1.addStretch()
        layout.addLayout(btn_layout1)
        
        btn_layout2 = QHBoxLayout()
        btn_layout2.addWidget(self.btn_refresh)
        btn_layout2.addWidget(self.btn_clear)
        btn_layout2.addStretch()
        layout.addLayout(btn_layout2)
        
        layout.addWidget(self.table)
        layout.addWidget(self.stats_label)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.setLayout(layout)
        
        # 初始化
        self.load_name_list()
        self.refresh_records()

    def load_name_list(self):
        """加载姓名列表到筛选框"""
        try:
            names = self.db.get_all_names()
            self.name_filter.clear()
            self.name_filter.addItem("全部")
            for name in names:
                self.name_filter.addItem(name)
        except Exception as e:
            print(f"加载姓名列表失败: {e}")

    def clear_filter(self):
        """清除筛选条件"""
        self.name_filter.setCurrentText("全部")
        self.date_filter.setDate(QDate.currentDate())
        self.refresh_records()

    def refresh_records(self):
        """刷新考勤记录"""
        try:
            # 获取筛选条件
            name = None if self.name_filter.currentText() == "全部" else self.name_filter.currentText()
            selected_date = self.date_filter.date().toString("yyyy-MM-dd")
            date = None if not selected_date else selected_date
            
            # 获取考勤记录
            records = self.db.get_attendance_records(name=name, date=date, limit=500)
            
            # 填充表格
            self.table.setRowCount(len(records))
            for row, (record_id, name_val, attendance_time, date_val, time_val) in enumerate(records):
                # 列0: ID (隐藏)
                self.table.setItem(row, 0, QTableWidgetItem(str(record_id)))
                # 列1: 姓名
                self.table.setItem(row, 1, QTableWidgetItem(str(name_val)))
                # 列2: 日期
                self.table.setItem(row, 2, QTableWidgetItem(str(date_val)))
                # 列3: 时间
                self.table.setItem(row, 3, QTableWidgetItem(str(time_val)))
                # 列4: 完整时间
                self.table.setItem(row, 4, QTableWidgetItem(str(attendance_time)))
                
                # 设置不可编辑
                for col in range(5):
                    item = self.table.item(row, col)
                    if item:
                        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            
            # 更新统计信息
            stats = self.db.get_attendance_statistics(name=name, start_date=date, end_date=date)
            total_count = stats['total_count']
            if name:
                stats_text = f"统计信息：{name} - 共 {total_count} 条记录"
            else:
                stats_text = f"统计信息：共 {total_count} 条记录"
            if date:
                stats_text += f"（{date}）"
            self.stats_label.setText(stats_text)
            
            print(f"已加载 {len(records)} 条考勤记录")
            
        except Exception as e:
            print(f"刷新考勤记录失败: {e}")
            import traceback
            traceback.print_exc()

    def add_record(self):
        """增加考勤记录"""
        from PyQt6.QtWidgets import QDialog, QFormLayout, QDateTimeEdit, QDialogButtonBox
        from PyQt6.QtCore import QDateTime
        
        dialog = QDialog(self)
        dialog.setWindowTitle("添加考勤记录")
        dialog.setMinimumWidth(400)
        
        layout = QFormLayout()
        
        # 姓名输入
        name_input = QComboBox()
        name_input.setEditable(True)
        names = self.db.get_all_names()
        for name in names:
            name_input.addItem(name)
        layout.addRow("姓名:", name_input)
        
        # 时间输入
        time_input = QDateTimeEdit()
        time_input.setDateTime(QDateTime.currentDateTime())
        time_input.setCalendarPopup(True)
        time_input.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        layout.addRow("考勤时间:", time_input)
        
        # 按钮
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        dialog.setLayout(layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            name = name_input.currentText().strip()
            if not name:
                QMessageBox.warning(self, "错误", "请输入姓名！")
                return
            
            attendance_time = time_input.dateTime().toString("yyyy-MM-dd HH:mm:ss")
            success, message = self.db.add_attendance_manual(name, attendance_time)
            
            if success:
                QMessageBox.information(self, "成功", message)
                self.load_name_list()  # 刷新姓名列表
                self.refresh_records()
            else:
                QMessageBox.critical(self, "失败", message)

    def edit_record(self):
        """修改考勤记录"""
        from PyQt6.QtWidgets import QDialog, QFormLayout, QDateTimeEdit, QDialogButtonBox
        from PyQt6.QtCore import QDateTime
        
        current_row = self.table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "提示", "请先选择要修改的记录！")
            return
        
        # 获取当前记录信息
        record_id = int(self.table.item(current_row, 0).text())
        current_name = self.table.item(current_row, 1).text()
        current_time = self.table.item(current_row, 4).text()
        
        dialog = QDialog(self)
        dialog.setWindowTitle("修改考勤记录")
        dialog.setMinimumWidth(400)
        
        layout = QFormLayout()
        
        # 姓名输入
        name_input = QComboBox()
        name_input.setEditable(True)
        names = self.db.get_all_names()
        for name in names:
            name_input.addItem(name)
        name_input.setCurrentText(current_name)
        layout.addRow("姓名:", name_input)
        
        # 时间输入
        time_input = QDateTimeEdit()
        time_input.setDateTime(QDateTime.fromString(current_time, "yyyy-MM-dd HH:mm:ss"))
        time_input.setCalendarPopup(True)
        time_input.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        layout.addRow("考勤时间:", time_input)
        
        # 按钮
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        dialog.setLayout(layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            name = name_input.currentText().strip()
            if not name:
                QMessageBox.warning(self, "错误", "请输入姓名！")
                return
            
            attendance_time = time_input.dateTime().toString("yyyy-MM-dd HH:mm:ss")
            success, message = self.db.update_attendance(record_id, name=name, attendance_time=attendance_time)
            
            if success:
                QMessageBox.information(self, "成功", message)
                self.refresh_records()
            else:
                QMessageBox.critical(self, "失败", message)

    def delete_record(self):
        """删除考勤记录"""
        current_row = self.table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "提示", "请先选择要删除的记录！")
            return
        
        # 获取记录信息
        record_id = int(self.table.item(current_row, 0).text())
        name = self.table.item(current_row, 1).text()
        time = self.table.item(current_row, 4).text()
        
        # 确认对话框
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除以下考勤记录吗？\n\n姓名: {name}\n时间: {time}\n\n此操作不可恢复！",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            success, message = self.db.delete_attendance(record_id)
            if success:
                QMessageBox.information(self, "成功", message)
                self.refresh_records()
            else:
                QMessageBox.critical(self, "失败", message)


# ============================
# 主窗口
# ============================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸识别系统")
        self.resize(850, 650)

        # load models (添加错误处理)
        try:
            print("正在加载人脸检测模型...")
            self.detector = FaceDetector()
            print("人脸检测模型加载成功")
        except Exception as e:
            print(f"人脸检测模型加载失败: {e}")
            print("请确保 yolov8x-face-lindevs.pt 文件存在")
            raise
        
        try:
            self.aligner = FaceAligner()
            print("人脸对齐器初始化成功")
        except Exception as e:
            print(f"人脸对齐器初始化失败: {e}")
            raise

        try:
            print("正在加载特征提取模型...")
            self.extractor = FeatureExtractor()
            print("特征提取模型加载成功")
        except Exception as e:
            print(f"特征提取模型加载失败: {e}")
            print("请确保已安装 insightface 库和相关模型")
            raise

        try:
            self.matcher = FaceMatcher()
            print("人脸匹配器初始化成功")
        except Exception as e:
            print(f"人脸匹配器初始化失败: {e}")
            raise

        try:
            print("正在连接数据库...")
            self.db = FaceDatabase()
            print("数据库连接成功")
        except Exception as e:
            print(f"数据库连接失败: {e}")
            print("请确保 MySQL 服务正在运行，且数据库和表已创建")
            raise

        # 页面切换
        self.stack = QStackedWidget()
        self.register_page = RegisterPage(self.db, self.detector, self.aligner, self.extractor)
        self.detect_page = DetectPage(self.db, self.detector, self.aligner, self.extractor, self.matcher)
        self.delete_page = DeletePage(self.db)
        self.attendance_page = AttendancePage(self.db)

        self.stack.addWidget(self.register_page)
        self.stack.addWidget(self.detect_page)
        self.stack.addWidget(self.delete_page)
        self.stack.addWidget(self.attendance_page)

        # 按钮
        btn_reg = QPushButton("注册人脸")
        btn_reg.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        btn_reg.setFixedHeight(40)

        btn_det = QPushButton("检测人脸")
        btn_det.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        btn_det.setFixedHeight(40)

        btn_delete = QPushButton("删除管理")
        btn_delete.clicked.connect(self.switch_to_delete_page)
        btn_delete.setFixedHeight(40)

        btn_attendance = QPushButton("考勤记录")
        btn_attendance.clicked.connect(self.switch_to_attendance_page)
        btn_attendance.setFixedHeight(40)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_reg)
        btn_layout.addWidget(btn_det)
        btn_layout.addWidget(btn_delete)
        btn_layout.addWidget(btn_attendance)

        layout = QVBoxLayout()
        layout.addWidget(self.stack)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def switch_to_delete_page(self):
        """切换到删除页面并刷新列表"""
        self.stack.setCurrentIndex(2)
        self.delete_page.refresh_list()

    def switch_to_attendance_page(self):
        """切换到考勤记录页面并刷新列表"""
        self.stack.setCurrentIndex(3)
        self.attendance_page.load_name_list()  # 刷新姓名列表
        self.attendance_page.refresh_records()


# 入口
if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        w = MainWindow()
        w.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"程序启动失败: {e}")
        import traceback
        traceback.print_exc()
        input("按回车键退出...")
        sys.exit(1)
