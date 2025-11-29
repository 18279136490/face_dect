import pymysql
import numpy as np
from datetime import datetime

class FaceDatabase:
    def __init__(self, host="localhost", user="root", password="123456", database="face_recognition"):
        try:
            self.conn = pymysql.connect(
                host=host,
                user=user,
                password=password,
                database=database
            )
            self.cursor = self.conn.cursor()
        except Exception as e:
            print(f"数据库连接失败: {e}")
            print("请确保 MySQL 服务正在运行，且数据库和表已创建")
            raise

    def check_name_exists(self, name):
        """检查名字是否已存在"""
        sql = "SELECT COUNT(*) FROM face_features WHERE name = %s"
        self.cursor.execute(sql, (name,))
        count = self.cursor.fetchone()[0]
        return count > 0

    def find_similar_face(self, feature, threshold=0.85):
        """
        检查是否存在相似的人脸特征
        返回: (是否存在, 匹配的姓名, 相似度)
        threshold: 相似度阈值，默认0.85（较高，确保是同一个人）
        """
        sql = "SELECT name, feature FROM face_features"
        self.cursor.execute(sql)
        rows = self.cursor.fetchall()

        for name, fea_bytes in rows:
            db_feature = np.frombuffer(fea_bytes, dtype=np.float32)
            # 计算余弦相似度
            cos_sim = np.dot(feature, db_feature)
            if cos_sim >= threshold:
                return True, name, cos_sim
        return False, None, 0.0

    def add(self, name, feature, overwrite_if_exists=False):
        """
        保存人脸特征到数据库
        overwrite_if_exists: 如果名字已存在，是否覆盖（默认False）
        """
        feature_bytes = feature.tobytes()
        
        if self.check_name_exists(name):
            if overwrite_if_exists:
                # 更新现有记录
                sql = "UPDATE face_features SET feature = %s WHERE name = %s"
                self.cursor.execute(sql, (feature_bytes, name))
                self.conn.commit()
                return "updated"
            else:
                raise ValueError(f"名字 '{name}' 已存在，请使用不同的名字或选择覆盖")
        
        # 插入新记录
        sql = "INSERT INTO face_features (name, feature) VALUES (%s, %s)"
        self.cursor.execute(sql, (name, feature_bytes))
        self.conn.commit()
        return "inserted"

    def load_all(self):
        """
        加载所有特征
        返回: dict，key为名字，value为特征向量
        注意：如果有多个同名记录，只返回第一个
        """
        sql = "SELECT name, feature FROM face_features"
        self.cursor.execute(sql)
        rows = self.cursor.fetchall()

        db = {}
        for name, fea_bytes in rows:
            feature = np.frombuffer(fea_bytes, dtype=np.float32)
            # 如果名字已存在，使用列表存储多个特征
            if name in db:
                if not isinstance(db[name], list):
                    db[name] = [db[name]]  # 将单个特征转为列表
                db[name].append(feature)
            else:
                db[name] = feature
        return db
    
    def get_database(self):
        """
        获取数据库（用于匹配）
        返回: dict，如果有多个同名记录，使用第一个特征
        """
        db = self.load_all()
        # 如果有列表，只取第一个特征（用于匹配）
        simplified_db = {}
        for name, value in db.items():
            if isinstance(value, list):
                simplified_db[name] = value[0]  # 使用第一个特征
            else:
                simplified_db[name] = value
        return simplified_db

    def get_all_names(self):
        """获取所有已注册的姓名列表"""
        sql = "SELECT DISTINCT name FROM face_features ORDER BY name"
        self.cursor.execute(sql)
        rows = self.cursor.fetchall()
        return [row[0] for row in rows]

    def delete(self, name):
        """
        删除指定名字的人脸信息
        返回: 删除的记录数
        """
        sql = "DELETE FROM face_features WHERE name = %s"
        self.cursor.execute(sql, (name,))
        deleted_count = self.cursor.rowcount
        self.conn.commit()
        return deleted_count

    def get_count_by_name(self, name):
        """获取指定名字的记录数量"""
        sql = "SELECT COUNT(*) FROM face_features WHERE name = %s"
        self.cursor.execute(sql, (name,))
        return self.cursor.fetchone()[0]

    def init_attendance_table(self):
        """初始化考勤表（如果不存在则创建）"""
        sql = """
        CREATE TABLE IF NOT EXISTS attendance_records (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            attendance_time DATETIME NOT NULL,
            date DATE NOT NULL,
            time TIME NOT NULL,
            INDEX idx_name (name),
            INDEX idx_date (date),
            INDEX idx_attendance_time (attendance_time)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
        self.cursor.execute(sql)
        self.conn.commit()

    def add_attendance(self, name):
        """
        添加考勤记录
        返回: (是否成功, 消息)
        """
        try:
            # 确保表存在
            self.init_attendance_table()
            
            # 获取当前时间
            now = datetime.now()
            attendance_time = now.strftime("%Y-%m-%d %H:%M:%S")
            date = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            
            # 检查今天是否已经记录过考勤（防止重复记录）
            # 可以设置时间间隔，比如5分钟内不重复记录
            sql_check = """
            SELECT id FROM attendance_records 
            WHERE name = %s AND date = %s 
            AND TIMESTAMPDIFF(MINUTE, attendance_time, %s) < 5
            ORDER BY attendance_time DESC LIMIT 1
            """
            self.cursor.execute(sql_check, (name, date, attendance_time))
            if self.cursor.fetchone():
                return False, "已在5分钟内记录过考勤"
            
            # 插入考勤记录
            sql = """
            INSERT INTO attendance_records (name, attendance_time, date, time) 
            VALUES (%s, %s, %s, %s)
            """
            self.cursor.execute(sql, (name, attendance_time, date, time_str))
            self.conn.commit()
            return True, f"考勤记录成功: {name} - {attendance_time}"
            
        except Exception as e:
            self.conn.rollback()
            return False, f"考勤记录失败: {str(e)}"

    def get_attendance_records(self, name=None, date=None, limit=100):
        """
        获取考勤记录
        name: 姓名过滤（可选）
        date: 日期过滤，格式 'YYYY-MM-DD'（可选）
        limit: 返回记录数限制
        返回: 记录列表 [(id, name, attendance_time, date, time), ...]
        """
        try:
            self.init_attendance_table()
            
            sql = "SELECT id, name, attendance_time, date, time FROM attendance_records WHERE 1=1"
            params = []
            
            if name:
                sql += " AND name = %s"
                params.append(name)
            
            if date:
                sql += " AND date = %s"
                params.append(date)
            
            sql += " ORDER BY attendance_time DESC LIMIT %s"
            params.append(limit)
            
            self.cursor.execute(sql, params)
            return self.cursor.fetchall()
            
        except Exception as e:
            print(f"获取考勤记录失败: {e}")
            return []

    def get_attendance_statistics(self, name=None, start_date=None, end_date=None):
        """
        获取考勤统计信息
        返回: {'total_count': 总次数, 'by_date': {日期: 次数}}
        """
        try:
            self.init_attendance_table()
            
            sql = "SELECT date, COUNT(*) as count FROM attendance_records WHERE 1=1"
            params = []
            
            if name:
                sql += " AND name = %s"
                params.append(name)
            
            if start_date:
                sql += " AND date >= %s"
                params.append(start_date)
            
            if end_date:
                sql += " AND date <= %s"
                params.append(end_date)
            
            sql += " GROUP BY date ORDER BY date DESC"
            
            self.cursor.execute(sql, params)
            rows = self.cursor.fetchall()
            
            total_count = sum(row[1] for row in rows)
            by_date = {row[0]: row[1] for row in rows}
            
            return {
                'total_count': total_count,
                'by_date': by_date
            }
            
        except Exception as e:
            print(f"获取考勤统计失败: {e}")
            return {'total_count': 0, 'by_date': {}}

    def get_attendance_by_id(self, record_id):
        """
        根据ID获取单条考勤记录
        返回: (id, name, attendance_time, date, time) 或 None
        """
        try:
            self.init_attendance_table()
            sql = "SELECT id, name, attendance_time, date, time FROM attendance_records WHERE id = %s"
            self.cursor.execute(sql, (record_id,))
            result = self.cursor.fetchone()
            return result
        except Exception as e:
            print(f"获取考勤记录失败: {e}")
            return None

    def update_attendance(self, record_id, name=None, attendance_time=None):
        """
        更新考勤记录
        record_id: 记录ID
        name: 新的姓名（可选）
        attendance_time: 新的时间，格式 'YYYY-MM-DD HH:MM:SS'（可选）
        返回: (是否成功, 消息)
        """
        try:
            self.init_attendance_table()
            
            # 如果提供了新时间，需要解析日期和时间
            if attendance_time:
                dt = datetime.strptime(attendance_time, "%Y-%m-%d %H:%M:%S")
                date = dt.strftime("%Y-%m-%d")
                time_str = dt.strftime("%H:%M:%S")
            
            # 构建更新SQL
            updates = []
            params = []
            
            if name:
                updates.append("name = %s")
                params.append(name)
            
            if attendance_time:
                updates.append("attendance_time = %s")
                params.append(attendance_time)
                updates.append("date = %s")
                params.append(date)
                updates.append("time = %s")
                params.append(time_str)
            
            if not updates:
                return False, "没有提供需要更新的字段"
            
            sql = f"UPDATE attendance_records SET {', '.join(updates)} WHERE id = %s"
            params.append(record_id)
            
            self.cursor.execute(sql, params)
            self.conn.commit()
            
            if self.cursor.rowcount > 0:
                return True, "更新成功"
            else:
                return False, "未找到要更新的记录"
                
        except Exception as e:
            self.conn.rollback()
            return False, f"更新失败: {str(e)}"

    def delete_attendance(self, record_id):
        """
        删除考勤记录
        record_id: 记录ID
        返回: (是否成功, 消息)
        """
        try:
            self.init_attendance_table()
            sql = "DELETE FROM attendance_records WHERE id = %s"
            self.cursor.execute(sql, (record_id,))
            deleted_count = self.cursor.rowcount
            self.conn.commit()
            
            if deleted_count > 0:
                return True, f"成功删除 {deleted_count} 条记录"
            else:
                return False, "未找到要删除的记录"
                
        except Exception as e:
            self.conn.rollback()
            return False, f"删除失败: {str(e)}"

    def add_attendance_manual(self, name, attendance_time):
        """
        手动添加考勤记录（不进行重复检查）
        name: 姓名
        attendance_time: 时间，格式 'YYYY-MM-DD HH:MM:SS'
        返回: (是否成功, 消息)
        """
        try:
            self.init_attendance_table()
            
            # 解析时间
            dt = datetime.strptime(attendance_time, "%Y-%m-%d %H:%M:%S")
            date = dt.strftime("%Y-%m-%d")
            time_str = dt.strftime("%H:%M:%S")
            
            # 插入记录
            sql = """
            INSERT INTO attendance_records (name, attendance_time, date, time) 
            VALUES (%s, %s, %s, %s)
            """
            self.cursor.execute(sql, (name, attendance_time, date, time_str))
            self.conn.commit()
            return True, f"考勤记录添加成功: {name} - {attendance_time}"
            
        except Exception as e:
            self.conn.rollback()
            return False, f"添加失败: {str(e)}"

    def close(self):
        self.cursor.close()
        self.conn.close()
