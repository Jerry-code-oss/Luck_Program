import mysql.connector

# 数据库配置
db_config = {
    'host': 'localhost',
    'user': 'beiwaiuser',
    'password': '1561897423',
    'database': 'beiwaiicclib'
}

# 连接到数据库
connection = mysql.connector.connect(**db_config)
cursor = connection.cursor()

# 获取 borrows 表中的所有书
cursor.execute("SELECT book_name, student_id FROM borrows")
borrowed_books = cursor.fetchall()

# 获取 returns 表中的所有书
cursor.execute("SELECT book_name, student_id FROM returns")
returned_books = cursor.fetchall()

# 关闭数据库连接
cursor.close()
connection.close()

# 找出尚未归还的书
not_returned_books = [book for book in borrowed_books if book not in returned_books]

for book, student_id in not_returned_books:
    print(f"Book '{book}' borrowed by student {student_id} has not been returned yet.")
print(f"Check finished!")

