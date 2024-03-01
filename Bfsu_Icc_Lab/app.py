from flask import Flask, request, redirect, render_template
import mysql.connector
import os
import webbrowser

app = Flask(__name__)

# 数据库配置
db_config = {
    'host': 'localhost',  # 数据库地址
    'user': 'beiwaiuser',  # 数据库用户名
    'password': '1561897423',  # 数据库密码
    'database': 'beiwaiicclib'  # 数据库名称
    
    #'host': 'localhost',
    #'user': os.getenv('DB_USER'),
    #'password': os.getenv('DB_PASS'),
    #'database': 'beiwaiicclib'
}
@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback_content = request.form['feedback']
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        
        cursor.execute("""
            INSERT INTO feedback (feedback_content)
            VALUES (%s)
        """, (feedback_content,))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        return redirect('/thankyou_feedback')
    except Exception as e:
        print(e)
        return f"Error occurred: {str(e)}", 500
        
@app.route('/submit_donation', methods=['POST'])
def submit_donation():
    try:
        donorName = request.form['donorName']
        bookName = request.form['bookName']
        studentId = request.form['studentId']
        email = request.form['email']
        phoneNumber = request.form['phoneNumber']
        donationDate = request.form['donationDate']

        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO donations (donorName, bookName, studentId, email, phoneNumber, donationDate)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (donorName, bookName, studentId, email, phoneNumber, donationDate))
        connection.commit()
        cursor.close()
        connection.close()
        return redirect('/thankyou_donation')
    except Exception as e:
        print(e)
        return f"Error occurred: {str(e)}", 500

@app.route('/submit_form', methods=['POST'])
def submit_borrow():
    try:
        book_name = request.form['book_name']
        student_id = request.form['student_id']
        full_name = request.form['full_name']
        borrow_date = request.form['borrow_date']
        email = request.form['email']
        phone_number = request.form['phone_number']
        wechat_id = request.form['wechat_id']

        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO borrows (book_name, student_id, full_name, borrow_date, email, phone_number, wechat_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (book_name, student_id, full_name, borrow_date, email, phone_number, wechat_id))
        connection.commit()
        cursor.close()
        connection.close()
        return redirect('/thankyou_borrow')
    except Exception as e:
        return f"Error occurred: {str(e)}", 500

@app.route('/submit_return', methods=['POST'])
def submit_return():
    try:
        returner_name = request.form['returner_name']
        book_name = request.form['book_name']
        student_id = request.form['student_id']
        return_date = request.form['return_date']
        email = request.form['email']

        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO returns (returner_name, book_name, student_id, return_date, email)
            VALUES (%s, %s, %s, %s, %s)
        """, (returner_name, book_name, student_id, return_date, email))
        connection.commit()
        cursor.close()
        connection.close()
        #webbrowser.open('file:///' + os.path.realpath('Submitted_Form/Return_Submit.html'))
        #return ('',204)
        return redirect('/thankyou_return')
    except Exception as e:
        return f"Error occurred: {str(e)}", 500
        
@app.route('/thankyou_feedback')
def thank_you_feedback():
    html_content = """
<!DOCTYPE html>
<html lang="zh-cn">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>感谢您的意见 - 北京外国语大学国际课程中心图书室</title>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;700&family=Roboto:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body, html {
            height: 100%;
            margin: 0;
            font-family: 'Noto Sans SC', sans-serif;
            background-color: #f5f5f5;
            display: flex;
            flex-direction: column;
        }

        .container {
            max-width: 800px;
            margin: auto;
            padding: 40px;
            background-color: #fff;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin-top: 50px;
            border-radius: 5px;
        }

        h1 {
            font-size: 24px;
            color: #333;
        }

        p {
            font-size: 16px;
            color: #666;
            line-height: 1.6;
        }

        .thank-you-note {
            margin-top: 20px;
            padding: 20px;
            background-color: #e7f4e4;
            color: #3c763d;
            border-left: 5px solid #3c763d;
        }

        footer {
            background-color: #333;
            color: white;
            text-align: center;
            padding: 10px 0;
            position: absolute;
            bottom: 0;
            width: 100%;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>感谢您提供宝贵的意见！</h1>
        <p>您的反馈对我们至关重要。它帮助我们持续改进，并确保我们能够及时反思。</p>
        <div class="thank-you-note">
            <p>我们将仔细审阅您的意见，并尽可能采取适当的行动。如果您留下了联系方式，我们可能会与您联系以获取更多信息。</p>
        </div>
    </div>

    <footer>
        <p>&copy; 2023 北京外国语大学国际课程中心图书室</p>
    </footer>
</body>
</html>
    """
    return html_content
    
@app.route('/thankyou_donation')
def thank_you_donation():
    html_content = """
<!DOCTYPE html>
<html lang="zh">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>感谢捐书</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
        }

        .container {
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background-color: #fff;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }

        .header {
            text-align: center;
            padding-bottom: 20px;
        }

        .header img {
            max-width: 250px;
        }

        .content {
            text-align: center;
        }

        .content h1 {
            color: #333;
        }

        .content p {
            font-size: 16px;
            color: #666;
            line-height: 1.5;
        }

        .footer {
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }

        .footer p {
            font-size: 14px;
            color: #999;
        }
    </style>
</head>

<body>
    <div class="container">
        <div class="header">
            <img src="ICC_Logo.png" alt="ICC Logo">
        </div>
        <div class="content">
            <h1>感谢您的捐赠！</h1>
            <p>我们真诚地感谢您捐赠的书籍。您的慷慨将有助于我们的图书馆继续为社区提供优质的阅读材料。再次感谢您的支持和贡献！</p>
        </div>
        <div class="footer">
            <p>ICC 图书馆团队</p>
        </div>
    </div>
</body>

</html>
"""
    return html_content

@app.route('/thankyou_borrow')
def thank_you_borrow():
    html_content = """
<!DOCTYPE html>
<html lang="zh">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>感谢借书</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
        }

        .container {
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background-color: #fff;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }

        .header {
            text-align: center;
            padding-bottom: 20px;
        }

        .header img {
            max-width: 250px;
        }

        .content {
            text-align: center;
        }

        .content h1 {
            color: #333;
        }

        .content p {
            font-size: 16px;
            color: #666;
            line-height: 1.5;
        }

        .footer {
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }

        .footer p {
            font-size: 14px;
            color: #999;
        }
    </style>
</head>

<body>
    <div class="container">
        <div class="header">
            <img src="ICC_Logo.png" alt="ICC Logo">
        </div>
        <div class="content">
            <h1>感谢您选择我们的图书！</h1>
            <p>我们深知每一本书都有其特殊的价值和意义。希望您在阅读中找到乐趣，收获知识，启发思考。请在一个月之内在还书界面还书。再次感谢您的支持！</p>
        </div>
        <div class="footer">
            <p>ICC 图书馆团队</p>
        </div>
    </div>
</body>

</html>
"""
    return html_content

@app.route('/thankyou_return')
def thank_you_return():
    html_content = """
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>感谢还书</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
        }

        .container {
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background-color: #fff;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }

        .header {
            text-align: center;
            padding-bottom: 20px;
        }

        .header img {
            max-width: 250px;
        }

        .content {
            text-align: center;
        }

        .content h1 {
            color: #333;
        }

        .content p {
            font-size: 16px;
            color: #666;
            line-height: 1.5;
        }

        .footer {
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }

        .footer p {
            font-size: 14px;
            color: #999;
        }
    </style>
</head>

<body>
    <div class="container">
        <div class="header">
            <img src="ICC_Logo.png" alt="ICC Logo">
        </div>
        <div class="content">
            <h1>感谢您及时归还图书！</h1>
            <p>我们深知每一本书都有其特殊的价值和意义。您的责任心和合作精神使得其他读者也能够及时享受到这本书的内容。再次感谢您的支持！</p>
        </div>
        <div class="footer">
            <p>ICC 图书馆团队</p>
        </div>
    </div>
</body>

</html>
"""
    return html_content

if __name__ == '__main__':
    app.run(port=8000)
