#                                                                                  به نام خدا

# ===========================================================================
#                            اولین جلسه  داکر
# ===========================================================================

from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def hello():
    return f'سلام! من در Docker اجرا می‌شم! 🐳'

@app.route('/info')
def info():
    return {
        'python_version': os.environ.get('PYTHON_VERSION', '3.12.6'),
        'hostname': os.environ.get('HOSTNAME', 'unknown')
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


# ------------------------------------------------------------
#  برای اجرای کد دستور زیر را اجرا می کنیم
#  docker run -it -p 5000:5000 -v "/D:/Projects/Visual Code/Python/DeepSeekSamples/AI-Journey-with-DeepSeek/Docker/Session1":/app python:3.12.6-slim sh
# ------------------------------------------------------------    
