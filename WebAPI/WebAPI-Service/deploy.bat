@echo off
echo Deploying NLP API to IIS...

:: ایجاد دایرکتوری‌های لازم
mkdir C:\inetpub\wwwroot\NLP-API
mkdir C:\inetpub\wwwroot\NLP-API\logs
mkdir C:\inetpub\wwwroot\NLP-API\model_cache

:: کپی فایل‌ها
copy Hugging_Face_Transformer_IIS.py C:\inetpub\wwwroot\NLP-API\
copy web.config C:\inetpub\wwwroot\NLP-API\
copy requirements-prod.txt C:\inetpub\wwwroot\NLP-API\

:: نصب dependencies
cd C:\inetpub\wwwroot\NLP-API
pip install -r requirements-prod.txt

echo Deployment completed!
echo API will be available at: http://your-server-ip/
pause