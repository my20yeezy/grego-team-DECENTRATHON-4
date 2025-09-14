@echo off
echo Установка зависимостей GREGO OCR...
python -m venv .venv
call .venv\.Scripts\activate.bat
pip install -r requirements.txt
echo Зависимости установлены! Активируй виртуальное окружение: .venv\Scripts\activate.bat
pause