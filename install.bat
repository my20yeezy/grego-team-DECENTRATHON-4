@echo off
echo Установка зависимостей GREGO OCR...

REM 
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Ошибка: Python не установлен или не добавлен в PATH.
    pause
    exit /b 1
)

REM 
pip install -r requirements.txt

echo Зависимости установлены глобально!
pause