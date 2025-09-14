#!/bin/bash
echo "Установка зависимостей GREGO OCR..."
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "Зависимости установлены! Активируй виртуальное окружение: source .venv/bin/activate"