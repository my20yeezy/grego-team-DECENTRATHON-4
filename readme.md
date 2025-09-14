📄 GREGO OCR - Умный анализатор документов
https://img.shields.io/badge/Python-3.11%252B-blue
https://img.shields.io/badge/Streamlit-1.32.0-red
https://img.shields.io/badge/OCR-Surya-green
https://img.shields.io/badge/LLM-Llama.cpp-orange

⚠️ ВАЖНЫЕ ТРЕБОВАНИЯ ПЕРЕД УСТАНОВКОЙ
1. Обязательно установите Python 3.11
Скачайте с официального сайта: https://www.python.org/downloads/

ВНИМАНИЕ: Только версия 3.11! Другие версии не поддерживаются

При установке ОБЯЗАТЕЛЬНО поставьте галочку "Add Python to PATH"

2. Для Windows - установите Visual Studio Build Tools
Скачайте здесь: https://visualstudio.microsoft.com/ru/visual-cpp-build-tools/

Выберите "C++ Build Tools"

Включите компонент "MSVC v143 - VS 2022 C++ x64/x86 build tools"

Без этого установка не произойдет!

3. Скачайте LLM модель Gregory отдельно
Перейдите по ссылке: https://drive.google.com/drive/folders/1RJNEqtzYuH4085EHIpkM29IvEobWmJ0Z?usp=sharing

Скачайте файл gregory.gguf

Положите его в папку с проектом (рядом с main.py)

🚀 Быстрая установка
Windows:
bash
# 1. Установите Python 3.11 и MSVC как указано выше
# 2. Скачайте модель с Google Drive
# 3. Запустите:
install.bat
Linux/Mac:
bash
# 1. Убедитесь что Python 3.11 установлен
python --version  # должно быть 3.11.x
# 2. Скачайте модель с Google Drive  
# 3. Запустите:
chmod +x install.sh run.sh
./install.sh
🏃 Запуск
После установки:

bash
run.bat  # для Windows
./run.sh # для Linux/Mac
Приложение откроется в браузере по адресу: http://localhost:8501

📦 Что внутри
text
GREGO_OCR/
├── main.py                 # Главный скрипт
├── gregory.gguf          # ⚠️ СКАЧАЙТЕ ОТДЕЛЬНО С Google Drive!
├── requirements.txt       # Все зависимости
├── install.bat           # Автоустановщик
├── install.sh           # Автоустановщик
├── run.bat              # Автозапуск
├── run.sh              # Автозапуск
└── .venv/               # Виртуальное окружение
❌ Если не сделать обязательные шаги:
Без Python 3.11 - ничего не установится

Без MSVC на Windows - ошибка сборки llama-cpp

Без модели gregory.gguf - приложение не запустится

🔧 Проверка установки
Убедитесь что все установлено правильно:

bash
python --version  # Должно быть Python 3.11.x
⏰ Время установки
Первый запуск: 5-15 минут (скачивание моделей OCR)

Последующие запуски: 10-30 секунд

🆘 Помощь
Если возникают ошибки:

Проверьте что Python 3.11 установлен и добавлен в PATH

Убедитесь что скачали модель с Google Drive

Для Windows - проверьте установку MSVC Build Tools

Запомните: Python 3.11 + модель с Google Drive + MSVC (для Windows) = рабочее приложение