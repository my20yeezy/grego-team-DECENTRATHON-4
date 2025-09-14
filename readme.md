# 📄 GREGO OCR - Умный анализатор документов

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red)
![OCR](https://img.shields.io/badge/OCR-Surya-green)
![LLM](https://img.shields.io/badge/LLM-Llama.cpp-orange)

Мощный инструмент для автоматического распознавания и анализа документов с использованием искусственного интеллекта.

## 🎯 Возможности

- 📄 Распознавание текста из PDF и изображений (JPG, PNG, BMP, TIFF)
- 🤖 Интеллектуальный анализ документов с помощью LLM модели Gregory
- 🏦 Поддержка банковских документов, договоров, чеков, выписок
- 🔄 Автоматическое исправление ошибок OCR
- 📊 Структурированный вывод в формате JSON
- ⚡ Быстрая обработка до 10 страниц

## ⚠️ Обязательные требования

### 1. Python 3.11
**Только эта версия!** Другие версии Python не поддерживаются.

**Скачать:** [https://www.python.org/downloads/](https://www.python.org/downloads/)

**Важно:** При установке обязательно отметьте галочку **"Add Python to PATH"**

### 2. Модель Gregory (требуется отдельная загрузка)
Модель не входит в репозиторий из-за большого размера.

**Скачать здесь:** [Google Drive](https://drive.google.com/drive/folders/1RJNEqtzYuH4085EHIpkM29IvEobWmJ0Z?usp=sharing)

Файл: `gregory.gguf` → положите в корневую папку проекта

### 3. Для пользователей Windows
**Требуется:** Visual C++ Build Tools

**Скачать:** [https://visualstudio.microsoft.com/ru/visual-cpp-build-tools/](https://visualstudio.microsoft.com/ru/visual-cpp-build-tools/)

При установке выберите:
- "C++ Build Tools"
- Компонент "MSVC v143 - VS 2022 C++ x64/x86 build tools"

## 🚀 Быстрая установка

### Windows
```bash
# 1. Установите Python 3.11 + MSVC Build Tools
# 2. Скачайте модель с Google Drive
# 3. Запустите:
install.bat
