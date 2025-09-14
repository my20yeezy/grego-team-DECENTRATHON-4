import os
import streamlit as st
from llama_cpp import Llama
import time
import json
from PIL import Image
import pypdfium2 as pdfium
import tempfile
import numpy as np

# если запуск происходит локально на слабом пк
os.environ['TORCH_DEVICE'] = 'cpu'
os.environ['DETECTOR_BATCH_SIZE'] = '1'
os.environ['RECOGNITION_BATCH_SIZE'] = '8'

# Инициализация модели (один раз при запуске)
@st.cache_resource
def load_llama_model():
    st.info("🔄 Будим модель Григория... Это может занять несколько минут")
    try:
        llm = Llama(
            model_path="gregory.gguf",
            n_ctx=8000,
            n_threads=8,
            n_gpu_layers=35,
            verbose=False
        )
        st.success("✅ Григорий успешно проснулся!")
        return llm
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}")
        return None

@st.cache_resource
def load_surya_models():
    st.info("🔄 Загрузка моделей Surya OCR...")
    try:
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
        
        foundation_predictor = FoundationPredictor()
        rec_predictor = RecognitionPredictor(foundation_predictor)
        det_predictor = DetectionPredictor()
        
        st.success("✅ Модели Surya OCR успешно загружены!")
        return rec_predictor, det_predictor
    except Exception as e:
        st.error(f"❌ Ошибка загрузки моделей OCR: {e}")
        return None, None

def ocr_file(file_path: str, rec_predictor, det_predictor, max_pages: int = 10) -> str:
    """Распознает текст из PDF или изображения"""
    
    result_text = ""
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Обработка PDF
    if file_path.lower().endswith('.pdf'):
        try:
            pdf = pdfium.PdfDocument(file_path)
            total_pages = len(pdf)
            pages_to_process = min(max_pages, total_pages)
            
            for page_num in range(pages_to_process):
                status_text.text(f"📄 Обработка страницы {page_num + 1}/{pages_to_process}")
                progress_bar.progress((page_num + 1) / pages_to_process)
                
                result_text += f"\n--- Страница {page_num + 1} ---\n"
                
                page = pdf.get_page(page_num)
                bitmap = page.render(scale=0.8)
                pil_image = bitmap.to_pil()
                
                if pil_image.width > 1000:
                    new_width = 800
                    new_height = int(800 * pil_image.height / pil_image.width)
                    pil_image = pil_image.resize((new_width, new_height))
                
                predictions = rec_predictor([pil_image], det_predictor=det_predictor)
                
                for line in predictions[0].text_lines:
                    result_text += line.text + "\n"
                
                result_text += f"Всего строк: {len(predictions[0].text_lines)}\n"
                    
        except Exception as e:
            result_text += f"Ошибка при обработке PDF: {e}\n"

    # Обработка изображений
    elif file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
        try:
            status_text.text("🖼️ Обработка изображения...")
            image = Image.open(file_path)
            result_text += "\n--- Изображение ---\n"
            
            if image.width > 1200:
                new_width = 1000
                new_height = int(1000 * image.height / image.width)
                image = image.resize((new_width, new_height))
            
            predictions = rec_predictor([image], det_predictor=det_predictor)
            
            for line in predictions[0].text_lines:
                result_text += line.text + "\n"
            
            result_text += f"Всего строк: {len(predictions[0].text_lines)}\n"
            progress_bar.progress(1.0)
            
        except Exception as e:
            result_text += f"Ошибка при обработке изображения: {e}\n"
    
    else:
        result_text = "Неподдерживаемый формат файла"

    status_text.empty()
    progress_bar.empty()
    return result_text

def analyze_document_with_llm(ocr_text, llm):
    """Анализирует документ Гриша"""
    
    prompt = f"""Ты — эксперт по анализу договоров и банковских документов. 
OCR может содержать ошибки, документ может быть на русском, английском или казахском языке. 
Твоя задача — исправить OCR-ошибки и вернуть необходимые нам структурированные данные в формате JSON.

ТРЕБОВАНИЯ:
1. Возвращай результат только в формате JSON. Начни с '{{' и закончи '}}'. Никакого другого текста.
2. JSON должен строго соответствовать схеме ниже. Не добавляй лишние поля.
3. Если данные повреждены — исправь (например, "20.13.2022" → "20.12.2022", "ОО0" → "ООО").
4. Если реквизит отсутствует — укажи null.
5. Игнорируй OCR шумы, артефакты и печати внутри текста.
6. Никакого текста кроме JSON, у тебя стоит полный ЗАПРЕТ на то что не JSON

JSON-СХЕМА:
{{
  "type": "object",
  "properties": {{
    "document_type": {{"type": "string", "description": "Тип документа (чек, договор, выписка, платежное поручение и т.п.)"}},
    "contract_number": {{"type": ["string", "null"], "description": "№ контракта или документа"}},
    "contract_date": {{"type": ["string", "null"], "description": "Дата заключения контракта или дата операции (ДД.ММ.ГГГГ)"}},
    "contract_end_date": {{"type": ["string", "null"], "description": "Дата окончания действия договора (ДД.ММ.ГГГГ) или null"}},
    "contractor_name": {{"type": ["string", "null"], "description": "Наименование контрагента или клиента"}},
    "account_number": {{"type": ["string", "null"], "description": "Номер счета клиента"}},
    "client_name": {{"type": ["string", "null"], "description": "ФИО клиента или название организации"}},
    "client_inn": {{"type": ["string", "null"], "description": "ИНН клиента"}},
    "transaction_date": {{"type": ["string", "null"], "description": "Дата операции (ДД.ММ.ГГГГ)"}},
    "transaction_amount": {{"type": ["string", "null"], "description": "Сумма операции"}},
    "recipient_account": {{"type": ["string", "null"], "description": "Номер счета получателя"}},
    "bik": {{"type": ["string", "null"], "description": "БИК банка"}},
    "purpose": {{"type": ["string", "null"], "description": "Назначение платежа или описание операции"}},
  }},
  "required": ["document_type"]
}}

ДОКУМЕНТ (OCR):
{ocr_text}
ОБЯЗАТЕЛЬНО ВЕРНИ ТОЛЬКО JSON и ничего лишнего
"""

    with st.spinner("🤖 Гриша изучает документ... Это может занять несколько минут"):
        response = llm(
            prompt,
            max_tokens=1024,  
            temperature=0.3,   
            top_p=0.9,
            echo=False
        )
    
    return response["choices"][0]["text"]

def main():
    st.set_page_config(
        page_title="Умный OCR от команды GREGO",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 GREGO OCR")
    st.markdown("Загрузите PDF или изображение для получения структурированного JSON")
    
    # Загрузка моделей
    llm = load_llama_model()
    rec_predictor, det_predictor = load_surya_models()
    
    if llm is None or rec_predictor is None:
        st.error("Не удалось загрузить необходимые модели. Проверьте установку зависимостей.")
        return
    
    # Загрузка файла
    uploaded_file = st.file_uploader(
        "Выберите файл (PDF, JPG, PNG)",
        type=['pdf', 'jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Поддерживаются PDF и изображения"
    )
    
    if uploaded_file is not None:
        # Просто работаем с загруженным файлом как есть
        
        # Показ предпросмотра для изображений
        if uploaded_file.type.startswith('image/'):
            st.image(uploaded_file, caption="Загруженное изображение", use_column_width=True)
        
        # Кнопка обработки
        if st.button("🚀 Начать анализ", type="primary"):
            # Сохраняем файл куда-то 
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # OCR обработка - просто делаем без всяких экспандеров
            ocr_result = ocr_file(uploaded_file.name, rec_predictor, det_predictor)
            
            # Анализ с LLM
            llm_result = analyze_document_with_llm(ocr_result, llm)
            
            # Отображение результатов
            st.success("✅ Анализ завершен!")
            
            # Исходный OCR текст
            with st.expander("📄 Исходный OCR текст"):
                st.text(ocr_result)
            
            # Результат анализа
            st.subheader("📊 Результат анализа")
            
            # Просто выводим ответ LLM как есть
            st.code(llm_result, language='json')

if __name__ == "__main__":
    main()