from fastapi.staticfiles import StaticFiles
import os
import shutil
import uuid
import textwrap
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from faster_whisper import WhisperModel
from dotenv import load_dotenv
import torch
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import cv2
import numpy as np
import tempfile
import scipy.io.wavfile as wavfile

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

OPENROUTER_KEY = os.getenv("OPENAI_API_KEY")

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Free AI Backend - Enhanced")

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Whisper with word timestamps for speaker detection
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

# TTS Models cache
tts_models = {}

SUPPORTED_LANGS = ["ru", "kz", "en", "tr", "zh"]

# TTS Language configs: (silero_lang, model_id, available_speakers)
TTS_CONFIGS = {
    "ru": {"lang": "ru", "model_id": "v3_1_ru", "speakers": ["aidar", "baya", "kseniya", "xenia", "eugene", "random"]},
    "en": {"lang": "en", "model_id": "v3_en", "speakers": ["en_0", "en_1", "en_2", "en_3", "en_4", "random"]},
    "de": {"lang": "de", "model_id": "v3_de", "speakers": ["bernd_ungerer", "eva_k", "friedrich", "karlsson", "random"]},
    "es": {"lang": "es", "model_id": "v3_es", "speakers": ["es_0", "es_1", "es_2", "random"]},
    "fr": {"lang": "fr", "model_id": "v3_fr", "speakers": ["fr_0", "fr_1", "fr_2", "fr_3", "fr_4", "random"]},
    # Для языков без Silero модели - используем английский
    "kz": {"lang": "en", "model_id": "v3_en", "speakers": ["en_0", "en_1", "random"], "note": "Kazakh TTS not available, using EN"},
    "tr": {"lang": "en", "model_id": "v3_en", "speakers": ["en_0", "en_1", "random"], "note": "Turkish TTS not available, using EN"},
    "zh": {"lang": "en", "model_id": "v3_en", "speakers": ["en_0", "en_1", "random"], "note": "Chinese TTS not available, using EN"},
}

def get_tts_model(lang: str):
    """Загружает и кэширует TTS модель для языка"""
    config = TTS_CONFIGS.get(lang, TTS_CONFIGS["en"])
    model_key = config["model_id"]
    
    if model_key not in tts_models:
        torch.set_num_threads(1)
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language=config["lang"],
            speaker=model_key
        )
        tts_models[model_key] = model
    
    return tts_models[model_key], config

# ============== OCR HELPERS ==============

def cv2_imread_unicode(path):
    with open(path, "rb") as f:
        data = f.read()
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

def preprocess_for_tesseract(path: str) -> str:
    img = cv2_imread_unicode(path)
    if img is None:
        raise FileNotFoundError(f"image not found: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}_pre.png")
    cv2.imwrite(tmp_path, th)
    return tmp_path

def _safe_float_conf(conf_str):
    try:
        return float(conf_str)
    except:
        return 0.0

def _normalize_text(text: str) -> str:
    text = text.replace("|", " ")
    text = text.replace("lam ", "I am ")
    return " ".join(text.split())

def ocr_image(path):
    pre = preprocess_for_tesseract(path)
    config = "--psm 6 --oem 3"
    try:
        text = pytesseract.image_to_string(pre, lang="eng+rus", config=config)
    except:
        text = pytesseract.image_to_string(pre, lang="eng", config=config)
    text = _normalize_text(text or "")
    try:
        os.remove(pre)
    except:
        pass
    return text.strip()

def ocr_lines(path):
    config = "--psm 6 --oem 3"
    try:
        data = pytesseract.image_to_data(path, lang="eng+rus", config=config, output_type=pytesseract.Output.DICT)
    except:
        return []
    
    texts = data.get("text", [])
    if not texts:
        return []
    
    lines = {}
    n = len(texts)
    
    for i in range(n):
        level = data.get("level", [None]*n)[i]
        if level != 5:
            continue
        txt = (data.get("text", [])[i] or "").strip().replace("|", "").strip()
        if not txt:
            continue
        
        line_id = data.get("line_num", [i+1]*n)[i]
        try:
            x, y, w, h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
        except:
            continue
        
        if line_id not in lines:
            lines[line_id] = {"text": txt, "bbox": [x, y, w, h]}
        else:
            lines[line_id]["text"] += " " + txt
            lx, ly, lw, lh = lines[line_id]["bbox"]
            new_x, new_y = min(lx, x), min(ly, y)
            new_x2, new_y2 = max(lx + lw, x + w), max(ly + lh, y + h)
            lines[line_id]["bbox"] = [new_x, new_y, new_x2 - new_x, new_y2 - new_y]
    
    result = list(lines.values())
    result.sort(key=lambda b: b["bbox"][1])
    return result

# ============== SPEAKER DIARIZATION ==============

def transcribe_with_speakers(path: str):
    """
    Транскрибирует аудио/видео с определением временных меток и псевдо-спикеров.
    Использует VAD и паузы для разделения на сегменты речи.
    """
    # Убираем VAD фильтр чтобы захватить ВСЁ аудио, включая музыку с вокалом
    segments, info = whisper_model.transcribe(
        str(path),
        word_timestamps=True,
        vad_filter=False,  # Отключаем VAD для полной транскрипции
        condition_on_previous_text=True,  # Улучшает контекст
        no_speech_threshold=0.5,  # Снижаем порог "не речь"
        compression_ratio_threshold=2.4,  # Более мягкий фильтр
    )
    
    result_segments = []
    full_text = []
    
    # Простая эвристика для определения спикеров по паузам
    current_speaker = 1
    last_end = 0.0
    speaker_change_threshold = 2.0  # секунды паузы для смены спикера
    
    for seg in segments:
        # Если большая пауза - возможно другой спикер
        if seg.start - last_end > speaker_change_threshold and last_end > 0:
            current_speaker = 3 - current_speaker  # переключаем между 1 и 2
        
        segment_data = {
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
            "speaker": f"Speaker {current_speaker}",
            "words": []
        }
        
        # Добавляем слова с временными метками
        if seg.words:
            for word in seg.words:
                segment_data["words"].append({
                    "word": word.word,
                    "start": round(word.start, 2),
                    "end": round(word.end, 2),
                    "confidence": round(word.probability, 2) if hasattr(word, 'probability') else None
                })
        
        result_segments.append(segment_data)
        full_text.append(seg.text)
        last_end = seg.end
    
    return {
        "full_text": " ".join(full_text),
        "segments": result_segments,
        "language": info.language,
        "duration": round(info.duration, 2) if hasattr(info, 'duration') else None
    }

# ============== IMAGE TEXT REPLACEMENT ==============

def draw_text_in_bbox(draw, bbox, text, font_path=None, fill=(0,0,0)):
    x, y, w, h = bbox
    try:
        base_font_size = max(10, int(h * 0.9))
        font = ImageFont.truetype(font_path or "arial.ttf", base_font_size)
    except:
        font = ImageFont.load_default()
        base_font_size = 12

    def measure_text(font, s):
        bbox = font.getbbox(s)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    while True:
        w_M, _ = measure_text(font, "M")
        avg_char_width = w_M if w_M > 0 else 7
        max_chars_per_line = max(1, int(w / avg_char_width))
        wrapped = textwrap.fill(text, width=max_chars_per_line)
        lines = wrapped.splitlines()
        total_h = sum(measure_text(font, line)[1] for line in lines) + (len(lines) - 1) * 2
        if total_h <= h or base_font_size <= 8:
            break
        base_font_size = max(8, base_font_size - 1)
        try:
            font = ImageFont.truetype(font_path or "arial.ttf", base_font_size)
        except:
            font = ImageFont.load_default()
            break

    cur_y = y + max(0, (h - total_h) // 2)
    for line in lines:
        lw, lh = measure_text(font, line)
        draw.text((x, cur_y), line, font=font, fill=fill)
        cur_y += lh + 2

def replace_text_on_image(image_path, lines, translated_lines):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    try:
        ImageFont.truetype("arial.ttf", 14)
        font_path = "arial.ttf"
    except:
        font_path = None

    for item, new_text in zip(lines, translated_lines):
        x, y, w, h = item["bbox"]
        pad_x, pad_y = max(2, int(w * 0.03)), max(1, int(h * 0.08))
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(img.width, x + w + pad_x), min(img.height, y + h + pad_y)
        draw.rectangle([x0, y0, x1, y1], fill="white")
        inner_bbox = (x0 + 2, y0 + 1, x1 - x0 - 4, y1 - y0 - 2)
        draw_text_in_bbox(draw, inner_bbox, new_text, font_path=font_path, fill=(0,0,0))

    out_path = OUTPUT_DIR / f"{uuid.uuid4()}_translated_image.jpg"
    img.save(out_path, quality=95)
    return out_path

# ============== OPENROUTER ==============

def openrouter_chat(model, messages):
    import requests
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages},
            timeout=40
        )
    except Exception as e:
        return f"[OpenRouter request failed] {e}"
    
    if r.status_code != 200:
        return f"[OpenRouter HTTP {r.status_code}] {r.text}"
    
    try:
        data = r.json()
        return str(data["choices"][0]["message"]["content"])
    except:
        return f"[Malformed Response] {r.text}"

def translate_lines(lines, target_lang):
    translated = []
    for item in lines:
        out = openrouter_chat("openai/gpt-3.5-turbo", [
            {"role": "system", "content": f"Translate into {target_lang}. Return ONLY translation, no comments."},
            {"role": "user", "content": item["text"]}
        ])
        translated.append(out.strip())
    return translated

# ============== ENDPOINTS ==============

@app.get("/tts/voices")
async def get_voices():
    """Возвращает доступные голоса для каждого языка"""
    return {lang: {"speakers": cfg["speakers"], "note": cfg.get("note")} for lang, cfg in TTS_CONFIGS.items()}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{file_id}_{file.filename}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"status": "ok", "filename": dest.name}

@app.post("/process")
async def process(
    filename: str = Form(...),
    translate_to: str = Form("ru"),
    style: str = Form("normal"),
    replace_image_text: bool = Form(True)
):
    if translate_to not in SUPPORTED_LANGS:
        return JSONResponse(status_code=400, content={"error": f"Unsupported language: {translate_to}"})

    path = UPLOAD_DIR / filename
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "file not found"})

    ext = path.suffix.lower()

    # AUDIO/VIDEO - с определением спикеров и временных меток
    if ext in [".mp3", ".wav", ".m4a", ".mp4", ".webm", ".aac", ".ogg"]:
        source = "audio"
        transcription = transcribe_with_speakers(str(path))
        text = transcription["full_text"]
        
        translated_image_url = None
        ocr_lines_result = None

    # IMAGE
    elif ext in [".jpg", ".jpeg", ".png", ".webp"]:
        source = "image"
        text = ocr_image(str(path))
        ocr_lines_result = ocr_lines(str(path))
        translated_lines_list = translate_lines(ocr_lines_result, translate_to)
        
        translated_image_url = None
        if replace_image_text and ocr_lines_result:
            out_path = replace_text_on_image(str(path), ocr_lines_result, translated_lines_list)
            translated_image_url = f"/outputs/{Path(out_path).name}"
        
        transcription = None
    else:
        return JSONResponse(status_code=400, content={"error": "unsupported file"})

    # Translation - разбиваем длинный текст на части
    def translate_long_text(text, target_lang, max_chunk=2000):
        if len(text) <= max_chunk:
            return openrouter_chat("openai/gpt-3.5-turbo", [
                {"role": "system", "content": f"Translate the following text completely into {target_lang}. Return ONLY the full translated text, nothing else."},
                {"role": "user", "content": text}
            ])
        
        # Разбиваем по предложениям
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""
        
        for s in sentences:
            if len(current) + len(s) < max_chunk:
                current += s + " "
            else:
                if current:
                    chunks.append(current.strip())
                current = s + " "
        if current:
            chunks.append(current.strip())
        
        translated_parts = []
        for chunk in chunks:
            part = openrouter_chat("openai/gpt-3.5-turbo", [
                {"role": "system", "content": f"Translate completely into {target_lang}. Return ONLY the translation."},
                {"role": "user", "content": chunk}
            ])
            translated_parts.append(part)
        
        return " ".join(translated_parts)
    
    translation = translate_long_text(text, translate_to)

    # Paraphrase
    paraphrase = openrouter_chat("openai/gpt-3.5-turbo", [
        {"role": "system", "content": f"Paraphrase in style {style} ({translate_to})"},
        {"role": "user", "content": translation}
    ])

    # Save files
    out1 = OUTPUT_DIR / f"{path.stem}_translation.txt"
    out1.write_text(translation, encoding="utf-8")
    out2 = OUTPUT_DIR / f"{path.stem}_paraphrase_{style}.txt"
    out2.write_text(paraphrase, encoding="utf-8")

    response = {
        "source": source,
        "original_text": text,
        "translation": translation,
        "paraphrase": paraphrase,
        "files": {"translation": f"/outputs/{out1.name}", "paraphrase": f"/outputs/{out2.name}"}
    }

    if source == "audio" and transcription:
        response["transcription"] = transcription  # Включает segments с спикерами и временными метками!
    
    if source == "image":
        response["ocr_lines"] = ocr_lines_result
        response["translated_lines"] = translated_lines_list
        if translated_image_url:
            response["translated_image_url"] = translated_image_url

    return JSONResponse(content=jsonable_encoder(response))

@app.post("/tts")
async def tts(
    text: str = Form(...),
    lang: str = Form("en"),
    speaker: str = Form("random")
):
    """
    Мультиязычный TTS с выбором голоса.
    Поддерживает: ru, en, de, es, fr (нативно), kz/tr/zh (через en)
    """
    try:
        model, config = get_tts_model(lang)
        
        # Валидация спикера
        available = config["speakers"]
        if speaker not in available:
            speaker = "random" if "random" in available else available[0]
        
        # Генерация
        audio = model.apply_tts(text=text[:1000], speaker=speaker, sample_rate=48000)
        
        out = OUTPUT_DIR / f"tts_{uuid.uuid4()}.wav"
        wavfile.write(str(out), 48000, (audio.numpy() * 32767).astype(np.int16))
        
        return FileResponse(out, media_type="audio/wav", headers={
            "X-TTS-Language": config["lang"],
            "X-TTS-Speaker": speaker,
            "X-TTS-Note": config.get("note", "")
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/tts/batch")
async def tts_batch(
    texts: str = Form(...),  # JSON array of texts
    lang: str = Form("en"),
    speaker: str = Form("random")
):
    """Генерация нескольких аудио файлов за один запрос"""
    import json
    try:
        text_list = json.loads(texts)
        model, config = get_tts_model(lang)
        
        available = config["speakers"]
        if speaker not in available:
            speaker = "random" if "random" in available else available[0]
        
        files = []
        for i, text in enumerate(text_list[:10]):  # Max 10
            audio = model.apply_tts(text=text[:500], speaker=speaker, sample_rate=48000)
            out = OUTPUT_DIR / f"tts_batch_{uuid.uuid4()}_{i}.wav"
            wavfile.write(str(out), 48000, (audio.numpy() * 32767).astype(np.int16))
            files.append(f"/outputs/{out.name}")
        
        return {"files": files, "speaker": speaker, "lang": config["lang"]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
