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

# Укажи путь к tesseract.exe на твоей системе (оставь как есть, если всё верно)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

OPENROUTER_KEY = os.getenv("OPENAI_API_KEY")

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Free AI Backend")

# STATIC FILES — для отдачи изображений и txt
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Whisper (инициализация как у тебя была)
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

# ---------------------------
# Helpers: preprocessing + safe parsing
# ---------------------------

def cv2_imread_unicode(path):
    with open(path, "rb") as f:
        data = f.read()
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

def preprocess_for_tesseract(path: str) -> str:
    """
    Предобработка: читаем изображение, переводим в grayscale, легкий бoльшой контраст,
    но НЕ меняем размер — это важно, чтобы bbox соответствовали исходному размеру.
    Сохраняем во временный файл и возвращаем путь.
    """
    img = cv2_imread_unicode(path)
    if img is None:
        raise FileNotFoundError(f"image not found: {path}")

    # конвертация в серое
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # небольшая обработка: сглаживание, адаптивный порог
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15
    )

    # можно иногда инвертировать, если фон тёмный — но пока не трогаем автоматом

    tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}_pre.png")
    cv2.imwrite(tmp_path, th)
    return tmp_path

def _safe_float_conf(conf_str):
    """pytesseract иногда возвращает '-1' or '' — парсим безопасно"""
    try:
        return float(conf_str)
    except Exception:
        return 0.0

def _normalize_text(text: str) -> str:
    text = text.replace("|", " ")
    # common tesseract mistake fix
    text = text.replace("lam ", "I am ")
    return " ".join(text.split())

# ---------------------------
# OCR (text only)
# ---------------------------

def ocr_image(path):
    """
    Быстрый OCR (возвращает очищённый текст). Использует предобработку.
    Возвращает строку текста.
    """
    pre = preprocess_for_tesseract(path)
    # Use Page segmentation mode 6 (assume a single uniform block of text)
    config = "--psm 6 --oem 3"
    try:
        text = pytesseract.image_to_string(pre, lang="eng+rus", config=config)
    except Exception:
        # fallback to english-only if something goes wrong
        text = pytesseract.image_to_string(pre, lang="eng", config=config)
    text = _normalize_text(text or "")
    try:
        os.remove(pre)
    except Exception:
        pass
    return text.strip()


def ocr_with_boxes(path):
    """
    Возвращает список слов с bbox и confidence:
    [ {"text": "...", "bbox": [x,y,w,h], "confidence": 95.0}, ... ]
    Использует оригинальное изображение (чтобы bbox соответствовали размеру).
    """
    config = "--psm 6 --oem 3"
    try:
        data = pytesseract.image_to_data(
            path,
            lang="eng+rus",
            config=config,
            output_type=pytesseract.Output.DICT
        )
    except Exception:
        # если что-то пошло не так, возвращаем пустой список
        return []

    results = []
    texts = data.get("text", [])
    n = len(texts)

    for i in range(n):
        txt_raw = texts[i]
        if txt_raw is None:
            continue
        txt = str(txt_raw).strip()
        if txt == "":
            continue

        # безопасный парсинг числовых полей
        try:
            left = int(data.get("left", [0]*n)[i])
            top = int(data.get("top", [0]*n)[i])
            width = int(data.get("width", [0]*n)[i])
            height = int(data.get("height", [0]*n)[i])
        except Exception:
            # пропускаем некорректные записи
            continue

        conf = _safe_float_conf(data.get("conf", [0]*n)[i])

        # Небольшая нормализация
        txt = txt.replace("|", "").strip()
        if txt == "":
            continue

        results.append({
            "text": txt,
            "bbox": [left, top, width, height],
            "confidence": conf
        })

    return results

def ocr_lines(path):
    """
    Возвращает строки с bbox:
    [ { "text": "...", "bbox": [x,y,w,h] }, ... ]
    bbox соответствует исходному изображению.
    Сгруппировано по line_num, отсортировано сверху вниз.
    """
    config = "--psm 6 --oem 3"
    try:
        data = pytesseract.image_to_data(
            path,
            lang="eng+rus",
            config=config,
            output_type=pytesseract.Output.DICT
        )
    except Exception:
        return []

    texts = data.get("text", [])
    if not texts:
        return []

    lines = {}
    n = len(texts)

    for i in range(n):
        # Только WORD-уровень (level == 5) имеет отдельные слова, но мы группируем по line_num
        level = data.get("level", [None]*n)[i]
        if level != 5:
            continue

        txt = (data.get("text", [])[i] or "").strip()
        if not txt:
            continue

        # убираем шум
        txt = txt.replace("|", "").strip()
        if not txt:
            continue

        line_id = data.get("line_num", [i+1]*n)[i]

        try:
            x = int(data.get("left", [0]*n)[i])
            y = int(data.get("top", [0]*n)[i])
            w = int(data.get("width", [0]*n)[i])
            h = int(data.get("height", [0]*n)[i])
        except Exception:
            continue

        if line_id not in lines:
            lines[line_id] = {
                "text": txt,
                "bbox": [x, y, w, h]
            }
        else:
            # добавляем слово через пробел
            lines[line_id]["text"] += " " + txt

            # расширяем bbox так, чтобы включить новый фрагмент
            lx, ly, lw, lh = lines[line_id]["bbox"]
            new_x = min(lx, x)
            new_y = min(ly, y)
            new_x2 = max(lx + lw, x + w)
            new_y2 = max(ly + lh, y + h)
            lines[line_id]["bbox"] = [
                new_x,
                new_y,
                new_x2 - new_x,
                new_y2 - new_y
            ]

    # превращаем в список и сортируем по вертикали
    result = list(lines.values())
    result.sort(key=lambda b: b["bbox"][1])
    return result

# ---------------------------
# Drawing helper: fit multiline text into a bbox
# ---------------------------

def draw_text_in_bbox(draw: ImageDraw.Draw, bbox, text, font_path=None, fill=(0,0,0)):
    """
    PIllow 10+ compatible version (no getsize)
    """
    x, y, w, h = bbox

    # load font
    try:
        base_font_size = max(10, int(h * 0.9))
        font = ImageFont.truetype(font_path or "arial.ttf", base_font_size)
    except Exception:
        font = ImageFont.load_default()
        base_font_size = 12

    def measure_text(font, s):
        """Return width, height bounding box of the text."""
        bbox = font.getbbox(s)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width, height

    # decrease font size until fits vertically
    while True:
        # measure average char width
        w_M, _ = measure_text(font, "M")
        avg_char_width = w_M if w_M > 0 else 7

        max_chars_per_line = max(1, int(w / avg_char_width))

        wrapped = textwrap.fill(text, width=max_chars_per_line)
        lines = wrapped.splitlines()

        total_h = 0
        for line in lines:
            _, lh = measure_text(font, line)
            total_h += lh
        total_h += (len(lines) - 1) * 2

        if total_h <= h or base_font_size <= 8:
            break

        base_font_size = max(8, base_font_size - 1)
        try:
            font = ImageFont.truetype(font_path or "arial.ttf", base_font_size)
        except Exception:
            font = ImageFont.load_default()
            break

    # draw lines centered vertically
    cur_y = y + max(0, (h - total_h) // 2)
    for line in lines:
        lw, lh = measure_text(font, line)
        cur_x = x + 0
        draw.text((cur_x, cur_y), line, font=font, fill=fill)
        cur_y += lh + 2

# ---------------------------
# Замена текста на изображении (исправленная)
# ---------------------------

def replace_text_on_image(image_path, boxes, translated_texts):
    """
    Рисуем на оригинальном изображении (не на предобработанном),
    используем bbox из ocr_with_boxes (которые соответствуют исходному размера).
    Подгоняем шрифт и делаем перенос/масштабирование.
    Возвращаем путь к сохранённому файлу.
    """
    if len(boxes) != len(translated_texts):
        # если длины не совпадают — берём по минимуму
        count = min(len(boxes), len(translated_texts))
        boxes = boxes[:count]
        translated_texts = translated_texts[:count]

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Попытаемся загрузить Arial, иначе default
    font_path = None
    try:
        # на Windows обычно есть Arial
        ImageFont.truetype("arial.ttf", 12)
        font_path = "arial.ttf"
    except Exception:
        font_path = None

    # Сортируем по вертикали (top) чтобы замазывать сверху вниз — минимизируем перекрытия
    pairs = list(zip(boxes, translated_texts))
    pairs.sort(key=lambda p: p[0]["bbox"][1])

    for box, new_text in pairs:
        x, y, w, h = box["bbox"]
        # небольшой отступ, чтобы не резать буквы по краю
        pad_x = max(2, int(w * 0.03))
        pad_y = max(1, int(h * 0.08))

        # paint white rectangle with padding to fully cover previous text
        # используем расширение по всем направлениям на случай рукописи/артефактов
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(img.width, x + w + pad_x)
        y1 = min(img.height, y + h + pad_y)

        draw.rectangle([x0, y0, x1, y1], fill="white")

        # разместить текст внутри [x0,y0,width,height]
        inner_bbox = (x0 + 2, y0 + 1, x1 - x0 - 4, y1 - y0 - 2)  # small inner margins
        draw_text_in_bbox(draw, inner_bbox, new_text, font_path=font_path, fill=(0,0,0))

    out_path = OUTPUT_DIR / f"{uuid.uuid4()}_translated_image.jpg"
    img.save(out_path, quality=95)
    return out_path

# ----------------------------------------
# OpenRouter Chat with Safe Handling
# ----------------------------------------
def openrouter_chat(model, messages):
    import requests
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": model, "messages": messages},
            timeout=40
        )
    except Exception as e:
        return f"[OpenRouter request failed] {str(e)}"

    if r.status_code != 200:
        return f"[OpenRouter HTTP {r.status_code}] {r.text}"

    try:
        data = r.json()
    except Exception:
        return f"[OpenRouter returned non-JSON] {r.text}"

    if "choices" not in data:
        return f"[Malformed OpenRouter Response] {data}"

    content = data["choices"][0]["message"]["content"]
    return str(content)  # гарантируем строку

def translate_lines(lines, target_lang):
    translated = []
    for item in lines:
        out = openrouter_chat(
            "openai/gpt-3.5-turbo",
            [
                {
                    "role": "system",
                    "content": (
                        f"Translate into {target_lang}. "
                        "Return ONLY translation, no comments, no explanations."
                    )
                },
                {"role": "user", "content": item["text"]}
            ]
        )
        translated.append(out.strip())
    return translated

SUPPORTED_LANGS = ["ru", "kz", "en", "tr", "zh"]

# ----------------------------------------
# Upload
# ----------------------------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{file_id}_{file.filename}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"status": "ok", "filename": dest.name}

# ----------------------------------------
# Process File
# ----------------------------------------
@app.post("/process")
async def process(
    filename: str = Form(...),
    translate_to: str = Form("ru"),
    style: str = Form("normal"),
    replace_image_text: bool = Form(True)
):
    if translate_to not in SUPPORTED_LANGS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported language: {translate_to}. Supported: {SUPPORTED_LANGS}"}
        )

    path = UPLOAD_DIR / filename
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "file not found"})

    ext = path.suffix.lower()

    # -------------------------------------------
    # AUDIO
    # -------------------------------------------
    if ext in [".mp3", ".wav", ".m4a", ".mp4", ".webm", ".aac", ".ogg"]:
        segments, info = whisper_model.transcribe(str(path))
        text = " ".join([seg.text for seg in segments])
        source = "audio"

        translated_image_url = None
        ocr_lines_result = None

    # -------------------------------------------
    # IMAGE
    # -------------------------------------------
    elif ext in [".jpg", ".jpeg", ".png", ".webp"]:
        source = "image"

        # 1) общий полный OCR текст
        text = ocr_image(str(path))

        # 2) OCR по строкам
        ocr_lines_result = ocr_lines(str(path))
        detected_texts = [l["text"] for l in ocr_lines_result]

        # 3) перевод строк
        translated_lines = translate_lines(ocr_lines_result, translate_to)

        # 4) вставка перевода в картинку
        translated_image_url = None
        if replace_image_text:
            out_path = replace_text_on_image(str(path), ocr_lines_result, translated_lines)
            translated_image_url = f"/outputs/{Path(out_path).name}"

    else:
        return JSONResponse(status_code=400, content={"error": "unsupported file"})

    # -------------------------------------------
    # FULL TRANSLATION
    # -------------------------------------------
    translation = openrouter_chat(
        "openai/gpt-3.5-turbo",
        [
            {
                "role": "system",
                "content": (
                    f"You are a professional translator. Translate the text into {translate_to}. "
                    "Return ONLY the translated text. Keep formatting. "
                    "For Chinese (zh), translate into simplified Chinese."
                )
            },
            {"role": "user", "content": text}
        ]
    )

    # -------------------------------------------
    # PARAPHRASE
    # -------------------------------------------
    paraphrase = openrouter_chat(
        "openai/gpt-3.5-turbo",
        [
            {"role": "system", "content": f"Paraphrase in style {style} ({translate_to})"},
            {"role": "user", "content": translation}
        ]
    )

    # SAVE TEXT FILES
    out1 = OUTPUT_DIR / f"{path.stem}_translation.txt"
    out1.write_text(translation, encoding="utf-8")

    out2 = OUTPUT_DIR / f"{path.stem}_paraphrase_{style}.txt"
    out2.write_text(paraphrase, encoding="utf-8")

    # -------------------------------------------
    # RESPONSE
    # -------------------------------------------
    response = {
        "source": source,
        "original_text": text,
        "translation": translation,
        "paraphrase": paraphrase,
        "files": {
            "translation": f"/outputs/{out1.name}",
            "paraphrase": f"/outputs/{out2.name}",
        }
    }

    if source == "image":
        response["ocr_lines"] = ocr_lines_result
        response["translated_lines"] = translated_lines

        if translated_image_url:
            response["translated_image_url"] = translated_image_url

    return JSONResponse(content=jsonable_encoder(response))


def replace_text_on_image(image_path, lines, translated_lines):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # try load Arial
    try:
        ImageFont.truetype("arial.ttf", 14)
        font_path = "arial.ttf"
    except:
        font_path = None

    for item, new_text in zip(lines, translated_lines):
        x, y, w, h = item["bbox"]

        # белый фон под текст
        draw.rectangle([x, y, x+w, y+h], fill="white")

        # подобрать размер шрифта так, чтобы влезло
        fs = int(h * 0.7)
        while fs > 8:
            try:
                font = ImageFont.truetype(font_path, fs) if font_path else ImageFont.load_default()
            except:
                font = ImageFont.load_default()

            bbox = font.getbbox(new_text)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            if tw <= w and th <= h:
                break

            fs -= 1

        draw.text((x, y), new_text, font=font, fill="black")

    out_path = OUTPUT_DIR / f"{uuid.uuid4()}_translated_image.jpg"
    img.save(out_path, quality=97)
    return out_path


# ----------------------------------------
# TTS
# ----------------------------------------
@app.post("/tts")
async def tts(text: str = Form(...)):
    torch.set_num_threads(1)

    model, _, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language='en',
        speaker='random'
    )

    audio = model.apply_tts(text=text)

    out = OUTPUT_DIR / f"tts_{uuid.uuid4()}.wav"
    wav = (audio * 32767).numpy().astype("int16")
    with open(out, "wb") as f:
        wav.tofile(f)
    return FileResponse(out, media_type="audio/wav")
