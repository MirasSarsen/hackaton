import os, requests
LIBRE_URL = os.getenv('LIBRE_TRANSLATE_URL')
LANG_MAP = {'ru': 'ru', 'kz': 'kk', 'en': 'en'}

def translate_text(text: str, target_lang='ru') -> str:
    if not text:
        return ''
    tgt = LANG_MAP.get(target_lang, target_lang)
    if LIBRE_URL:
        try:
            r = requests.post(f'{LIBRE_URL}/translate', json={
                'q': text,
                'source': 'auto',
                'target': tgt,
                'format': 'text'
            }, timeout=30)
            if r.ok:
                return r.json().get('translatedText', text)
        except Exception as e:
            print('libre translate failed:', e)
    return text
