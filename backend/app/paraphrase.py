import os
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
if OPENAI_KEY:
    try:
        import openai
        openai.api_key = OPENAI_KEY
    except Exception:
        openai = None
else:
    openai = None

def paraphrase_text(text: str, style: str = 'hype') -> str:
    if not text:
        return ''
    if openai:
        prompt = f"Paraphrase the following text into a short {style} style. Keep it concise. Text: '''{text}'''"
        try:
            resp = openai.ChatCompletion.create(
                model='gpt-4o-mini',
                messages=[{'role':'user','content':prompt}],
                max_tokens=200,
                temperature=0.9
            )
            return resp['choices'][0]['message']['content'].strip()
        except Exception as e:
            print('openai paraphrase failed:', e)
    if style == 'hype':
        out = text.upper() + '!!! ' + 'Unbelievable, nailed it — keep going!'
    elif style == 'optimist':
        out = text + ' — Отличный результат! Продолжай в том же духе.'
    elif style == 'calm':
        out = 'Сдержанно: ' + text.capitalize()
    else:
        out = text
    return out
