# app.py
# Backend Flask para receber arquivo/texto, pré-processar, classificar e gerar resposta.

from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from pathlib import Path
from werkzeug.utils import secure_filename
import os, re

import pdfplumber
from PIL import Image
import pytesseract

try:
    import openai
except Exception:
    openai = None

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB
UPLOAD_FOLDER = Path('uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

# Lista fixa de stopwords (PT + EN)
STOPWORDS = set([
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com",
    "no", "na", "os", "as", "um", "uma", "uns", "umas", "por", "como",
    "the", "and", "is", "in", "to", "of", "for", "on", "with", "at", "by"
])

def preprocess_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)  # Remove múltiplos espaços
    text = re.sub(r'[^\w\sÀ-ÿ]', ' ', text)  # Remove pontuação
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return ' '.join(tokens)

TRAIN_TEXTS = [
    'Preciso de suporte no sistema, erro ao validar contrato',
    'Atualização sobre o caso 1234: aguardando retorno do cliente',
    'Obrigado pelo atendimento, ótima ajuda!',
    'Parabéns pelo time, feliz aniversário!',
    'Solicito agendamento de manutenção urgente no servidor',
    'Só passando para cumprimentar e desejar boa sorte'
]
TRAIN_LABELS = ['Produtivo', 'Produtivo', 'Improdutivo', 'Improdutivo', 'Produtivo', 'Improdutivo']

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform([preprocess_text(t) for t in TRAIN_TEXTS])
clf = MultinomialNB()
clf.fit(X_train, TRAIN_LABELS)

def extract_text_from_pdf_advanced(filepath):
    text = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if not text.strip():
            # OCR
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    im = page.to_image(resolution=300).original
                    page_text = pytesseract.image_to_string(im, lang='por+eng')
                    if page_text:
                        text += page_text + "\n"

    except Exception as e:
        print("Erro ao extrair PDF avançado:", e)

    return text.strip()

def classify_local(text):
    text = text.lower()
    keywords_produtivas = [
        'relatório', 'projeto', 'envio', 'resposta', 'analisar', 'prazo',
        'agendar', 'confirmar', 'reunião', 'documento', 'pedido', 'solicitação',
        'erro', 'suporte', 'ajuda', 'problema', 'caso', 'atualização', 'pendente',
        'retorno', 'anexo', 'validação', 'contrato', 'ajustar', 'corrigir'
    ]
    keywords_improdutivas = [
        'obrigado', 'obrigada', 'bom dia', 'boa tarde', 'boa noite', 'abraço',
        'feliz aniversário', 'parabéns', 'agradeço', 'saudações', 'tudo bem',
        'olá', 'oi'
    ]
    if any(word in text for word in keywords_produtivas):
        return 'Produtivo'
    elif any(word in text for word in keywords_improdutivas):
        return 'Improdutivo'
    else:
        if len(text.split()) > 10:
            return 'Produtivo'
        else:
            return 'Improdutivo'

def classify_with_openai(text: str):
    if openai is None:
        raise RuntimeError('openai library not installed')

    prompt = f"""
Você é um classificador de e-mails corporativos.

**Regras de classificação:**
- "Produtivo": e-mails que contêm pedidos, tarefas, relatórios, reuniões, prazos, respostas, feedbacks, solicitações, dúvidas ou qualquer conteúdo que exija ação.
- "Improdutivo": e-mails apenas de agradecimento, cumprimentos, conversas casuais, notificações automáticas ou mensagens sem necessidade de ação.

Analise o e-mail abaixo e responda SOMENTE com um JSON no formato exato:

{{
  "category": "Produtivo" ou "Improdutivo",
  "suggested_response": "uma resposta curta e educada apropriada ao caso"
}}

E-MAIL:
{text}
"""

    resp = openai.ChatCompletion.create(
        model=os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'),
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=400,
        temperature=0.1,
    )

    import json
    try:
        parsed = json.loads(resp['choices'][0]['message']['content'])
        return parsed.get('category', 'Produtivo'), parsed.get('suggested_response', '')
    except Exception:
        cat = 'Improdutivo' if 'Improdutivo' in resp else 'Produtivo'
        return cat, resp

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_email():
    text = ''
    if 'file' in request.files and request.files['file'].filename != '':
        f = request.files['file']
        filename = secure_filename(f.filename)
        filepath = Path(app.config['UPLOAD_FOLDER']) / filename
        f.save(filepath)
        if filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf_advanced(filepath)
        else:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
                text = fh.read()
        try:
            filepath.unlink()
        except Exception:
            pass
    elif 'text' in request.form:
        text = request.form['text']

    if not text.strip():
        return jsonify({'error': 'Nenhum conteúdo recebido.'}), 400

    preproc = preprocess_text(text)

    try:
        if os.environ.get('OPENAI_API_KEY') and openai is not None:
            openai.api_key = os.environ['OPENAI_API_KEY']
            category, suggested = classify_with_openai(text)
        else:
            category = classify_local(preproc)
            if category == 'Produtivo':
                suggested = 'Olá,\n\nRecebemos sua solicitação e iremos analisá-la. Por favor, nos envie mais detalhes se houver.\n\nAtenciosamente,'
            else:
                suggested = 'Olá,\n\nAgradecemos a sua mensagem! No momento não é necessária nenhuma ação.\n\nAtenciosamente,'
    except Exception:
        category = classify_local(preproc)
        suggested = 'Erro no serviço externo; resposta gerada localmente.'

    return jsonify({'category': category, 'suggested_response': suggested})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
