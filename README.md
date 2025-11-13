# AutoU - Classificador e Gerador de Respostas de E-mails (Demo)

Projeto simples para candidatura — frontend responsivo com animações sutis e backend em Flask.

## Como rodar localmente
```bash
git clone <repo>
cd autou-email-classifier
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# (opcional) configurar OpenAI
export OPENAI_API_KEY="sua_chave_aqui"
export OPENAI_MODEL="gpt-4o-mini"  # opcional

python app.py
# Acesse http://localhost:5000
```

## Observações
- `textract` é opcional para extração de PDFs e pode exigir dependências do SO. Para evitar problemas, aceite apenas .txt.
- O classificador local é um fallback didático; treine com dados reais para produção.
- Para deploy no Heroku, adicione este repositório a um app Heroku, configure as ENV vars e faça `git push heroku main`.
