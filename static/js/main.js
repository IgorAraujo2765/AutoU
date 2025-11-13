// static/js/main.js
// Form handling: envia arquivo/texto para /classify e exibe resultados com animações sutis.

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('emailForm');
  const resultsSection = document.getElementById('results');
  const categorySpan = document.getElementById('category');
  const responseBox = document.getElementById('suggestedResponse');
  const clearBtn = document.getElementById('clearBtn');
  const fileInput = document.getElementById('fileInput');
  const textInput = document.getElementById('textInput');

  clearBtn.addEventListener('click', () => {
    fileInput.value = '';
    textInput.value = '';
    resultsSection.classList.add('hidden');
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData();
    if (fileInput.files.length > 0) {
      formData.append('file', fileInput.files[0]);
    } else if (textInput.value.trim().length > 0) {
      formData.append('text', textInput.value.trim());
    } else {
      alert('Por favor, envie um arquivo .txt/.pdf ou cole o texto do e-mail.');
      return;
    }

    // animação de carregamento simples
    const originalBtn = document.querySelector('.btn');
    originalBtn.disabled = true;
    const prevText = originalBtn.textContent;
    originalBtn.textContent = 'Processando...';

    try {
      const resp = await fetch('/classify', { method: 'POST', body: formData });
      if (!resp.ok) {
        const text = await resp.text();
        alert('Erro no servidor: ' + text);
        return;
      }
      const data = await resp.json();
     let categoryColor = '';
if (data.category.toLowerCase().includes('produtivo')) {
  categoryColor = 'green';
} else {
  categoryColor = 'red';
}

// Aplica texto e cor
categorySpan.textContent = data.category;
categorySpan.className = 'category-label ' + data.category.toLowerCase();

// Mostra a resposta sugerida
responseBox.textContent = data.suggested_response;

      // mostra com leve transição
      resultsSection.classList.remove('hidden');
      resultsSection.style.opacity = 0;
      resultsSection.style.transform = 'translateY(8px)';
      requestAnimationFrame(() => {
        resultsSection.style.transition = 'opacity .25s ease, transform .25s ease';
        resultsSection.style.opacity = 1;
        resultsSection.style.transform = 'translateY(0)';
      });
    } catch (err) {
      alert('Erro ao comunicar com o servidor: ' + err.message);
    } finally {
      originalBtn.disabled = false;
      originalBtn.textContent = prevText;
    }
  });
});
