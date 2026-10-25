[← Voltar para Família Code](../README.md) | [← Voltar ao Início](../../../README.md)

# LLPT Code Nano (1B)

**LLPT Code Nano** é o modelo ultralight especializado em programação da família LLPT, dimensionado com **1 bilhão de parâmetros**. Projetado especificamente para excelente desempenho em engenharia de software, autocompletar código em tempo real, preenchimento Fill-In-the-Middle (FIM) e execução tanto em **Desktop** quanto em **Mobile**.

---

## 📊 Especificações Técnicas

- **Família:** LLPT Code (Programação & Engenharia de Software)
- **Escala de Parâmetros:** 1B (1 Bilhão)
- **Arquitetura Base:** Llama Causal LM com suporte nativo a Fill-In-the-Middle (FIM)
- **Janela de Contexto:** 8.192 tokens
- **Linguagens Principais:** Python, JavaScript/TypeScript, C/C++, Rust, Go, Java, SQL, Shell, HTML/CSS
- **Hardware Recomendado:**
  - **Desktop:** CPU / GPU (2GB+ VRAM para inferência, 6GB+ VRAM para treino/fine-tuning)
  - **Mobile:** Dispositivos Android (Termux / MLC-LLM) / iOS com 4GB+ RAM (modelo quantizado Q4_K_M)
- **Latência Alvo:** Sub-50ms para autocompletar inline

---

## 🛠️ Estrutura de Treinamento e Pesos

O modelo já possui toda a estrutura pronta para ser treinado e utilizado. Os arquivos e pastas estão organizados como segue:

```text
models/code/llpt-code-nano/
├── README.md               # Documentação principal
├── COMANDOS.txt            # Guia completo de comandos para Desktop e Mobile
├── PESOS_E_FORMATOS.txt    # Especificação dos locais, formatos (.safetensors, .gguf) e precisões
├── config.json             # Configuração da arquitetura neural do modelo (1B, FIM)
├── tokenizer_config.json   # Configuração do tokenizer com tokens especiais de FIM
├── src/
│   └── train.py            # Script de treinamento e fine-tuning
├── dataset/                # Diretório para os dados de treino (train.jsonl, valid.jsonl)
└── weights/                # Diretório reservado para armazenamento dos pesos
    ├── raw/                # Pesos originais em Safetensors / PyTorch (BF16 / FP16)
    ├── checkpoints/        # Checkpoints intermediários do treinamento
    ├── final/              # Modelo final consolidado pós-treino
    └── gguf/               # Pesos quantizados (Q4_K_M, Q8_0) para llama.cpp e Mobile
```

---

## 🚀 Como Treinar e Utilizar

Para obter todos os comandos detalhados de treinamento, fine-tuning com LoRA, conversão para GGUF e inferência:

1. Consulte **[COMANDOS.txt](./COMANDOS.txt)** para a lista completa de comandos organizados por plataforma (**Desktop** e **Mobile**).
2. Consulte **[PESOS_E_FORMATOS.txt](./PESOS_E_FORMATOS.txt)** para detalhes de onde posicionar os arquivos de pesos e quais formatos utilizar (`.safetensors`, `.gguf`, `.onnx`).

---

## 🎯 Casos de Uso Recomendados

- Autocompletar trechos de código (Inline Completion) em IDEs como VS Code, Neovim e JetBrains.
- Preenchimento Fill-In-the-Middle (FIM) para inserção de código no meio de arquivos existentes.
- Geração instantânea de funções, testes unitários e docstrings.
- Execução local em smartphones Android/iOS e computadores sem necessidade de conexão com a internet.

---

## 🔄 Navegação na Família Code

- **Modelo Atual:** LLPT Code Nano (1B)
- **Próximo Modelo:** [LLPT Code Small (10B) →](../llpt-code-small/README.md)
