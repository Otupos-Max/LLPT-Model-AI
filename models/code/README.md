[← Voltar ao Início](../../README.md)

# Família LLPT Code — Programação e Engenharia de Software

A família **LLPT Code** é especializada no domínio do desenvolvimento de software, engenharia de sistemas, análise de código-fonte, geração de testes, depuração e documentação técnica.

Treinados com vasta literatura de programação, repositórios open-source de alta qualidade e conjuntos de dados sintéticos de raciocínio lógico-matemático, estes modelos são otimizados para compreender contextos complexos de software e múltiplos arquivos.

---

## Modelos da Família

| Modelo | Parâmetros | Foco Principal | Hardware Recomendado | Documentação |
| :--- | :---: | :--- | :--- | :---: |
| **LLPT Code Nano** | 1B | Autocompletar local e sugestões em tempo real | CPU / GPU de entrada (4GB VRAM) | [Acessar](./llpt-code-nano/README.md) |
| **LLPT Code Small** | 10B | Geração de funções e refatoração rápida | GPU Intermediária (12GB - 16GB VRAM) | [Acessar](./llpt-code-small/README.md) |
| **LLPT Code Medium** | 30B | Desenvolvimento completo de módulos e depuração | GPU Profissional (24GB VRAM) | [Acessar](./llpt-code-medium/README.md) |
| **LLPT Code Large** | 70B | Arquitetura de software e análise de repositórios | Dual GPU / Server (48GB - 80GB VRAM) | [Acessar](./llpt-code-large/README.md) |
| **LLPT Code Max** | 120B | Raciocínio de sistemas complexos e otimização | Cluster de GPUs (80GB+ VRAM) | [Acessar](./llpt-code-max/README.md) |

---

## Estrutura da Pasta

```text
models/code/
├── README.md             # Este arquivo (Visão geral da família)
├── llpt-code-nano/       # Especificações do LLPT Code Nano (1B)
├── llpt-code-small/      # Especificações do LLPT Code Small (10B)
├── llpt-code-medium/     # Especificações do LLPT Code Medium (30B)
├── llpt-code-large/      # Especificações do LLPT Code Large (70B)
└── llpt-code-max/        # Especificações do LLPT Code Max (120B)
```

---

## Diretrizes de Uso e Escolha

- **Para autocompletar e IDE inline completion:** Utilize o **LLPT Code Nano** (1B) ou **LLPT Code Small** (10B).
- **Para agentes de codificação e resolução de bugs:** Utilize o **LLPT Code Medium** (30B).
- **Para análise de grandes repositórios e refatoração arquitetural:** Utilize o **LLPT Code Large** (70B) ou **LLPT Code Max** (120B).

Para informações detalhadas sobre a arquitetura e capacidade de cada modelo, acesse a documentação do modelo específico.
