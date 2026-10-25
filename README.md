# LLPT — Large Language para Trabalho

Uma família de modelos de inteligência artificial voltada para trabalho, desenvolvimento de software, raciocínio e tarefas gerais.

O **LLPT** é um projeto de IA que reúne duas famílias principais de modelos: **LLPT (Uso Geral)** e **LLPT Code (Programação e Engenharia de Software)**.

O projeto possui diferentes escalas de modelos, desde versões ultraleves (1B de parâmetros) para execução local até arquiteturas planejadas de grande escala (com até 5 trilhões de parâmetros).

---

## 📂 Estrutura do Repositório

O repositório é organizado de forma clara e intuitiva para facilitar a navegação:

```text
.
├── README.md                           # Este arquivo (Visão geral do projeto)
└── models/                             # Diretório principal de modelos
    ├── general/                        # Família LLPT (Uso Geral)
    │   ├── README.md                   # Documentação da família Geral
    │   ├── llpt-nano/                  # LLPT Nano (1B)
    │   ├── llpt-mini/                  # LLPT Mini (30B)
    │   ├── llpt-base/                  # LLPT Base (100B)
    │   ├── llpt-pro/                   # LLPT Pro (500B)
    │   ├── llpt-max/                   # LLPT Max (1T)
    │   └── llpt-xion/                  # LLPT Xion (5T)
    └── code/                           # Família LLPT Code (Programação)
        ├── README.md                   # Documentação da família Code
        ├── llpt-code-nano/             # LLPT Code Nano (1B)
        ├── llpt-code-small/            # LLPT Code Small (10B)
        ├── llpt-code-medium/           # LLPT Code Medium (30B)
        ├── llpt-code-large/            # LLPT Code Large (70B)
        └── llpt-code-max/              # LLPT Code Max (120B)
```

---

## 🤖 Famílias de Modelos

### 1. [Família LLPT — Uso Geral](./models/general/README.md)

Modelos destinados a tarefas gerais de linguagem, assistência, análise, raciocínio e produtividade.

| Modelo | Parâmetros | Descrição / Foco | Documentação |
| :--- | :---: | :--- | :---: |
| **LLPT Nano** | 1B | Dispositivos locais, baixíssima latência e borda | [Acessar](./models/general/llpt-nano/README.md) |
| **LLPT Mini** | 30B | Assistência diária, redação e análise intermediária | [Acessar](./models/general/llpt-mini/README.md) |
| **LLPT Base** | 100B | Raciocínio analítico e síntese de conhecimento | [Acessar](./models/general/llpt-base/README.md) |
| **LLPT Pro** | 500B | Demandas corporativas e pesquisas complexas | [Acessar](./models/general/llpt-pro/README.md) |
| **LLPT Max** | 1T | Base central de conhecimento universal e síntese massiva | [Acessar](./models/general/llpt-max/README.md) |
| **LLPT Xion** | 5T | Fronteira do raciocínio automatizado e ciência | [Acessar](./models/general/llpt-xion/README.md) |

👉 *[Ver documentação completa da família Geral →](./models/general/README.md)*

---

### 2. [Família LLPT Code — Programação](./models/code/README.md)

Modelos especializados em programação, geração de código, depuração, análise de projetos e engenharia de software.

| Modelo | Parâmetros | Descrição / Foco | Documentação |
| :--- | :---: | :--- | :---: |
| **LLPT Code Nano** | 1B | Autocompletar em tempo real e sugestões inline em IDEs | [Acessar](./models/code/llpt-code-nano/README.md) |
| **LLPT Code Small** | 10B | Geração de funções, testes e refatoração rápida | [Acessar](./models/code/llpt-code-small/README.md) |
| **LLPT Code Medium** | 30B | Agentes de codificação autônomos e debugging | [Acessar](./models/code/llpt-code-medium/README.md) |
| **LLPT Code Large** | 70B | Análise de repositórios inteiros e arquitetura de software | [Acessar](./models/code/llpt-code-large/README.md) |
| **LLPT Code Max** | 120B | Raciocínio avançado em engenharia de sistemas | [Acessar](./models/code/llpt-code-max/README.md) |

👉 *[Ver documentação completa da família Code →](./models/code/README.md)*

---

## 🎯 Objetivos do Projeto

- Desenvolver uma família própria de modelos de linguagem.
- Criar modelos em diferentes escalas de capacidade.
- Especializar modelos para programação e engenharia de software.
- Explorar raciocínio, compreensão de contexto e resolução de problemas.
- Estruturar uma base para futuras versões e melhorias.
- Integrar os modelos a ferramentas e plataformas de IA.

---

## ⚙️ Arquitetura e Desenvolvimento

Cada modelo possui sua própria configuração de arquitetura, conjunto de dados, tokenizer, estratégia de treinamento e método de inferência.

As decisões técnicas são documentadas na pasta individual de cada modelo, incluindo:

- Arquitetura neural e quantidade de parâmetros.
- Dados e metodologia de treinamento.
- Uso de memória e requisitos de hardware.
- Contexto máximo e capacidades.
- Avaliações de qualidade e desempenho.
- Formatos de distribuição e inferência.

> **Nota:** A quantidade de parâmetros, isoladamente, não determina a qualidade ou a velocidade de um modelo. Arquitetura, dados, treinamento e otimização também são fatores decisivos.

---

## 📌 Status

**Status geral:** em planejamento e desenvolvimento.

Os números apresentados representam as escalas planejadas para a família LLPT. A disponibilidade dos pesos, o estado do treinamento e os resultados de avaliação são documentados individualmente na pasta de cada modelo.

---

## 📄 Licença

A licença do projeto deverá ser especificada conforme as decisões de distribuição e uso dos modelos.

---

LLPT — Large Language para Trabalho
*Uma família de modelos para linguagem, produtividade e programação.*
