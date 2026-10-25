[← Voltar ao Início](../../README.md)

# Família LLPT — Uso Geral

A família **LLPT** é composta por modelos de inteligência artificial desenvolvidos para uso geral em linguagem natural, assistência virtual, análise de dados, raciocínio lógico, criação de conteúdo e produtividade.

Esta linha engloba desde modelos altamente leves (1B de parâmetros) executáveis em dispositivos locais até arquiteturas de escala massiva (5T de parâmetros) projetadas para supercomputadores.

---

## Modelos da Família

| Modelo | Parâmetros | Foco Principal | Hardware Recomendado | Documentação |
| :--- | :---: | :--- | :--- | :---: |
| **LLPT Nano** | 1B | Dispositivos locais e borda | CPU / GPU de entrada (4GB VRAM) | [Acessar](./llpt-nano/README.md) |
| **LLPT Mini** | 30B | Assistência intermediária e workflows locais | GPU Consumer (24GB VRAM) | [Acessar](./llpt-mini/README.md) |
| **LLPT Base** | 100B | Raciocínio avançado e análises profundas | Multi-GPU / Server (80GB VRAM) | [Acessar](./llpt-base/README.md) |
| **LLPT Pro** | 500B | Soluções corporativas e análises complexas | Cluster de GPUs (8x 80GB) | [Acessar](./llpt-pro/README.md) |
| **LLPT Max** | 1T | Conhecimento enciclopédico e síntese massiva | Cluster de Computação Distribuída | [Acessar](./llpt-max/README.md) |
| **LLPT Xion** | 5T | Fronteira do raciocínio e pesquisa avançada | Infraestrutura de Supercomputação | [Acessar](./llpt-xion/README.md) |

---

## Estrutura da Pasta

```text
models/general/
├── README.md           # Este arquivo (Visão geral da família)
├── llpt-nano/          # Especificações do LLPT Nano (1B)
├── llpt-mini/          # Especificações do LLPT Mini (30B)
├── llpt-base/          # Especificações do LLPT Base (100B)
├── llpt-pro/           # Especificações do LLPT Pro (500B)
├── llpt-max/           # Especificações do LLPT Max (1T)
└── llpt-xion/          # Especificações do LLPT Xion (5T)
```

---

## Diretrizes de Uso e Escolha

- **Para execução local/offline rápida:** Utilize o **LLPT Nano** (1B).
- **Para assistentes pessoais de alta precisão:** Utilize o **LLPT Mini** (30B).
- **Para servidores corporativos e análise pesada de texto:** Utilize o **LLPT Base** (100B) ou **LLPT Pro** (500B).
- **Para pesquisas de alta complexidade e conhecimento massivo:** Utilize o **LLPT Max** (1T) ou **LLPT Xion** (5T).

Para informações detalhadas de cada modelo, navegue até a pasta correspondente.
