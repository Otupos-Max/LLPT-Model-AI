[← Voltar para Família Geral](../README.md) | [← Voltar ao Início](../../../README.md)

# LLPT Base

**LLPT Base** é o pilar central da família geral com **100 bilhões de parâmetros**. Foi dimensionado para entregar alto desempenho em raciocínio analítico, síntese de conhecimento e resolução de problemas complexos.

---

## 📊 Especificações Técnicas

- **Família:** LLPT (Uso Geral)
- **Escala de Parâmetros:** 100B (100 Bilhões)
- **Janela de Contexto Planejada:** 65.536 tokens
- **Hardware Recomendado:** Servidor dedicado com múltiplas GPUs (ex: 2x a 4x 80GB VRAM - A100/H100)
- **Formato Target:** Safetensors / vLLM / TensorRT-LLM

---

## 🎯 Casos de Uso Recomendados

- Agentes autônomos avançados para tomada de decisões.
- Análise detalhada de dados e síntese de relatórios complexos.
- Tradução de alta precisão entre múltiplos domínios técnicos.
- Motores de busca semântica corporativa e RAG em escala.

---

## 🛠️ Arquitetura e Estrutura

- **Atenção e Posição:** RoPE escalar com janela estendida, suporte a atenção esparsa/densa.
- **Treinamento:** Pipeline distribuído com otimização 3D Parallelism (Data, Tensor, Pipeline).
- **Status:** Em planejamento e estruturação inicial.

---

## 🔄 Navegação na Família General

- **Modelo Anterior:** [← LLPT Mini (30B)](../llpt-mini/README.md)
- **Modelo Atual:** LLPT Base (100B)
- **Próximo Modelo:** [LLPT Pro (500B) →](../llpt-pro/README.md)
