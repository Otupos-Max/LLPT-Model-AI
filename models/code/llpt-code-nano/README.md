[← Voltar para Família Code](../README.md) | [← Voltar ao Início](../../../README.md)

# LLPT Code Nano (1B)

**LLPT Code Nano** é o modelo ultralight especializado em programação da família LLPT, dimensionado com **1 bilhão de parâmetros**. Projetado especificamente para excelente desempenho em engenharia de software, autocompletar código em tempo real, preenchimento Fill-In-the-Middle (FIM) e execução tanto em **Desktop** quanto em **Mobile**.

---

## 📊 Especificações Técnicas

- **Família:** LLPT Code (Programação & Engenharia de Software)
- **Escala de Parâmetros:** 1B (1 Bilhão de Parâmetros)
- **Arquitetura Base:** Llama Causal LM (`LlamaForCausalLM`) com suporte nativo a Fill-In-the-Middle (FIM)
- **Janela de Contexto:** Configurável até 8.192 tokens
- **Linguagens e Conversação Suportadas:** Python, JavaScript, TypeScript, Java, Kotlin, C, C++, C#, Rust, Go, PHP, Ruby, Swift, Dart, HTML, CSS, SQL, Shell, JSON, Scala, Elixir, Haskell, Zig, Lua, R, Perl, Julia, Clojure, Assembly (x86_64/ARM64), Solidity, GraphQL, YAML/TOML/XML, Ada, Apex, AWK, Azure Bicep/ARM, CMake/Makefile, Common Lisp, Crystal, D, Delphi, Dockerfile/Compose, Elm, GDScript (Godot 4), GLSL/HLSL, Gradle/Groovy, Hack, HCL/Terraform, Hy, Protocol Buffers/IDL, KCL/CUE, Kotlin Multiplatform, Lean 4, LLVM IR, Luau, Mojo, Nix Flakes, Apple Metal MSL/OpenCL, Mercury, Q#, Racket, ReScript, SAS/Stata, Scala 3 Cats/ZIO, Smalltalk/Pharo, SystemVerilog/VHDL, Tcl/Tk, Unreal Engine C++, VBA/VBScript, WebAssembly Text (WAT), Wolfram/Mathematica, XSLT/XPath, Exemplos FIM, além de Q&A, System Design, Code Review e documentações/conversações técnicas.
- **Hardware Recomendado:**
  - **Desktop:** CPU / GPU (2GB+ VRAM para inferência, 6GB+ VRAM para treino/fine-tuning completo de 1B)
  - **Mobile:** Dispositivos Android (Termux / MLC-LLM) / iOS com 4GB+ RAM (modelo quantizado Q4_K_M)
- **Latência Alvo:** Sub-50ms para autocompletar inline

---

## 📂 Estrutura do Dataset (.txt) v1

O LLPT Code Nano conta com o maior dataset de programação da versão v1 da família, organizando os arquivos de texto plano (`.txt`) em duas pastas principais dentro do diretório `dataset/`:

```text
models/code/llpt-code-nano/dataset/
├── Programação/
│   ├── python.txt                       # Código Python
│   ├── javascript.txt                   # Código JavaScript
│   ├── typescript.txt                   # Código TypeScript
│   ├── java.txt                         # Código Java
│   ├── kotlin.txt                       # Código Kotlin
│   ├── c.txt                            # Código C
│   ├── cpp.txt                          # Código C++
│   ├── csharp.txt                       # Código C#
│   ├── rust.txt                         # Código Rust
│   ├── go.txt                           # Código Go
│   ├── php.txt                          # Código PHP
│   ├── ruby.txt                         # Código Ruby
│   ├── swift.txt                        # Código Swift
│   ├── dart.txt                         # Código Dart
│   ├── scala.txt                        # Código Scala
│   ├── elixir.txt                       # Código Elixir
│   ├── haskell.txt                      # Código Haskell
│   ├── zig.txt                          # Código Zig
│   ├── lua.txt                          # Código Lua
│   ├── r.txt                            # Código R
│   ├── perl.txt                         # Código Perl
│   ├── julia.txt                        # Código Julia
│   ├── clojure.txt                      # Código Clojure
│   ├── assembly.txt                     # Assembly (x86_64 / ARM64)
│   ├── solidity.txt                     # Smart Contracts Solidity
│   ├── graphql.txt                      # Schemas e Queries GraphQL
│   ├── yaml_toml_xml.txt                # Configurações K8s, TOML, XML
│   ├── ada.txt                          # Ada 2012 / SPARK
│   ├── apex.txt                         # Salesforce Apex & SOQL
│   ├── awk.txt                          # Scripting e parsing AWK
│   ├── bicep_arm.txt                    # Azure Bicep & ARM IaC
│   ├── cmake_makefile.txt               # CMake & Makefile
│   ├── common_lisp.txt                  # Common Lisp / SBCL
│   ├── crystal.txt                      # Linguagem Crystal
│   ├── d_lang.txt                       # D Programming Language
│   ├── delphi_pascal.txt                # Object Pascal / Delphi
│   ├── dockerfile_compose.txt           # Dockerfile & Docker Compose
│   ├── elm.txt                          # Elm Frontend Architecture
│   ├── gdscript.txt                     # Godot 4 GDScript
│   ├── glsl_hlsl.txt                    # Shaders GLSL, HLSL, WGSL
│   ├── groovy_gradle.txt                # Gradle DSL & Jenkins pipelines
│   ├── hack.txt                         # Hack / HHVM
│   ├── hcl_terraform.txt                # HashiCorp HCL & Terraform
│   ├── hy.txt                           # Hy (Lisp em Python AST)
│   ├── idl_protobuf.txt                 # Protocol Buffers v3 & gRPC
│   ├── kcl_cue.txt                      # Linguagens CUE & KCL
│   ├── kotlin_multiplatform.txt         # Kotlin Multiplatform, Coroutines, Flow
│   ├── lean.txt                         # Theorem Prover Lean 4
│   ├── llvm_ir.txt                      # LLVM IR & TableGen
│   ├── lua_luau.txt                     # Luau (Roblox / Typed Lua)
│   ├── mojo.txt                         # Mojo AI Programming
│   ├── nix.txt                          # Nix Expressions & Flakes
│   ├── opencl_metal.txt                 # Metal MSL & OpenCL
│   ├── prolog_mercury.txt               # Mercury & Declarative Logic
│   ├── qsharp.txt                       # Q# Quantum & OpenQASM
│   ├── racket.txt                       # Racket Functional Language
│   ├── reasonml_rescript.txt            # ReScript & ReasonML
│   ├── sas_stata.txt                    # SAS & Stata Data Science
│   ├── scala_3_cats.txt                 # Scala 3, Cats Effect, ZIO
│   ├── smalltalk.txt                    # Smalltalk / Pharo OOP
│   ├── systemverilog_vhdl.txt           # SystemVerilog, Verilog, VHDL
│   ├── tcl_tk.txt                       # Scripting Tcl/Tk
│   ├── unreal_cpp_blueprints.txt        # Unreal C++ & Unity C#
│   ├── vbs_vba.txt                      # VBA Excel & VBScript
│   ├── wasm_wat.txt                     # WebAssembly Text (WAT)
│   ├── wolfram_mathematica.txt          # Wolfram Language
│   ├── xslt_xpath.txt                   # XSLT 3.0 & XPath
│   └── fim_examples.txt                 # Exemplos FIM com tokens <PRE>, <SUF>, <MID>, <EOT>
│
└── Conversação/
    ├── documentation.txt                # Documentação técnica e padrões SOLID
    ├── conversacao_tecnica_qa.txt       # Perguntas, respostas e discussões técnicas
    ├── prompt_engineering_instructions.txt # Instruções de Prompt Engineering
    ├── system_design_architecture.txt   # Guia de System Design e Arquitetura
    └── code_review_best_practices.txt   # Boas práticas e checklist de Code Review
```

### Como Adicionar Novos Arquivos e Categorias

1. Adicione arquivos `.txt` dentro de `models/code/llpt-code-nano/dataset/Programação/` para código-fonte e exemplos de linguagens.
2. Adicione arquivos `.txt` dentro de `models/code/llpt-code-nano/dataset/Conversação/` para conversas, Q&A, explicações de tarefas e auxílio ao desenvolvedor.
3. O leitor do `train.py` varre o diretório e categoriza automaticamente as amostras.
4. Utilize a flag `--prog_ratio` (ex: `--prog_ratio 0.8`) no `train.py` para controlar a proporção de exemplos de programação e conversação.
5. Arquivos vazios ou compostos apenas por espaços são identificados e ignorados com aviso no log.

---

## 🛠️ Estrutura do Projeto e Pesos

```text
models/code/llpt-code-nano/
├── README.md               # Documentação principal
├── COMANDOS.txt            # Guia completo de comandos para Desktop e Mobile
├── PESOS_E_FORMATOS.txt    # Especificação dos locais, formatos (.safetensors, .gguf) e precisões
├── config.json             # Configuração da arquitetura neural do modelo (1B, FIM)
├── tokenizer_config.json   # Configuração do tokenizer com tokens especiais de FIM
├── src/
│   ├── model.py            # Definição e carregamento do modelo PyTorch
│   ├── train.py            # Pipeline completo de treinamento, FIM, chunking, categorização e checkpoints
│   └── inference.py        # Pipeline de geração de código e autocompletar FIM
├── tests/
│   └── test_pipeline.py    # Suíte de testes automatizados do pipeline
├── dataset/                # Diretório com datasets organizados em Programação/ e Conversação/
│   ├── Programação/
│   └── Conversação/
└── weights/                # Diretório reservado para armazenamento dos pesos
    ├── checkpoints/        # Checkpoints salvos durante o treinamento (checkpoint-epoch-X/)
    ├── final/              # Modelo final consolidado pós-treino
    └── gguf/               # Pesos quantizados (Q4_K_M, Q8_0) para llama.cpp e Mobile
```

---

## ⚠️ Diferença entre Pesos Inicializados e Pesos Treinados

- **Pesos Inicializados (Aleatórios):** Quando o modelo é instanciado sem carregar pesos pré-treinados (`pytorch_model.bin` ou `model.safetensors`), seus parâmetros são inicializados aleatoriamente a partir da distribuição definida em `config.json`. A geração de texto resultará em tokens aleatórios até que o modelo seja treinado.
- **Pesos Treinados:** Pesos gerados após a execução do processo de otimização (`train.py`) com backpropagation sobre o dataset. Arquivos grandes de pesos não devem ser salvos diretamente no Git.

---

## 🚀 Como Treinar e Utilizar

1. **Dependências:** `pip install torch transformers peft datasets accelerate`
2. **Iniciar Treinamento:**
   ```bash
   python3 models/code/llpt-code-nano/src/train.py \
     --config models/code/llpt-code-nano/config.json \
     --data_path models/code/llpt-code-nano/dataset/ \
     --output_dir models/code/llpt-code-nano/weights/checkpoints/ \
     --epochs 3 --batch_size 2 --learning_rate 5e-5 --prog_ratio 0.8
   ```
3. **Retomar Treinamento:**
   ```bash
   python3 models/code/llpt-code-nano/src/train.py \
     --resume_from_checkpoint models/code/llpt-code-nano/weights/checkpoints/checkpoint-epoch-1
   ```
4. **Executar Inferência:**
   ```bash
   python3 models/code/llpt-code-nano/src/inference.py \
     --model_dir models/code/llpt-code-nano/weights/checkpoints/final/ \
     --prompt "def calculate_fibonacci(n):"
   ```

Para lista completa de comandos organizados por plataforma (**Desktop** e **Mobile / Termux**), consulte **[COMANDOS.txt](./COMANDOS.txt)**.
