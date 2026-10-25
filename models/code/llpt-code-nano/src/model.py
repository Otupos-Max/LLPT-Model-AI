#!/usr/bin/env python3
"""
Modelo LLPT Code Nano (1B)

Este módulo define a inicialização e o carregamento do modelo LLPT Code Nano
com base nas configurações definidas em config.json.
"""

import os
import json
import torch
from transformers import LlamaConfig, LlamaForCausalLM, AutoModelForCausalLM, AutoConfig


def load_model_config(config_path: str):
    """Carrega as configurações do modelo a partir do arquivo JSON."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Arquivo de configuração '{config_path}' não encontrado.")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    # Remover chaves personalizadas não reconhecidas por LlamaConfig se houver
    llama_dict = {k: v for k, v in config_dict.items() if k != "fim_tokens"}
    return LlamaConfig(**llama_dict), config_dict


def build_model(config_path: str, torch_dtype: torch.dtype = torch.float32):
    """
    Constrói e inicializa um modelo LLPT Code Nano com pesos aleatórios
    com base na arquitetura definida em config.json.
    """
    config, raw_dict = load_model_config(config_path)
    model = LlamaForCausalLM(config)
    return model


def load_model_and_tokenizer(model_path_or_dir: str, config_path: str = None, device: str = "cpu"):
    """
    Carrega o modelo e seus pesos a partir de um diretório de checkpoint ou weights/raw/final.
    Se nenhum peso pré-treinado for encontrado no diretório, inicializa com pesos aleatórios.
    """
    if config_path is None:
        config_path = os.path.join(model_path_or_dir, "config.json")

    if os.path.exists(os.path.join(model_path_or_dir, "model.safetensors")) or \
       os.path.exists(os.path.join(model_path_or_dir, "pytorch_model.bin")):
        model = AutoModelForCausalLM.from_pretrained(model_path_or_dir)
    else:
        print(f"Nenhum peso pré-treinado encontrado em '{model_path_or_dir}'. Inicializando pesos aleatórios.")
        model = build_model(config_path)

    model.to(device)
    return model


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "models/code/llpt-code-nano/config.json"
    m = build_model(cfg_path)
    num_params = sum(p.numel() for p in m.parameters())
    print(f"Modelo LLPT Code Nano construído com sucesso! Total de parâmetros: {num_params:,}")
