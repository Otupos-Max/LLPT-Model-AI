#!/usr/bin/env python3
"""
Script de Inferência para LLPT Code Nano (1B)

Permite executar geração de código e autocompletar Fill-in-the-Middle (FIM)
carregando pesos treinados ou checkpoints do modelo.
"""

import os
import sys
import argparse
import json
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_model, load_model_config

MODEL_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULT_MODEL_DIR = os.path.join(MODEL_ROOT, "weights", "checkpoints", "final")
DEFAULT_CONFIG = os.path.join(MODEL_ROOT, "config.json")


def get_tokenizer(model_dir: str):
    """
    Carrega o tokenizer salvo junto com o checkpoint (o mesmo usado no treino).
    Se não houver, usa o mesmo fallback do treino (GPT-2 ou tokenizer local).
    """
    from train import get_or_create_tokenizer
    return get_or_create_tokenizer(model_dir)


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 128, device: str = "cpu"):
    """Gera continuação de código a partir de um prompt."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            do_sample=False
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=False)


def generate_fim(model, tokenizer, prefix: str, suffix: str, max_new_tokens: int = 128, device: str = "cpu"):
    """Gera preenchimento FIM entre prefix e suffix."""
    fim_prompt = f"<PRE>{prefix}<SUF>{suffix}<MID>"
    res = generate_text(model, tokenizer, fim_prompt, max_new_tokens=max_new_tokens, device=device)
    return res


def main():
    parser = argparse.ArgumentParser(description="Inferência LLPT Code Nano (1B)")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR, help="Diretório de pesos/checkpoint")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Caminho para config.json")
    parser.add_argument("--prompt", type=str, default="def calculate_factorial(n):", help="Prompt para geração")
    parser.add_argument("--prefix", type=str, default="", help="Prefix para FIM")
    parser.add_argument("--suffix", type=str, default="", help="Suffix para FIM")
    parser.add_argument("--max_tokens", type=int, default=64, help="Número máximo de tokens novos")
    parser.add_argument("--device", type=str, default="cpu", help="Dispositivo (cpu/cuda)")

    args = parser.parse_args()

    print("=== LLPT Code Nano - Execução de Inferência ===")
    print(f"Modelo: {args.model_dir}")
    print(f"Dispositivo: {args.device}")

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    tokenizer = get_tokenizer(args.model_dir)

    weight_file = os.path.join(args.model_dir, "pytorch_model.bin")
    safetensors_file = os.path.join(args.model_dir, "model.safetensors")

    # Usa o config.json salvo no checkpoint (já com o vocabulário do treino), se existir
    ckpt_config = os.path.join(args.model_dir, "config.json")
    if os.path.exists(weight_file) or os.path.exists(safetensors_file):
        if os.path.exists(ckpt_config):
            from transformers import LlamaConfig, LlamaForCausalLM
            model = LlamaForCausalLM(LlamaConfig.from_pretrained(args.model_dir))
        else:
            model = build_model(args.config)
    else:
        model = build_model(args.config)

    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    if os.path.exists(safetensors_file) or os.path.exists(weight_file):
        print(f"Carregando pesos treinados de {args.model_dir}...")
        if os.path.exists(safetensors_file):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_file)
        else:
            state_dict = torch.load(weight_file, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        print("AVISO: Nenhum peso treinado encontrado no diretório especificado. Usando pesos inicializados aleatoriamente.")

    model.to(device)

    if args.prefix or args.suffix:
        print(f"\n[Modo FIM]\nPrefix: {args.prefix}\nSuffix: {args.suffix}")
        output = generate_fim(model, tokenizer, args.prefix, args.suffix, max_new_tokens=args.max_tokens, device=device)
    else:
        print(f"\n[Modo Continuação]\nPrompt: {args.prompt}")
        output = generate_text(model, tokenizer, args.prompt, max_new_tokens=args.max_tokens, device=device)

    print("\n--- Resultado da Geração ---")
    print(output)


if __name__ == "__main__":
    main()
