#!/usr/bin/env python3
"""
Script de Inferência para LLPT Code Nano (1B)

Permite executar geração de código e autocompletar Fill-in-the-Middle (FIM)
carregando pesos treinados ou checkpoints do modelo.
"""

import os
import argparse
import json
import torch
from transformers import AutoTokenizer

try:
    from model import build_model, load_model_config
except ImportError:
    from models.code.llpt_code_nano.src.model import build_model, load_model_config


def get_tokenizer(model_dir: str):
    """
    Carrega ou cria o tokenizer compatível com LLPT Code Nano.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        special_tokens = ["<PRE>", "<SUF>", "<MID>", "<EOT>"]
        tokenizer.add_special_tokens({
            "pad_token": "<pad>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "additional_special_tokens": special_tokens
        })
    return tokenizer


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
    parser.add_argument("--model_dir", type=str, default="models/code/llpt-code-nano/weights/checkpoints/", help="Diretório de pesos/checkpoint")
    parser.add_argument("--config", type=str, default="models/code/llpt-code-nano/config.json", help="Caminho para config.json")
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
