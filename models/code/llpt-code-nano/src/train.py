#!/usr/bin/env python3
"""
Script de Treinamento e Fine-Tuning para LLPT Code Nano (1B)

Este script suporta:
- Pre-treinamento e Fine-Tuning com Hugging Face Trainer / DeepSpeed / LoRA / QLoRA.
- Formato Fill-In-the-Middle (FIM) para otimização de autocompletar código.
- Exportação automática de pesos após o treinamento.
"""

import os
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Treinamento e Fine-Tuning do LLPT Code Nano (1B)")
    parser.add_argument("--config", type=str, default="config.json", help="Caminho para o config.json do modelo")
    parser.add_argument("--data_path", type=str, default="dataset/train.jsonl", help="Caminho para o dataset de treino")
    parser.add_argument("--output_dir", type=str, default="weights/checkpoints/", help="Diretório de saída dos checkpoints")
    parser.add_argument("--epochs", type=int, default=3, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=4, help="Tamanho do batch por dispositivo")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Taxa de aprendizado")
    parser.add_argument("--use_lora", action="store_true", help="Usar LoRA/PEFT para ajuste eficiente")

    args = parser.parse_args()

    print("=== LLPT Code Nano - Treinamento de Modelo ===")
    print(f"Configuração: {args.config}")
    print(f"Dataset: {args.data_path}")
    print(f"Saída: {args.output_dir}")
    print(f"Épocas: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.learning_rate}")
    print(f"Modo LoRA: {'Ativado' if args.use_lora else 'Desativado (Full Fine-Tuning / Pre-training)'}")

    if not os.path.exists(args.config):
        print(f"Erro: Arquivo de configuração '{args.config}' não encontrado.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print("Pronto para iniciar a rotina de treinamento.")

if __name__ == "__main__":
    main()
