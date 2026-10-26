#!/usr/bin/env python3
"""
Script de Treinamento e Fine-Tuning para LLPT Code Nano (1B)

Este script suporta:
- Leitura recursiva e categorizada de arquivos .txt em diretórios (Programação vs Conversação).
- Identificação e descarte de arquivos vazios com avisos claros.
- Suporte nativo a Fill-In-The-Middle (FIM) e continuação de código normal.
- Tokenização, chunking em janelas de contexto (até 8192 tokens).
- Controle de proporção entre exemplos de programação e conversação (--prog_ratio).
- Treinamento real via PyTorch com otimizador AdamW, backpropagation, cálculo de perda e validação.
- Salvamento e retomada de checkpoints.
"""

import os
import re
import sys
import glob
import math
import gc
import json
import argparse
import logging
from typing import List, Dict, Tuple, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from model import build_model, load_model_config
except ImportError:
    from models.code.llpt_code_nano.src.model import build_model, load_model_config

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def find_txt_files(data_path: str) -> List[str]:
    """
    Encontra recursivamente todos os arquivos .txt em um diretório
    ou retorna o próprio arquivo se um caminho de arquivo .txt for fornecido.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Caminho do dataset '{data_path}' não foi encontrado.")

    if os.path.isfile(data_path):
        if data_path.endswith(".txt"):
            return [data_path]
        else:
            logging.warning(f"O arquivo fornecido '{data_path}' não possui extensão .txt.")
            return [data_path]

    txt_files = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))

    return sorted(txt_files)


def classify_file_category(filepath: str, base_data_path: str) -> str:
    """
    Classifica um arquivo .txt na categoria 'Programação', 'Conversação' ou 'Geral'
    com base no caminho relativo e diretórios pai.
    """
    rel_path = os.path.relpath(filepath, base_data_path) if os.path.isdir(base_data_path) else filepath
    parts = [p.lower() for p in rel_path.split(os.sep)]

    for part in parts:
        if "program" in part:
            return "Programação"
        if "convers" in part:
            return "Conversação"

    return "Geral"


def load_categorized_txt_datasets(data_path: str) -> Dict[str, Tuple[List[str], List[Dict[str, str]]]]:
    """
    Lê os arquivos .txt e separa em categorias ('Programação', 'Conversação', 'Geral').
    Retorna um dicionário: {categoria: (standard_texts, fim_samples)}
    """
    txt_files = find_txt_files(data_path)
    if not txt_files:
        logging.warning(f"Nenhum arquivo .txt encontrado em '{data_path}'.")
        return {}

    logging.info(f"Encontrados {len(txt_files)} arquivo(s) em '{data_path}'. Processando e categorizando...")

    categorized = {
        "Programação": ([], []),
        "Conversação": ([], []),
        "Geral": ([], [])
    }
    empty_files_count = 0
    fim_pattern = re.compile(r'<PRE>(.*?)(?:<SUF>(.*?))?<MID>(.*?)(?:<EOT>|$)', re.DOTALL)

    for filepath in txt_files:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            if not content or not content.strip():
                logging.warning(f"Arquivo vazio ou apenas com espaços ignorado: '{filepath}'")
                empty_files_count += 1
                continue

            category = classify_file_category(filepath, data_path)
            std_list, fim_list = categorized[category]

            if "<PRE>" in content and "<MID>" in content:
                matches = fim_pattern.findall(content)
                if matches:
                    for match in matches:
                        prefix, suffix, middle = match
                        fim_list.append({
                            "prefix": prefix,
                            "suffix": suffix or "",
                            "middle": middle
                        })
                    logging.info(f"[{category}] Arquivo '{os.path.basename(filepath)}': {len(matches)} exemplo(s) FIM extraído(s).")
                else:
                    std_list.append(content)
            else:
                std_list.append(content)
        except Exception as e:
            logging.error(f"Erro ao ler arquivo '{filepath}': {e}")

    if empty_files_count > 0:
        logging.info(f"Total de arquivos vazios ignorados: {empty_files_count}")

    for cat, (stds, fims) in categorized.items():
        if stds or fims:
            logging.info(f"Categoria '{cat}': {len(stds)} texto(s) padrão, {len(fims)} amostra(s) FIM.")

    return categorized


def load_txt_datasets(data_path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Lê os arquivos .txt, ignora arquivos vazios com aviso,
    e separa amostras FIM de amostras de código normal.
    Mantém compatibilidade com assinaturas existentes.
    """
    categorized = load_categorized_txt_datasets(data_path)
    all_standard = []
    all_fim = []
    for stds, fims in categorized.values():
        all_standard.extend(stds)
        all_fim.extend(fims)
    return all_standard, all_fim


def get_or_create_tokenizer(model_dir_or_config: str):
    """
    Garante um tokenizer funcional e adiciona tokens FIM.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir_or_config)
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


class CodeTXTDataset(Dataset):
    """
    Dataset customizado para tokenização e chunking em sequências de contexto,
    com suporte opcional a categorização e controle de proporção (--prog_ratio).
    """
    def __init__(
        self,
        standard_texts: Optional[List[str]] = None,
        fim_samples: Optional[List[Dict[str, str]]] = None,
        tokenizer = None,
        max_length: int = 2048,
        categorized_data: Optional[Dict[str, Tuple[List[str], List[Dict[str, str]]]]] = None,
        prog_ratio: Optional[float] = None
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = min(max_length, 8192)

        if categorized_data is not None:
            self._init_from_categorized(categorized_data, prog_ratio)
        else:
            standard_texts = standard_texts or []
            fim_samples = fim_samples or []
            self._init_from_raw(standard_texts, fim_samples)

    def _tokenize_item_list(self, standard_texts: List[str], fim_samples: List[Dict[str, str]]) -> List[List[int]]:
        chunks = []
        for text in standard_texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            for i in range(0, len(tokens), self.max_length):
                chunk = tokens[i:i + self.max_length]
                if len(chunk) > 1:
                    chunks.append(chunk)

        for fim in fim_samples:
            fim_text = f"<PRE>{fim['prefix']}<SUF>{fim['suffix']}<MID>{fim['middle']}<EOT>"
            tokens = self.tokenizer.encode(fim_text, add_special_tokens=True)
            for i in range(0, len(tokens), self.max_length):
                chunk = tokens[i:i + self.max_length]
                if len(chunk) > 1:
                    chunks.append(chunk)
        return chunks

    def _init_from_raw(self, standard_texts: List[str], fim_samples: List[Dict[str, str]]):
        self.samples = self._tokenize_item_list(standard_texts, fim_samples)

    def _init_from_categorized(
        self,
        categorized_data: Dict[str, Tuple[List[str], List[Dict[str, str]]]],
        prog_ratio: Optional[float] = None
    ):
        prog_stds, prog_fims = categorized_data.get("Programação", ([], []))
        conv_stds, conv_fims = categorized_data.get("Conversação", ([], []))
        gen_stds, gen_fims = categorized_data.get("Geral", ([], []))

        prog_chunks = self._tokenize_item_list(prog_stds, prog_fims)
        conv_chunks = self._tokenize_item_list(conv_stds, conv_fims)
        gen_chunks = self._tokenize_item_list(gen_stds, gen_fims)

        logging.info(
            f"Chunking de sequências (max_length={self.max_length}): "
            f"Programação: {len(prog_chunks)} sequências | "
            f"Conversação: {len(conv_chunks)} sequências | "
            f"Geral: {len(gen_chunks)} sequências."
        )

        if prog_ratio is not None and 0.0 < prog_ratio < 1.0 and (prog_chunks or conv_chunks):
            if len(prog_chunks) > 0 and len(conv_chunks) > 0:
                target_conv_count = max(1, int(len(prog_chunks) * (1.0 - prog_ratio) / prog_ratio))
                if target_conv_count < len(conv_chunks):
                    selected_conv_chunks = conv_chunks[:target_conv_count]
                else:
                    times = math.ceil(target_conv_count / len(conv_chunks))
                    selected_conv_chunks = (conv_chunks * times)[:target_conv_count]

                balanced_samples = prog_chunks + selected_conv_chunks + gen_chunks
                actual_prog_ratio = len(prog_chunks) / (len(prog_chunks) + len(selected_conv_chunks))
                logging.info(
                    f"Ajuste de Proporção Aplicado (prog_ratio={prog_ratio:.2f}): "
                    f"Programação={len(prog_chunks)}, Conversação (ajustado)={len(selected_conv_chunks)}. "
                    f"Proporção final de Programação: {actual_prog_ratio:.2%}"
                )
                self.samples = balanced_samples
            else:
                self.samples = prog_chunks + conv_chunks + gen_chunks
        else:
            self.samples = prog_chunks + conv_chunks + gen_chunks
            total_cat = len(prog_chunks) + len(conv_chunks)
            if total_cat > 0:
                logging.info(
                    f"Proporção natural do dataset: "
                    f"Programação={len(prog_chunks)/total_cat:.2%}, "
                    f"Conversação={len(conv_chunks)/total_cat:.2%}"
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, pad_token_id: int):
    """
    Agrupa e faz padding de sequências de comprimento variável.
    """
    max_len = max(len(s) for s in batch)
    input_ids = []
    labels = []
    attention_mask = []

    for item in batch:
        pad_len = max_len - len(item)
        padded_ids = item + [pad_token_id] * pad_len
        padded_labels = item + [-100] * pad_len
        mask = [1] * len(item) + [0] * pad_len

        input_ids.append(padded_ids)
        labels.append(padded_labels)
        attention_mask.append(mask)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
    }


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    steps = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / steps if steps > 0 else 0.0


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            total_loss += outputs.loss.item()
            steps += 1

    return total_loss / steps if steps > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Treinamento LLPT Code Nano (1B) com Datasets TXT Categorizados")
    parser.add_argument("--config", type=str, default="models/code/llpt-code-nano/config.json", help="Caminho para o config.json")
    parser.add_argument("--data_path", type=str, default="models/code/llpt-code-nano/dataset/", help="Diretório ou arquivo .txt do dataset")
    parser.add_argument("--output_dir", type=str, default="models/code/llpt-code-nano/weights/checkpoints/", help="Diretório de saída dos checkpoints")
    parser.add_argument("--epochs", type=int, default=3, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=2, help="Tamanho do batch")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Taxa de aprendizado")
    parser.add_argument("--max_length", type=int, default=512, help="Comprimento máximo da sequência (máx 8192)")
    parser.add_argument("--val_split", type=float, default=0.1, help="Proporção de validação")
    parser.add_argument("--prog_ratio", type=float, default=None, help="Proporção de exemplos de programação (ex: 0.8)")
    parser.add_argument("--device", type=str, default="cpu", help="Dispositivo (cpu/cuda)")
    parser.add_argument("--use_lora", action="store_true", help="Usar LoRA se disponível")
    parser.add_argument("--resume_from_checkpoint", type=str, default="", help="Caminho do checkpoint para retomar")

    args = parser.parse_args()

    print("=== LLPT Code Nano - Treinamento de Modelo ===")
    print(f"Configuração: {args.config}")
    print(f"Dataset: {args.data_path}")
    print(f"Saída: {args.output_dir}")
    print(f"Épocas: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.learning_rate}")
    if args.prog_ratio is not None:
        print(f"Proporção de Programação solicitada: {args.prog_ratio:.2f}")

    # 1. Carregar dataset categorizado
    categorized_data = load_categorized_txt_datasets(args.data_path)
    has_data = any(stds or fims for stds, fims in categorized_data.values())
    if not has_data:
        logging.error("Erro: Nenhum dado válido foi encontrado para iniciar o treinamento. Abortando.")
        sys.exit(1)

    # 2. Tokenizer & Model
    tokenizer = get_or_create_tokenizer(args.config)

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    logging.info(f"Usando dispositivo: {device}")

    model = build_model(args.config)

    # Redimensionar embeddings para cobrir vocab_size
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        logging.info(f"Retomando treinamento a partir do checkpoint: {args.resume_from_checkpoint}")
        ckpt_weight = os.path.join(args.resume_from_checkpoint, "pytorch_model.bin")
        if os.path.exists(ckpt_weight):
            model.load_state_dict(torch.load(ckpt_weight, map_location=device), strict=False)

    model.to(device)

    # 3. Preparar Dataset e DataLoader
    dataset = CodeTXTDataset(
        categorized_data=categorized_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        prog_ratio=args.prog_ratio
    )
    if len(dataset) == 0:
        logging.error("O dataset resultou em 0 amostras de sequências. Verifique o conteúdo dos arquivos .txt.")
        sys.exit(1)

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size

    if val_size > 0:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    else:
        train_dataset = dataset
        val_dataset = None

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_id))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_id)) if val_dataset else None

    # 4. Otimizador
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    os.makedirs(args.output_dir, exist_ok=True)

    # 5. Loop de Treinamento
    logging.info("Iniciando otimização por backpropagation real...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device) if val_loader else None

        val_str = f" | Val Loss: {val_loss:.4f}" if val_loss is not None else ""
        logging.info(f"Época {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}{val_str}")

        # Salvar Checkpoint
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(ckpt_dir)
        with open(os.path.join(ckpt_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(model.config.to_dict(), f, indent=2)

    # Salvar modelo final
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(final_dir)
    logging.info(f"Treinamento concluído. Checkpoints e pesos finais salvos em {args.output_dir}")

    # Liberação de recursos
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
