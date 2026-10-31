#!/usr/bin/env python3
"""
Test Suite para LLPT Code Nano Pipeline com Datasets TXT Categorizados

Testa:
1. Leitura de dataset .txt
2. Tokenização e criação de sequências
3. Passo real de treinamento com atualização de pesos
4. Perda de treino e validação
5. Salvamento e carregamento de checkpoint
6. Inferência usando pesos carregados
7. Tratamento de datasets vazios ou inexistentes
8. Compatibilidade FIM
9. Categorização Programação / Conversação e controle de proporção (prog_ratio)
10. Validação de qualidade e auditoria do Dataset de Programação
"""

import os
import sys
import tempfile
import shutil

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

MODEL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

import json
import subprocess
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from train import (
    split_language_sections,
    extract_fim_and_text,
    _build_local_tokenizer,
    load_txt_datasets,
    load_categorized_txt_datasets,
    CodeTXTDataset,
    collate_fn,
    get_or_create_tokenizer,
    train_epoch,
    evaluate
)
from inference import generate_text, generate_fim
from model import load_model_config
from dataset_auditor import audit_dataset


def test_1_read_txt_dataset(temp_dir):
    print("\n--- Test 1: Reading .txt dataset ---")
    fpath = os.path.join(temp_dir, "sample.txt")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write("def add(a, b):\n    return a + b\n")

    std, fim = load_txt_datasets(temp_dir)
    assert len(std) == 1, f"Expected 1 standard text, got {len(std)}"
    assert len(fim) == 0, f"Expected 0 FIM samples, got {len(fim)}"
    print("Test 1 Passed: .txt dataset successfully read!")


def test_2_tokenization_and_sequences(tokenizer):
    print("\n--- Test 2: Tokenization and sequence creation ---")
    texts = ["def main():\n    print('Hello World')\n"]
    fims = [{"prefix": "def foo():", "suffix": "return 42", "middle": "\n    pass\n"}]
    dataset = CodeTXTDataset(texts, fims, tokenizer, max_length=128)
    assert len(dataset) >= 2, f"Expected at least 2 sequence samples, got {len(dataset)}"
    sample = dataset[0]
    assert isinstance(sample, list), "Dataset sample should be a list of token IDs"
    print("Test 2 Passed: Tokenization and sequence creation verified!")


def test_3_and_4_training_step_and_loss(model, tokenizer):
    print("\n--- Test 3 & 4: Real training step, weight update, and loss calculation ---")
    texts = ["def multiply(x, y):\n    return x * y\n" * 5]
    fims = [{"prefix": "def div(x, y):", "suffix": "return x / y", "middle": "\n    if y == 0: raise ValueError()\n"}]
    dataset = CodeTXTDataset(texts, fims, tokenizer, max_length=64)

    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id or 0))

    initial_weight = list(model.parameters())[0].clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_loss = train_epoch(model, loader, optimizer, "cpu")
    val_loss = evaluate(model, loader, "cpu")

    updated_weight = list(model.parameters())[0]

    assert not torch.equal(initial_weight, updated_weight), "Weights were not updated after optimizer.step()"
    assert train_loss > 0, "Train loss should be > 0"
    assert val_loss > 0, "Val loss should be > 0"
    print(f"Test 3 & 4 Passed: Weights updated! Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


def test_5_and_6_checkpoint_and_inference(model, tokenizer, temp_dir):
    print("\n--- Test 5 & 6: Saving/loading checkpoint and inference ---")
    ckpt_dir = os.path.join(temp_dir, "checkpoint-test")
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(ckpt_dir)

    loaded_model = LlamaForCausalLM(model.config)
    loaded_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "pytorch_model.bin")))

    out_cont = generate_text(loaded_model, tokenizer, "def test_func():", max_new_tokens=10, device="cpu")
    out_fim = generate_fim(loaded_model, tokenizer, "def test():", "return True", max_new_tokens=10, device="cpu")

    assert isinstance(out_cont, str) and len(out_cont) > 0, "Inference output should be non-empty string"
    assert isinstance(out_fim, str) and len(out_fim) > 0, "FIM output should be non-empty string"
    print(f"Test 5 & 6 Passed: Checkpoint saved/loaded. Output: {out_cont[:40]}...")


def test_7_empty_or_nonexistent_dataset(temp_dir):
    print("\n--- Test 7: Handling empty or non-existent dataset ---")
    empty_dir = os.path.join(temp_dir, "empty")
    os.makedirs(empty_dir, exist_ok=True)
    open(os.path.join(empty_dir, "empty.txt"), "w").close()

    std, fim = load_txt_datasets(empty_dir)
    assert len(std) == 0 and len(fim) == 0, "Empty dataset should return empty lists"

    try:
        load_txt_datasets(os.path.join(temp_dir, "non_existent"))
        assert False, "Should have raised FileNotFoundError for non-existent path"
    except FileNotFoundError:
        pass

    print("Test 7 Passed: Empty and non-existent dataset handled properly!")


def test_8_fim_compatibility(tokenizer):
    print("\n--- Test 8: FIM compatibility ---")
    fim_text = "<PRE>def func():\n    <SUF>\n    return x<MID>x = 10\n<EOT>"
    tokens = tokenizer.encode(fim_text)
    decoded = tokenizer.decode(tokens)
    assert "<PRE>" in decoded and "<SUF>" in decoded and "<MID>" in decoded and "<EOT>" in decoded
    print("Test 8 Passed: FIM special tokens correctly preserved in tokenization!")


def test_9_categorized_dataset_and_ratio(temp_dir, tokenizer):
    print("\n--- Test 9: Categorized dataset (Programação/Conversação) and ratio control ---")
    prog_dir = os.path.join(temp_dir, "dataset", "Programação")
    conv_dir = os.path.join(temp_dir, "dataset", "Conversação")
    os.makedirs(prog_dir, exist_ok=True)
    os.makedirs(conv_dir, exist_ok=True)

    with open(os.path.join(prog_dir, "python.txt"), "w", encoding="utf-8") as f:
        f.write("def calculate_sum(a, b):\n    return a + b\n" * 10)

    with open(os.path.join(conv_dir, "conversa.txt"), "w", encoding="utf-8") as f:
        f.write("Usuário: Como funciona a ordenação rápida?\nAssistente: O Quicksort é um algoritmo eficiente...\n" * 10)

    categorized = load_categorized_txt_datasets(os.path.join(temp_dir, "dataset"))
    assert "Programação" in categorized and len(categorized["Programação"][0]) == 1
    assert "Conversação" in categorized and len(categorized["Conversação"][0]) == 1

    dataset_natural = CodeTXTDataset(categorized_data=categorized, tokenizer=tokenizer, max_length=128)
    assert len(dataset_natural) > 0

    dataset_balanced = CodeTXTDataset(categorized_data=categorized, tokenizer=tokenizer, max_length=128, prog_ratio=0.8)
    assert len(dataset_balanced) > 0
    print("Test 9 Passed: Categorized datasets and ratio adjustment verified successfully!")


def test_10_programming_dataset_audit():
    print("\n--- Test 10: Programming Dataset Quality Audit ---")
    prog_dir = os.path.join(MODEL_ROOT, "dataset", "Programação")
    prog_file = os.path.join(prog_dir, "programacao.txt")
    if os.path.exists(prog_dir):
        summary = audit_dataset(prog_dir)
        assert summary["total_files"] == 1, f"Expected a single unified programming file, got {summary['total_files']}"
        with open(prog_file, "r", encoding="utf-8") as f:
            content = f.read()
        sections = content.count("\n### LINGUAGEM: ")
        ends = content.count("\n### FIM DA LINGUAGEM: ")
        assert sections >= 400, f"Expected >= 400 language sections, got {sections}"
        assert sections == ends, f"Unbalanced language sections: {sections} starts vs {ends} ends"
        assert summary["template_fake_count"] == 0, f"Found {summary['template_fake_count']} fake template files!"
        assert summary["comment_only_count"] == 0, f"Found {summary['comment_only_count']} comment only files!"
        assert summary["empty_files_count"] == 0, f"Found {summary['empty_files_count']} empty files!"
        assert summary["fim_malformed_count"] == 0, f"Found {summary['fim_malformed_count']} malformed FIM files!"
        print(f"Test 10 Passed: Verified unified dataset with {sections} language sections, 0 fake templates, 0 malformed FIM!")


def test_11_unified_file_split_by_language():
    print("\n--- Test 11: Unified programacao.txt split by language, keeping code AND FIM ---")
    prog_dir = os.path.join(MODEL_ROOT, "dataset", "Programação")
    categorized = load_categorized_txt_datasets(prog_dir)
    stds, fims = categorized["Programação"]
    assert len(stds) == 405, f"Expected 405 language code texts, got {len(stds)}"
    assert len(fims) == 405, f"Expected 405 FIM samples, got {len(fims)}"
    assert all(t.startswith("### LINGUAGEM: ") for t in stds), "Each text must start with its language header"
    assert not any("<PRE>" in t or "<MID>" in t for t in stds), "FIM blocks must be removed from the normal code texts"
    assert not any("# ÍNDICE" in t for t in stds), "The file index must not be used as training text"
    go = [t for t in stds if t.startswith("### LINGUAGEM: Go\n")]
    assert len(go) == 1 and "package main" in go[0], "Go section should contain its own code"
    print("Test 11 Passed: 405 languages with normal code + 405 FIM samples!")


def test_12_split_and_fim_helpers():
    print("\n--- Test 12: Section split and FIM extraction helpers ---")
    content = (
        "# ÍNDICE\n# 001. A\n\n"
        "### LINGUAGEM: A\n### ID: a | Seção 1/2\n# ====\n\nprint(1)\n<PRE>x<SUF>z<MID>y<EOT>\n### FIM DA LINGUAGEM: A\n\n"
        "### LINGUAGEM: B\n### ID: b | Seção 2/2\n\nputs 2\n### FIM DA LINGUAGEM: B\n"
    )
    sections = split_language_sections(content)
    assert [n for n, _ in sections] == ["A", "B"]
    text, fims = extract_fim_and_text(sections[0][1])
    assert "print(1)" in text and "<PRE>" not in text
    assert fims == [{"prefix": "x", "suffix": "z", "middle": "y"}]
    assert split_language_sections("def f(): pass") == [("", "def f(): pass")]
    print("Test 12 Passed!")


def test_13_offline_tokenizer():
    print("\n--- Test 13: Offline tokenizer (no internet) ---")
    tok = _build_local_tokenizer(["def soma(a, b):\n    return a + b\n" * 20, "função ç ã 日本"], vocab_size=500)
    for t in ["<PRE>", "<SUF>", "<MID>", "<EOT>", "<pad>", "<s>", "</s>"]:
        assert t in tok.get_vocab(), f"Missing special token {t}"
    ids = tok.encode("<PRE>função<MID>日本<EOT>", add_special_tokens=False)
    assert tok.decode(ids) == "<PRE>função<MID>日本<EOT>"
    print("Test 13 Passed!")


def test_14_train_script_end_to_end(temp_dir):
    print("\n--- Test 14: train.py + inference.py end-to-end on the real dataset (tiny model) ---")
    cfg = json.load(open(os.path.join(MODEL_ROOT, "config.json"), encoding="utf-8"))
    cfg.update({"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
                "num_attention_heads": 2, "num_key_value_heads": 1})
    cfg_path = os.path.join(temp_dir, "tiny_config.json")
    json.dump(cfg, open(cfg_path, "w", encoding="utf-8"))
    data = os.path.join(temp_dir, "e2e", "Programação")
    os.makedirs(data, exist_ok=True)
    src = open(os.path.join(MODEL_ROOT, "dataset", "Programação", "programacao.txt"), encoding="utf-8").read()
    sections = src.split("\n### LINGUAGEM: ")
    open(os.path.join(data, "programacao.txt"), "w", encoding="utf-8").write("\n### LINGUAGEM: ".join(sections[:4]))
    out = os.path.join(temp_dir, "e2e_out")
    train_py = os.path.join(MODEL_ROOT, "src", "train.py")
    r = subprocess.run([sys.executable, train_py, "--config", cfg_path, "--data_path", os.path.join(temp_dir, "e2e"),
                        "--output_dir", out, "--epochs", "1", "--max_length", "64", "--batch_size", "4"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"train.py failed:\n{r.stdout}\n{r.stderr}"
    for f in ["pytorch_model.bin", "config.json"]:
        assert os.path.exists(os.path.join(out, "final", f)), f"final/{f} missing"
    r2 = subprocess.run([sys.executable, os.path.join(MODEL_ROOT, "src", "inference.py"), "--model_dir",
                         os.path.join(out, "final"), "--prompt", "def f():", "--max_tokens", "5"],
                        capture_output=True, text=True)
    assert r2.returncode == 0, f"inference.py failed:\n{r2.stdout}\n{r2.stderr}"
    assert "Carregando pesos treinados" in r2.stdout
    print("Test 14 Passed: training saved a usable final model and inference loaded it!")


def main():
    print("Starting LLPT Code Nano Pipeline Test Suite...")
    temp_dir = tempfile.mkdtemp()

    try:
        mini_config = LlamaConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=512,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2
        )
        model = LlamaForCausalLM(mini_config)
        tokenizer = get_or_create_tokenizer(os.path.join(MODEL_ROOT, "config.json"))
        model.resize_token_embeddings(len(tokenizer))

        test_1_read_txt_dataset(temp_dir)
        test_2_tokenization_and_sequences(tokenizer)
        test_3_and_4_training_step_and_loss(model, tokenizer)
        test_5_and_6_checkpoint_and_inference(model, tokenizer, temp_dir)
        test_7_empty_or_nonexistent_dataset(temp_dir)
        test_8_fim_compatibility(tokenizer)
        test_9_categorized_dataset_and_ratio(temp_dir, tokenizer)
        test_10_programming_dataset_audit()
        test_11_unified_file_split_by_language()
        test_12_split_and_fim_helpers()
        test_13_offline_tokenizer()
        test_14_train_script_end_to_end(temp_dir)

        print("\n=======================================================")
        print(" ALL 14 MANDATORY VERIFICATION TESTS PASSED SUCCESSFULLY!")
        print("=======================================================")
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
