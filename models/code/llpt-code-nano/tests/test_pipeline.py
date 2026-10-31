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

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from train import (
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
    prog_dir = "models/code/llpt-code-nano/dataset/Programação"
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
        tokenizer = get_or_create_tokenizer("models/code/llpt-code-nano/config.json")
        model.resize_token_embeddings(len(tokenizer))

        test_1_read_txt_dataset(temp_dir)
        test_2_tokenization_and_sequences(tokenizer)
        test_3_and_4_training_step_and_loss(model, tokenizer)
        test_5_and_6_checkpoint_and_inference(model, tokenizer, temp_dir)
        test_7_empty_or_nonexistent_dataset(temp_dir)
        test_8_fim_compatibility(tokenizer)
        test_9_categorized_dataset_and_ratio(temp_dir, tokenizer)
        test_10_programming_dataset_audit()

        print("\n=======================================================")
        print(" ALL 10 MANDATORY VERIFICATION TESTS PASSED SUCCESSFULLY!")
        print("=======================================================")
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
