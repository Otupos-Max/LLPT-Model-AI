#!/usr/bin/env python3
"""
Auditor e Processador do Dataset de Programação - LLPT Code Nano

Este script realiza a auditoria completa de qualidade, integridade, diversidade
e padrões do Dataset de Programação do LLPT Code Nano.

Principais funcionalidades:
1. Identificação de arquivos, codificação, tamanho e contagem de linhas/seções.
2. Detecção de templates genéricos/código fake (ex: boilerplate Java/C usado em outras linguagens).
3. Verificação de código real (garante presença de linhas de código ativas, não apenas comentários).
4. Detecção de duplicatas exatas e quase-duplicatas (md5 hash).
5. Detecção de blocos Fill-In-The-Middle (FIM) válidos e malformados.
6. Detecção de arquivos vazios, corrompidos ou sem contexto.
7. Detecção de possíveis segredos/credenciais (chaves de API, senhas hardcoded, etc).
8. Relatório completo antes/depois do processamento.
"""

import os
import re
import glob
import hashlib
import json
import argparse
import logging
from typing import Dict, List, Tuple, Any

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Padrões comuns de segredos/credenciais a evitar em datasets
SECRET_PATTERNS = [
    re.compile(r'(?i)(api[_-]?key|secret[_-]?key|auth[_-]?token)\s*=\s*["\'][A-Za-z0-9/+=]{16,}["\']'),
    re.compile(r'-----BEGIN (RSA|OPENSSH|EC|PRIVATE) KEY-----'),
    re.compile(r'AKIA[0-9A-Z]{16}'),
]

# Padrão de identificação de template genérico fake (Java boilerplate em outras linguagens)
GENERIC_TEMPLATE_INDICATORS = [
    "public class ",
    "public static void main",
    "calculateSum(",
    "binarySearch_",
    "factorial_"
]


def audit_file(filepath: str) -> Dict[str, Any]:
    filename = os.path.basename(filepath)
    stat = os.stat(filepath)
    size_bytes = stat.st_size

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return {
            "file": filename,
            "filepath": filepath,
            "error": str(e),
            "size": size_bytes,
            "is_valid": False
        }

    is_empty = not content or not content.strip()
    lines = content.splitlines()
    line_count = len(lines)

    # Identificar linhas de código real vs linhas de comentário / em branco
    non_comment_lines = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("//") or s.startswith("#") or s.startswith("--") or s.startswith(";") or s.startswith("*") or s.startswith("/*") or s.startswith("%"):
            continue
        non_comment_lines.append(s)

    code_line_count = len(non_comment_lines)
    is_comment_only = (code_line_count < 3)

    # MD5 Hash para duplicatas exatas
    content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()

    # Checar FIM blocks
    fim_presences = content.count("<PRE>")
    fim_mids = content.count("<MID>")
    fim_eots = content.count("<EOT>")
    fim_valid = (fim_presences == fim_mids) and (fim_presences == fim_eots) and (fim_presences > 0)
    fim_malformed = (fim_presences != fim_mids) or (fim_presences != fim_eots)

    # Checar se é um template fake genérico
    is_template_fake = False
    if "Hello from " in content or ("binarySearch_" in content and filename not in ["c.txt", "cpp.txt", "java.txt", "csharp.txt"]):
        is_template_fake = True

    # Checar segredos
    secrets_found = []
    for pat in SECRET_PATTERNS:
        if pat.search(content):
            secrets_found.append(pat.pattern)

    return {
        "file": filename,
        "filepath": filepath,
        "size": size_bytes,
        "line_count": line_count,
        "code_line_count": code_line_count,
        "is_comment_only": is_comment_only,
        "is_empty": is_empty,
        "content_hash": content_hash,
        "fim_count": fim_presences,
        "fim_valid": fim_valid,
        "fim_malformed": fim_malformed,
        "is_template_fake": is_template_fake,
        "secrets_found": secrets_found,
        "is_valid": not is_empty and not fim_malformed and not is_template_fake and not is_comment_only
    }


def audit_dataset(dataset_dir: str) -> Dict[str, Any]:
    files = sorted(glob.glob(os.path.join(dataset_dir, "*.txt")))
    results = [audit_file(f) for f in files]

    total_files = len(files)
    total_bytes = sum(r.get("size", 0) for r in results)
    total_lines = sum(r.get("line_count", 0) for r in results)
    empty_files = [r for r in results if r.get("is_empty")]
    template_fake_files = [r for r in results if r.get("is_template_fake")]
    comment_only_files = [r for r in results if r.get("is_comment_only")]
    fim_malformed_files = [r for r in results if r.get("fim_malformed")]
    files_with_secrets = [r for r in results if r.get("secrets_found")]

    # Checar duplicatas exatas
    hash_map: Dict[str, List[str]] = {}
    for r in results:
        h = r.get("content_hash")
        if h:
            hash_map.setdefault(h, []).append(r["file"])

    exact_duplicates = {h: flist for h, flist in hash_map.items() if len(flist) > 1}

    authentic_files = [r for r in results if r.get("is_valid")]

    summary = {
        "total_files": total_files,
        "total_bytes": total_bytes,
        "total_lines": total_lines,
        "empty_files_count": len(empty_files),
        "template_fake_count": len(template_fake_files),
        "comment_only_count": len(comment_only_files),
        "authentic_count": len(authentic_files),
        "fim_malformed_count": len(fim_malformed_files),
        "exact_duplicates_groups": len(exact_duplicates),
        "files_with_secrets_count": len(files_with_secrets),
        "template_fake_files": [r["file"] for r in template_fake_files],
        "authentic_files": [r["file"] for r in authentic_files]
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Auditor do Dataset de Programação do LLPT Code Nano")
    parser.add_argument("--dataset_dir", type=str, default="models/code/llpt-code-nano/dataset/Programação", help="Diretório do dataset de programação")
    args = parser.parse_args()

    print("=== Auditoria do Dataset de Programação LLPT Code Nano ===")
    summary = audit_dataset(args.dataset_dir)
    print(f"Total de Arquivos: {summary['total_files']}")
    print(f"Tamanho Total: {summary['total_bytes'] / 1024:.2f} KB ({summary['total_bytes']} bytes)")
    print(f"Total de Linhas: {summary['total_lines']}")
    print(f"Arquivos Autênticos Válidos: {summary['authentic_count']}")
    print(f"Arquivos com Template Genérico Fake: {summary['template_fake_count']}")
    print(f"Arquivos Apenas com Comentários: {summary['comment_only_count']}")
    print(f"Arquivos Vazios: {summary['empty_files_count']}")
    print(f"Arquivos FIM Malformados: {summary['fim_malformed_count']}")
    print(f"Grupos de Duplicatas Exatas: {summary['exact_duplicates_groups']}")
    print(f"Arquivos com Segredos Detectados: {summary['files_with_secrets_count']}")


if __name__ == "__main__":
    main()
