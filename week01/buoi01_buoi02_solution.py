"""Week 1 solutions."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image




TEXTS = [
    "hoc may va tri tue nhan tao",
    "tri tue nhan tao va khoa hoc du lieu",
    "lap trinh python cho may hoc",
    "phan tich du lieu bang python va numpy",
    "hoc sau hon ve ma tran va dai so tuyen tinh",
    "xay dung mo hinh hoc may tu du lieu",
    "xu ly ngon ngu tu nhien va van ban",
    "thuat toan tim kiem thong tin",
]


def tokenize(text: str) -> list[str]:
    return [word for word in text.lower().split() if word]


def build_bow_matrix(texts: list[str]) -> tuple[np.ndarray, list[str]]:
    vocab = sorted({word for text in texts for word in tokenize(text)})
    vocab_index = {word: idx for idx, word in enumerate(vocab)}
    matrix = np.zeros((len(texts), len(vocab)), dtype=float)
    for row, text in enumerate(texts):
        for word in tokenize(text):
            matrix[row, vocab_index[word]] += 1.0
    return matrix, vocab


def cosine_similarity_matrix(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    if y is None:
        y = x
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)
    x_norm[x_norm == 0] = 1.0
    y_norm[y_norm == 0] = 1.0
    return (x @ y.T) / (x_norm * y_norm.T)


def search_query(query: str, texts: list[str], top_k: int = 3) -> list[tuple[int, float, str]]:
    bow, vocab = build_bow_matrix(texts)
    vocab_index = {word: idx for idx, word in enumerate(vocab)}
    q_vec = np.zeros((1, len(vocab)), dtype=float)
    for word in tokenize(query):
        if word in vocab_index:
            q_vec[0, vocab_index[word]] += 1.0
    sims = cosine_similarity_matrix(q_vec, bow).ravel()
    order = np.argsort(-sims)
    return [(int(idx), float(sims[idx]), texts[idx]) for idx in order[:top_k]]





def load_image_gray(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if size is not None:
        image = image.resize(size)
    return np.asarray(image, dtype=float) / 255.0


def reconstruct_image(image_matrix: np.ndarray, k: int) -> np.ndarray:
    u, s, vt = np.linalg.svd(image_matrix, full_matrices=False)
    return (u[:, :k] * s[:k]) @ vt[:k, :]


def show_reconstructions(image_matrix: np.ndarray, ks: Iterable[int]) -> None:
    ks = list(ks)
    cols = len(ks) + 1
    plt.figure(figsize=(4 * cols, 4))
    plt.subplot(1, cols, 1)
    plt.title("Original")
    plt.imshow(image_matrix, cmap="gray")
    plt.axis("off")

    for i, k in enumerate(ks, start=2):
        plt.subplot(1, cols, i)
        plt.title(f"k={k}")
        plt.imshow(reconstruct_image(image_matrix, k), cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.show()





def lsa_coordinates(texts: list[str], n_components: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bow, _ = build_bow_matrix(texts)
    x_centered = bow - bow.mean(axis=0)
    u, s, vt = np.linalg.svd(x_centered, full_matrices=False)
    coords = u[:, :n_components] * s[:n_components]
    return x_centered, coords, vt


def plot_text_coordinates(texts: list[str], coords: np.ndarray) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(coords[:, 0], coords[:, 1], s=80)
    for idx, text in enumerate(texts):
        plt.annotate(str(idx + 1), (coords[idx, 0], coords[idx, 1]), xytext=(4, 4), textcoords="offset points")
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.axvline(0, color="gray", linewidth=0.8)
    plt.title("2D LSA coordinates of texts")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.show()





def main() -> None:
    bow, vocab = build_bow_matrix(TEXTS)
    print("Bai 1 - vocabulary size:", len(vocab))
    print("Bai 1 - matrix shape:", bow.shape)
    print("Bai 1 - top 3 for query 'hoc may python':")
    for idx, score, text in search_query("hoc may python", TEXTS, top_k=3):
        print(f"  {idx + 1}: score={score:.4f} | {text}")

    x = np.linspace(0, 1, 160)
    y = np.linspace(0, 1, 120)
    xv, yv = np.meshgrid(x, y)
    synthetic_image = np.clip(0.65 * xv + 0.35 * np.sin(8 * math.pi * yv) * 0.2 + 0.15, 0, 1)
    recon = reconstruct_image(synthetic_image, k=10)
    print("Bai 2A - synthetic image shape:", synthetic_image.shape)
    print("Bai 2A - reconstruction mse:", float(np.mean((synthetic_image - recon) ** 2)))

    x_centered, coords, vt = lsa_coordinates(TEXTS, n_components=2)
    print("Bai 2B - centered matrix shape:", x_centered.shape)
    print("Bai 2B - coords shape:", coords.shape)
    print("Bai 2B - first two singular vectors shape:", vt[:2].shape)


if __name__ == "__main__":
    main()
