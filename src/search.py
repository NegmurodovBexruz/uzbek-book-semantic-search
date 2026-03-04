import os
import re
from typing import List, Tuple

os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"

from chromadb import PersistentClient
from langchain_ollama import OllamaEmbeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "db")

COLLECTION = "books_extractive_sections"
EMBED_MODEL = "bge-m3"

TOP_K = 30
MAX_RESULTS = 5

SENT_WINDOW = 2


def norm_text(s: str) -> str:
    s = s.lower()
    s = s.replace("ʻ", "'").replace("’", "'")
    s = re.sub(r"[^\w\s'-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def query_tokens(q: str) -> List[str]:
    qn = norm_text(q)

    # Variantlar: iymon/imon, tafviyz/tafviz (minimal)
    variants = {qn, qn.replace("iymon", "imon"), qn.replace("imon", "iymon"),
                qn.replace("tafviyz", "tafviz"), qn.replace("tafviz", "tafviyz")}

    toks = []
    stop = {"bu", "nima", "qanday", "qaysi", "va", "ham", "uchun", "bilan", "degani", "deyiladi"}
    for v in variants:
        toks.extend([t for t in v.split() if len(t) >= 3 and t not in stop])

    uniq = []
    for t in toks:
        if t not in uniq:
            uniq.append(t)
    return uniq


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.\?\!])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def extract_snippet(text: str, q_tokens: List[str], sent_window: int = 2) -> Tuple[str, int]:
    """
    Matndan query tokenlari uchragan sentence atrofidan snippet kesadi.
    Return: (snippet, hit_count). Agar topilmasa: ("", 0)
    """
    sents = split_sentences(text)
    if not sents:
        return "", 0

    best_i = -1
    best_hits = 0

    for i, s in enumerate(sents):
        ns = norm_text(s)
        hits = sum(1 for tok in q_tokens if tok in ns)
        if hits > best_hits:
            best_hits = hits
            best_i = i

    if best_i == -1 or best_hits == 0:
        return "", 0

    start = max(0, best_i - sent_window)
    end = min(len(sents), best_i + sent_window + 1)

    snippet = " ".join(sents[start:end]).strip()
    return snippet, best_hits


def main():
    client = PersistentClient(path=DB_DIR)
    col = client.get_collection(name=COLLECTION)
    emb = OllamaEmbeddings(model=EMBED_MODEL)

    print("✅ Ready. Collection:", COLLECTION, "| Count:", col.count())

    while True:
        q = input("\nSavol (exit=chiqish): ").strip()
        if not q:
            continue
        if q.lower() == "exit":
            break

        q_toks = query_tokens(q)
        if not q_toks:
            print("\nSavol juda umumiy. Kalit so'z bilan yozing (masalan: 'iymon shartlari').")
            continue

        qvec = emb.embed_query(q)
        res = col.query(
            query_embeddings=[qvec],
            n_results=TOP_K,
            include=["documents", "metadatas"]
        )

        docs = res["documents"][0]
        metas = res["metadatas"][0]

        if not docs:
            print("\nJavob topilmadi.")
            continue

        results = []
        for doc, meta in zip(docs, metas):
            snippet, hits = extract_snippet(doc, q_toks, sent_window=SENT_WINDOW)
            if hits == 0 or not snippet:
                continue

            results.append({
                "book_title": meta.get("book_title", ""),
                "heading": meta.get("heading", "") or "NO_HEADING",
                "hits": hits,
                "snippet": snippet
            })

        if not results:
            print("\nJavob topilmadi.")
            continue

        results.sort(key=lambda x: -x["hits"])
        results = results[:MAX_RESULTS]

        for r in results:
            print("\n------------------------------")
            print(f"kitob nomi: {r['book_title']}")
            print(f"heading: {r['heading']}")
            print(f"information: {r['snippet']}")
        print("\n------------------------------")


if __name__ == "__main__":
    main()
