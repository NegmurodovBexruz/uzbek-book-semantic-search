import os
import json
import hashlib
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from chromadb import PersistentClient

from utils import read_docx_sections_clean, book_title_from_path

os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOKS_DIR = os.path.join(BASE_DIR, "books")
DB_DIR = os.path.join(BASE_DIR, "db")

COLLECTION = "books_extractive_sections"
EMBED_MODEL = "bge-m3"

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 250

MANIFEST_PATH = os.path.join(DB_DIR, "books_db.json")
SECTIONS_CACHE = os.path.join(DB_DIR, "sections.jsonl")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    if not os.path.exists(MANIFEST_PATH):
        return {}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(m: dict) -> None:
    os.makedirs(DB_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)


def append_sections_cache(book_title: str, sections: list):
    os.makedirs(DB_DIR, exist_ok=True)
    with open(SECTIONS_CACHE, "a", encoding="utf-8") as f:
        for s_idx, sec in enumerate(sections):
            f.write(json.dumps({
                "book_title": book_title,
                "heading": sec["heading"],
                "section_index": s_idx,
                "text": sec["text"]
            }, ensure_ascii=False) + "\n")


def rebuild_sections_cache_from_manifest(manifest: dict):
    """
    Kitob yangilansa eski sections.jsonl ichida eski matn qolib ketmasin.
    Shuning uchun (oddiy va ishonchli) cache'ni qayta yig'amiz.
    """
    if os.path.exists(SECTIONS_CACHE):
        os.remove(SECTIONS_CACHE)

    for fname in manifest.keys():
        fpath = os.path.join(BOOKS_DIR, fname)
        if not os.path.exists(fpath):
            continue
        title = book_title_from_path(fpath)
        sections = read_docx_sections_clean(fpath)
        append_sections_cache(title, sections)


def delete_book_from_collection(col, book_title: str):
    """
    Kitob yangilanganda eski chunklarni o'chirish.
    """
    # Chroma filtering delete
    col.delete(where={"book_title": book_title})


def main():
    os.makedirs(DB_DIR, exist_ok=True)

    client = PersistentClient(path=DB_DIR)
    col = client.get_or_create_collection(name=COLLECTION)
    emb = OllamaEmbeddings(model=EMBED_MODEL)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    manifest = load_manifest()

    files = [f for f in os.listdir(BOOKS_DIR) if f.lower().endswith(".docx")]
    if not files:
        raise FileNotFoundError("books papkada .docx yo‘q")

    changed = False

    for fname in files:
        path = os.path.join(BOOKS_DIR, fname)
        file_hash = sha256_file(path)
        old = manifest.get(fname, {}).get("sha256")

        if old == file_hash:
            print(f"⏭️ Skip (unchanged): {fname}")
            continue

        title = book_title_from_path(path)
        print(f"🧠 Indexing: {fname}")

        if fname in manifest:
            delete_book_from_collection(col, title)

        sections = read_docx_sections_clean(path)
        append_sections_cache(title, sections)

        texts, metas, ids = [], [], []

        for s_idx, sec in enumerate(sections):
            heading = sec["heading"]
            full_text = sec["text"]

            parts = splitter.split_text(f"[HEADING: {heading}]\n" + full_text)
            for j, part in enumerate(parts):
                if len(part) < 120:
                    continue

                cid = hashlib.sha1(f"{title}|sec{s_idx}|c{j}".encode("utf-8")).hexdigest()
                texts.append(part)
                metas.append({
                    "book_title": title,
                    "heading": heading,
                    "section_index": s_idx,
                    "chunk_index": j
                })
                ids.append(cid)

        vecs = emb.embed_documents(texts)
        col.add(ids=ids, documents=texts, metadatas=metas, embeddings=vecs)

        manifest[fname] = {"sha256": file_hash, "collection": COLLECTION}
        changed = True
        print(f"✅ Done: {fname} | chunks={len(texts)}")

    if changed:
        save_manifest(manifest)
        rebuild_sections_cache_from_manifest(manifest)
        print("✅ Manifest updated & sections cache rebuilt.")
    else:
        print("✅ No changes. Nothing indexed.")


if __name__ == "__main__":
    main()
