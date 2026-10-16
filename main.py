import argparse
from src.loader import load_documents_from_folder
from src.chunker import create_document_chunks
from src.vector_store import VectorStoreManager
from src.rag_engine import GroundedRAGEngine


def run_ingestion(docs_folder: str, store: VectorStoreManager):
    print(f"\n🚀 Starting document ingestion from '{docs_folder}'...")
    segments = load_documents_from_folder(docs_folder)
    if not segments:
        print("⚠️ No valid documents (.pdf, .md, .txt) found.")
        return

    chunks = create_document_chunks(segments, chunk_size=1200, chunk_overlap=250)
    store.index_chunks(chunks)


def run_chat_loop(store: VectorStoreManager, top_k: int = 8, debug: bool = False):
    if store.collection.count() == 0:
        print("⚠️ Vector store is empty! Run `python main.py --ingest` first.")
        return

    rag = GroundedRAGEngine()
    print("\n========================================================")
    print("💬 Advanced RAG Assistant (Type 'exit' or 'quit' to stop)")
    print(f"📚 Knowledge Base: {store.collection.count()} chunks indexed (Hybrid Search: Dense + BM25)")
    print("========================================================\n")

    while True:
        try:
            question = input("\n🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("👋 Goodbye!")
            break

        # 1. Hybrid Retrieval (Dense Vector + BM25 Keyword Search with RRF)
        retrieved = store.retrieve(question, top_k=top_k)

        if debug:
            print("\n🔎 [DEBUG] Top Retrieved Contexts:")
            for idx, c in enumerate(retrieved, 1):
                meta = c.get("metadata", {})
                print(f"  {idx}. [{meta.get('level', 'Doc')}] {meta.get('section', 'General')} "
                      f"(Page {meta.get('page', '?')}) | RRF Score: {c.get('rrf_score', 'N/A')}")
                snippet = c['text'].replace('\n', ' ')[:140]
                print(f"     Preview: {snippet}...\n")

        # 2. Generate grounded answer
        result = rag.answer_question(question, retrieved)

        print(f"\n🤖 Assistant:\n{result['answer']}")
        if result["sources"]:
            print(f"\n📌 Verified Sources: {', '.join(result['sources'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Document RAG (Hybrid Dense + BM25)")
    parser.add_argument("--docs", type=str, default="./documents", help="Path to documents folder")
    parser.add_argument("--ingest", action="store_true", help="Ingest and re-index documents")
    parser.add_argument("--debug", action="store_true", help="Display retrieved chunk details and match scores")
    parser.add_argument("--top_k", type=int, default=8, help="Number of context chunks to retrieve (default: 8)")
    args = parser.parse_args()

    vector_store = VectorStoreManager(persist_dir="./chroma_db")

    if args.ingest:
        run_ingestion(args.docs, vector_store)

    run_chat_loop(vector_store, top_k=args.top_k, debug=args.debug)
