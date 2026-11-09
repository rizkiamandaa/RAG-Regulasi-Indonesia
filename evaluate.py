"""Evaluation harness for the regulation Q&A system.

Usage (run from the project root, with the Streamlit app stopped):
    python evaluate.py <tag>             # retrieval metrics; no API calls
    python evaluate.py <tag> --answers   # answer behaviour (answer vs refuse); calls Gemini, resumable

<tag> names the system version being measured (for example v3_embedding).
Results are written to results/ and summarised in results/summary*.csv (one row per tag).
"""
import sys

import torch.nn as nn

sys.modules["nn"] = nn  # compatibility shim, see app.py

import argparse
import datetime
import os
import re
import time

import pandas as pd

import rag_core

EVAL_SET = "eval_set.csv"
RESULTS_DIR = "results"
PAUSE_SECONDS = 7          # pause between Gemini calls (free tier: 10 requests/minute)
MAX_CONSECUTIVE_ERRORS = 3
ANSWERABLE_TYPES = ("text", "ocr")


# --------------------------------------------------------------------------- #
# Pure helpers (unit-testable)
# --------------------------------------------------------------------------- #
def normalize(text):
    """Lower-case, collapse whitespace, and unify '36 - 50' with '36-50'."""
    text = text.lower()
    text = re.sub(r"\s*-\s*", "-", text)
    return " ".join(text.split())


def score_retrieval(chunk_texts, keyword):
    """Return (rank of the first chunk containing the keyword, number of such chunks). Rank 0 = not found."""
    key = normalize(keyword)
    flags = [key in normalize(text) for text in chunk_texts]
    rank = next((i for i, hit in enumerate(flags, start=1) if hit), 0)
    return rank, sum(flags)


def confusion_metrics(pairs):
    """pairs: list of (should_refuse, refused). Positive class = 'should refuse'."""
    tp = sum(1 for should, did in pairs if should and did)
    fn = sum(1 for should, did in pairs if should and not did)
    fp = sum(1 for should, did in pairs if not should and did)
    tn = sum(1 for should, did in pairs if not should and not did)

    def ratio(a, b):
        return round(a / b, 3) if b else None

    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = round(2 * precision * recall / (precision + recall), 3)
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "accuracy": ratio(tp + tn, tp + fn + fp + tn),
        "precision": precision, "recall": recall, "f1": f1,
        "false_refusal_rate": ratio(fp, fp + tn),   # answerable question, system refused
        "hallucination_rate": ratio(fn, fn + tp),   # unanswerable question, system answered
    }


def upsert_summary(path, row):
    """Add a result row to a summary CSV, replacing an earlier row with the same tag."""
    new = pd.DataFrame([row])
    if os.path.exists(path):
        old = pd.read_csv(path)
        new = pd.concat([old[old["tag"] != row["tag"]], new], ignore_index=True)
    new.to_csv(path, index=False)


def load_eval_set():
    return pd.read_csv(EVAL_SET, encoding="utf-8-sig").fillna("")


# --------------------------------------------------------------------------- #
# Layer 1: retrieval (no API calls)
# --------------------------------------------------------------------------- #
def run_retrieval(tag):
    questions = load_eval_set()
    questions = questions[questions["type"].isin(ANSWERABLE_TYPES)]

    index, collection = rag_core.open_existing_index()
    retriever = index.as_retriever(similarity_top_k=rag_core.TOP_K)
    n_docs = len({m["file_name"] for m in collection.get(include=["metadatas"])["metadatas"]})

    rows = []
    for _, q in questions.iterrows():
        nodes = retriever.retrieve(q["question"])
        rank, n_relevant = score_retrieval([n.node.get_content() for n in nodes], q["expected_keyword"])
        top = nodes[0].node.metadata if nodes else {}
        rows.append({
            "document": q["document"], "type": q["type"], "question": q["question"],
            "rank": rank, "n_relevant_top_k": n_relevant,
            "top1": f"{top.get('file_name')} p.{top.get('page')}",
        })

    res = pd.DataFrame(rows)
    res["hit1"] = res["rank"] == 1
    res["hit"] = res["rank"] > 0
    res["rr"] = res["rank"].apply(lambda r: 1 / r if r else 0)
    res["precision"] = res["n_relevant_top_k"] / rag_core.TOP_K

    needs_ocr = res["type"] == "ocr"
    hit1, hit3 = res["hit1"].mean(), res["hit"].mean()
    precision, mrr = res["precision"].mean(), res["rr"].mean()
    ocr_hits, ocr_total = int(res.loc[needs_ocr, "hit"].sum()), int(needs_ocr.sum())

    print(f"[{tag}] questions: {len(res)} | documents: {n_docs} | chunks: {collection.count()} "
          f"| embeddings: {rag_core.EMBED_MODEL}")
    print(f"Hit@1       : {hit1:.3f} ({int(res['hit1'].sum())}/{len(res)})  # top chunk is correct")
    print(f"Hit@3       : {hit3:.3f} ({int(res['hit'].sum())}/{len(res)})  # = Recall@3 (one correct passage per question)")
    print(f"Precision@3 : {precision:.3f}  # mean share of correct chunks in the top 3 (max ~0.33 with one correct chunk)")
    print(f"MRR         : {mrr:.3f}")
    print(f"OCR-dependent questions (scanned appendix): {ocr_hits}/{ocr_total} hit")
    print("\nBy document:")
    print(res.groupby("document")[["hit1", "hit", "rr"]].mean().round(3))
    misses = res[~res["hit"]]
    print(f"\nMISSES ({len(misses)}):")
    print(misses[["document", "question", "top1"]].to_string() if len(misses) else "(none)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    res.to_csv(f"{RESULTS_DIR}/eval_{tag}.csv", index=False)
    upsert_summary(f"{RESULTS_DIR}/summary.csv", {
        "tag": tag, "date": datetime.date.today().isoformat(), "n_questions": len(res),
        "n_docs": n_docs, "n_chunks": collection.count(),
        "hit_at_1": round(hit1, 3), "hit_at_3": round(hit3, 3),
        "precision_at_3": round(precision, 3), "mrr": round(mrr, 3),
        "ocr_questions_hit": f"{ocr_hits}/{ocr_total}", "embed_model": rag_core.EMBED_MODEL,
    })
    print(f"\nSaved: {RESULTS_DIR}/eval_{tag}.csv and a row in {RESULTS_DIR}/summary.csv")


# --------------------------------------------------------------------------- #
# Layer 2: answer behaviour (calls Gemini; resumable)
# --------------------------------------------------------------------------- #
def run_answers(tag):
    questions = load_eval_set()
    index, _ = rag_core.open_existing_index()
    retriever = index.as_retriever(similarity_top_k=rag_core.TOP_K)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    answers_path = f"{RESULTS_DIR}/answers_{tag}.csv"

    # Resume: reuse questions that already succeeded, retry the others.
    done = {}
    if os.path.exists(answers_path):
        old = pd.read_csv(answers_path, encoding="utf-8-sig")
        old = old[old["api_ok"].astype(str).str.lower() == "true"]
        done = {(r["group"], r["question"]): r for r in old.to_dict("records")}
        print(f"Resuming: {len(done)} questions already answered successfully, skipping them.")
    print(f"Model: {rag_core.GEMINI_MODEL} | tag: {tag}")

    rows, consecutive_errors, aborted = [], 0, False
    for i, (_, q) in enumerate(questions.iterrows(), start=1):
        group = "answerable" if q["type"] in ANSWERABLE_TYPES else "unanswerable"
        key = (group, q["question"])
        if key in done:
            rows.append(done[key])
            print(f"[{i}/{len(questions)}] {group:12s} -> (already done) | {q['question'][:70]}")
            continue

        nodes = retriever.retrieve(q["question"])
        answer, ok = rag_core.call_gemini(rag_core.build_prompt(q["question"], nodes))
        refused = rag_core.is_refusal(answer) if ok else None
        rows.append({
            "group": group, "type": q["type"], "document": q["document"], "question": q["question"],
            "expected_keyword": q["expected_keyword"], "answer": answer, "refused": refused, "api_ok": ok,
            "sources": "; ".join(f"{n.metadata.get('file_name')} p.{n.metadata.get('page')}" for n in nodes),
        })
        status = "REFUSED" if refused else "ANSWERED" if ok else "ERROR"
        print(f"[{i}/{len(questions)}] {group:12s} -> {status} | {q['question'][:70]}")
        if not ok:
            print(f"      reason: {answer}")
        pd.DataFrame(rows).to_csv(answers_path, index=False, encoding="utf-8-sig")  # save progress

        consecutive_errors = 0 if ok else consecutive_errors + 1
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            aborted = True
            break
        time.sleep(PAUSE_SECONDS)

    df = pd.DataFrame(rows)
    df.to_csv(answers_path, index=False, encoding="utf-8-sig")

    if aborted:
        raise SystemExit(
            f"\nStopped after {MAX_CONSECUTIVE_ERRORS} consecutive API errors. Last reason:\n  {rows[-1]['answer']}\n"
            "Fix the cause (usually the daily quota) and run the same command again: finished questions are skipped."
        )
    failed = int((df["api_ok"].astype(str).str.lower() != "true").sum())
    if failed:
        raise SystemExit(f"\n{failed} question(s) still failed. Run the same command again to retry them. "
                         "No summary was written because the run is incomplete.")

    refused = df["refused"].map(lambda v: str(v).strip().lower() == "true")
    metrics = confusion_metrics(list(zip(df["group"] == "unanswerable", refused)))

    print(f"\n[{tag}] CONFUSION MATRIX (positive = should refuse)")
    print(f"{'':24s}{'System REFUSES':>16s}{'System ANSWERS':>16s}")
    print(f"{'Should refuse':24s}{'TP = ' + str(metrics['tp']):>16s}{'FN = ' + str(metrics['fn']):>16s}")
    print(f"{'Should answer':24s}{'FP = ' + str(metrics['fp']):>16s}{'TN = ' + str(metrics['tn']):>16s}")
    for label in ("accuracy", "precision", "recall", "f1", "false_refusal_rate", "hallucination_rate"):
        print(f"{label:20s}: {metrics[label]}")
    print("\nFalse refusals (answerable but refused):")
    for question in df[(df["group"] == "answerable") & refused]["question"]:
        print(" -", question[:100])
    print("Hallucinations (unanswerable but answered):")
    for question in df[(df["group"] == "unanswerable") & ~refused]["question"]:
        print(" -", question[:100])

    upsert_summary(f"{RESULTS_DIR}/summary_answers.csv", {
        "tag": tag, "date": datetime.date.today().isoformat(), "model": rag_core.GEMINI_MODEL, **metrics,
    })
    print(f"\nSaved: {answers_path} and a row in {RESULTS_DIR}/summary_answers.csv")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the regulation Q&A system.")
    parser.add_argument("tag", nargs="?", default="latest", help="name of the system version, e.g. v3_embedding")
    parser.add_argument("--answers", action="store_true", help="evaluate answer vs refuse behaviour (calls Gemini)")
    args = parser.parse_args()
    try:
        run_answers(args.tag) if args.answers else run_retrieval(args.tag)
    except RuntimeError as error:  # index missing / built with another embedding model
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()