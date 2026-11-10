# Indonesian Regulation Q&A (RAG)

A question-answering assistant for Indonesian government regulations. It retrieves the relevant pages from regulation PDFs (including scanned ones) and asks Gemini to answer **only** from those pages, citing the document and page. If the documents do not contain the answer, it says so instead of guessing.

The interface is in English. The regulations, the questions and the answers are in Indonesian, because that is the language of the documents and of the people who need them.

## Demo

![demo interface](docs_assets/demo.png)

## Results

Evaluated on a hand-checked set of 30 answerable questions (retrieval) and 40 questions in total (answer behaviour).
The corpus is 5 regulation PDFs, 215 pages. Each row of the table changes one thing.

### Retrieval: is the right page in the top 3?

| Version | Change | Hit@1 | Hit@3 | Precision@3 | MRR |
| ------- | ------ | ----- | ----- | ----------- | --- |
| v1_baseline | `paraphrase-multilingual-MiniLM`, page-level chunks, one PDF indexed twice | 12/30 (0.400) | 17/30 (0.567) | 0.211 | 0.472 |
| v2_dedupe | removed the duplicate PDF | 12/30 (0.400) | 17/30 (0.567) | 0.189 | 0.472 |
| **v3_embedding** | **`multilingual-e5-small`** (retrieval-trained), exact vector search | **19/30 (0.633)** | **27/30 (0.900)** | **0.300** | **0.739** |

- Removing the duplicate changed no rank metric (the same 13 questions missed), so duplicates were a hygiene problem, not the cause of the weak retrieval. Swapping the embedding model       took Hit@3 from 17 to 27 questions and Hit@1 from 12 to 19. The largest change is on Permen Komdigi 9/2026 (11 questions): 3/11 hits with the old model, 11/11 with the new one.
- Per-document results and the remaining 3 misses are in `results/eval_v3_embedding.csv`. Of the two questions that depend on the scanned appendix, one is answered and one stays a miss in   every version: Tesseract dropped the numbers, so that answer is not in the index.
- v1 and v2 were first measured with Chroma's default approximate search. v2 was re-measured with exact search (`v2_exact`, old model) and gave identical results (12/30, 17/30, MRR          0.472), so the gain in v3 comes from the embedding model, not from the search method.

### Answer behaviour: answers when it should, refuses when it should (v1 pipeline)

30 answerable and 10 unanswerable questions (5 related to the domain, 5 unrelated), Gemini 2.5 Flash.

|  | System refuses | System answers |
| --- | --- | --- |
| **Should refuse** | 9 | 1 |
| **Should answer** | 10 | 20 |

Accuracy 0.725, refusal precision 0.474, refusal recall 0.900, F1 0.621. **Hallucination rate 0.10** (1 of 10):
the other 9 unanswerable questions were refused, including trivia such as the capital of France. **False refusal rate 0.33** (10 of 30), and all 10 false refusals are retrieval misses: the model refused only when the right page was not retrieved, and it answered every question whose page was retrieved. This was measured before the embedding change, so the false refusal rate is expected to drop; it has not been re-measured (see Limitations).

### Targets (set after the v1 baseline, before any change)

| Target | Result | Status |
| ------ | ------ | ------ |
| Hit@3 ≥ 0.80 | 0.900 (v3) | met |
| MRR ≥ 0.60 | 0.739 (v3) | met |
| Hallucination rate ≤ 0.10 | 0.100 (v1) | met on the boundary |
| False refusal rate ≤ 0.10 | 0.333 (v1) | not re-measured on v3 |

## How the evaluation works

| Metric | Meaning |
| ------ | ------- |
| Hit@1 | the top retrieved chunk contains the expected passage (like accuracy) |
| Hit@3 | the expected passage is among the top 3 chunks (like recall; one correct passage per question) |
| Precision@3 | share of the 3 chunks that contain it (about 0.33 at best with one correct chunk) |
| MRR | mean of 1/rank of the first correct chunk |
| Refusal confusion matrix | positive class = "should refuse"; gives accuracy, precision, recall, F1, hallucination and false refusal rates |

A chunk counts as correct when a distinctive phrase from the source passage (a number, a duration, a term) appears in it. `eval_set.csv` holds the questions: 28 text questions, 2 that depend on scanned appendix pages, and 10 that the documents cannot answer.

```
python evaluate.py v3_embedding             # retrieval metrics, no API calls
python evaluate.py v3_embedding --answers   # answer vs refuse, calls Gemini (resumable)
```
Results go to `results/` (`summary.csv`, `summary_answers.csv` and one CSV per run).

## How it works

```
docs/*.pdf -> read page by page -> pages with no/garbled text go through OCR (Tesseract) -> one chunk per page
           -> multilingual-e5-small embeddings -> ChromaDB (rebuilt automatically when PDFs or model change)
question  -> top-3 pages -> prompt "answer only from these pages, otherwise refuse" -> Gemini -> answer + sources
```

Design notes:
- **OCR per page, not per file.** A PDF with native-text pages plus a scanned appendix would otherwise lose the appendix.
- **The index fingerprints its inputs** (PDF hashes and the embedding model name), so changing either rebuilds it. The evaluation script refuses to run against an index built with a         different model.
- **Exact vector search.** Chroma's default approximate search (HNSW, `search_ef=10`) can miss true nearest neighbours. Two builds of the same index scored 27/30 and 26/30 Hit@3 under it    (approximate search is the suspected cause), so the index is now created with `search_ef` above the chunk count, which makes search effectively exact and the metrics reproducible:
  v3 gave 27/30 on two separate runs, and v2 re-measured with exact search matched its earlier numbers.
- **Rate limits.** The Gemini caller separates per-minute limits (retry) from daily quota and key errors (stop at once), and the evaluation resumes where it stopped.

## Setup

```
git clone https://github.com/rizkiamandaa/RAG-Regulasi-Indonesia.git
cd RAG-Regulasi-Indonesia
pip install -r requirements.txt
```

1. **OCR tools.** Install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) with the Indonesian language data
   (`ind.traineddata`) and [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) (Windows), or
   `brew install tesseract tesseract-lang poppler` (macOS), or
   `sudo apt install tesseract-ocr tesseract-ocr-ind poppler-utils` (Debian/Ubuntu).
2. **Configuration.** Copy `.env.example` to `.env` and fill it in. Get a Gemini key at
   [Google AI Studio](https://aistudio.google.com/apikey). On Windows set `TESSERACT_PATH` (to `tesseract.exe`) and
   `POPPLER_PATH` (to the folder containing `pdftoppm.exe`). Never commit `.env`.
3. **Documents.** Put the regulation PDFs in `docs/`.
4. **Run.** `python -m streamlit run app.py`. The first run builds the index (slow when pages need OCR and the embedding
   model has to be downloaded); later runs start immediately.

## Repository

```
app.py          Streamlit UI
rag_core.py     reading PDFs and OCR, index, prompt, Gemini calls
evaluate.py     retrieval and answer-behaviour evaluation
eval_set.csv    the 40 evaluation questions
results/        one CSV per evaluated version, plus summaries
docs/           regulation PDFs
```

## Limitations

- **Small test set.** 30 answerable questions: one question moves Hit@3 by 0.033. Counts are reported next to ratios.
- **In-sample.** The questions were drafted with AI assistance and checked against the PDFs by the author. Changes were chosen after seeing the baseline failures on this same set (only      one alternative embedding model was tried), and there is no held-out set. Two of the five documents have no questions and act as distractors.
- **The model swap changed two things at once:** the training objective (retrieval vs sentence similarity) and the context window (512 vs, as far as I know, 128 tokens). The data cannot     separate them.
- **Answer behaviour was measured on the v1 pipeline only.** The free Gemini quota allowed roughly 20 calls per day, so a full run (40 calls) takes two days. Re-measure with `--answers`     to get the current numbers.
- **Answer correctness and faithfulness were only spot-checked**, not measured systematically.
- **One unanswerable question may be answerable:** "police emergency number" is covered by the 112 regulation (public-order emergencies), so the 0.10 hallucination rate may be overstated.
- **Scanned tables.** Tesseract with default settings dropped numeric cells in a scanned appendix table. A binarized image recovered 4 of 6 values in a manual test; this is not in the       pipeline.
- The stamped signature date and Berita Negara number of Permen 9/2026 are image overlays and are not searchable.

## Possible next steps

Hold out 10 new questions written from the two unquestioned documents, add image preprocessing for scanned tables, test smaller chunks or article-aware chunking, and re-run the answer-behaviour evaluation on the current pipeline.
