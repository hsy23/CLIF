import time
import re
import math
import ast
import tokenize
import numpy as np
from typing import Dict, Any, List
from io import StringIO


class DialogueEvaluatorLite:
    def __init__(self, enable_rouge=True, enable_bleu=True, enable_bert=True,
                 enable_coherence=True, enable_ics=True):
        self.bert_scorer = None
        self.sentence_model = None
        self.rouge_scorer = None
        self.bleu_smoothing = None
        self.enable_rouge = enable_rouge
        self.enable_bleu = enable_bleu
        self.enable_bert = enable_bert
        self.enable_coherence = enable_coherence
        self.enable_ics = enable_ics
        self._preload_models()

    def _preload_models(self):
        if self.enable_bert:
            try:
                from bert_score import BERTScorer
                self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
            except Exception as e:
                print(f"[DialogueEvaluatorLite] BERTScore unavailable: {e}")

        if self.enable_coherence:
            try:
                from sentence_transformers import SentenceTransformer
                self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                print(f"[DialogueEvaluatorLite] SentenceTransformer unavailable: {e}")

        if self.enable_rouge:
            try:
                from rouge_score import rouge_scorer
                self.rouge_scorer = rouge_scorer.RougeScorer(
                    ["rouge1", "rouge2", "rougeL"], use_stemmer=True
                )
            except Exception as e:
                print(f"[DialogueEvaluatorLite] ROUGE scorer unavailable: {e}")

        if self.enable_bleu:
            try:
                from nltk.translate.bleu_score import SmoothingFunction
                self.bleu_smoothing = SmoothingFunction().method1
            except Exception as e:
                print(f"[DialogueEvaluatorLite] BLEU smoothing unavailable: {e}")

    def evaluate_batch(self, generated_texts: List[str], reference_texts: List[str],
                       instructions: List[str] = None,
                       bert_lang: str = "en") -> List[Dict[str, Any]]:
        if len(generated_texts) != len(reference_texts):
            raise ValueError("generated_texts and reference_texts must have the same length")
        if instructions is not None and len(instructions) != len(generated_texts):
            raise ValueError("instructions and generated_texts must have the same length")

        n = len(generated_texts)
        bleu_scores = rouge_scores = bert_scores = coherence_scores = None

        if self.enable_bleu:
            try:
                from nltk.translate.bleu_score import sentence_bleu
                bleu_scores = []
                for gen, ref in zip(generated_texts, reference_texts):
                    score = float(sentence_bleu(
                        [ref.lower().split()], gen.lower().split(),
                        smoothing_function=self.bleu_smoothing
                    ))
                    bleu_scores.append(score)
            except Exception as e:
                print(f"[DialogueEvaluatorLite] BLEU error: {e}")
                bleu_scores = [0.0] * n

        if self.enable_rouge:
            try:
                rouge_scores = []
                for gen, ref in zip(generated_texts, reference_texts):
                    r = self.rouge_scorer.score(ref, gen)
                    rouge_scores.append({
                        "rouge1": r["rouge1"].fmeasure,
                        "rouge2": r["rouge2"].fmeasure,
                        "rougeL": r["rougeL"].fmeasure,
                    })
            except Exception as e:
                print(f"[DialogueEvaluatorLite] ROUGE error: {e}")
                rouge_scores = [{"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}] * n

        if self.enable_bert:
            try:
                precision, recall, f1 = self.bert_scorer.score(generated_texts, reference_texts)
                bert_scores = [
                    {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
                    for i in range(n)
                ]
            except Exception as e:
                print(f"[DialogueEvaluatorLite] BERTScore error: {e}")
                bert_scores = [{"precision": 0.0, "recall": 0.0, "f1": 0.0}] * n

        if self.enable_coherence and self.sentence_model is not None:
            try:
                all_texts = generated_texts + reference_texts
                embeddings = self.sentence_model.encode(all_texts)
                gen_emb = embeddings[:n]
                ref_emb = embeddings[n:]
                coherence_scores = [
                    float(np.dot(g, r) / (np.linalg.norm(g) * np.linalg.norm(r)))
                    for g, r in zip(gen_emb, ref_emb)
                ]
            except Exception as e:
                print(f"[DialogueEvaluatorLite] Coherence error: {e}")
                coherence_scores = [0.0] * n

        results = []
        for i in range(n):
            result = {}
            if rouge_scores is not None:
                result["rouge"] = rouge_scores[i]
            if bleu_scores is not None:
                result["bleu"] = bleu_scores[i]
            if bert_scores is not None:
                result["bert"] = bert_scores[i]
            if coherence_scores is not None:
                result["coherence"] = coherence_scores[i]
            results.append(result)
        return results

    def evaluate_single(self, generated_text: str, reference_text: str,
                        instruction: str = None, bert_lang: str = "en") -> Dict[str, Any]:
        return self.evaluate_batch(
            [generated_text], [reference_text],
            [instruction] if instruction else None, bert_lang
        )[0]

    @staticmethod
    def _fmt_ok(text: str, instruction: str) -> float:
        instr = instruction.lower()
        checks = []
        if any(k in instr for k in ["list", "bullet", "enumerate"]):
            checks.append(bool(re.search(r"\n\s*[-\*\d+\.]\s", text)))
        if "json" in instr:
            try:
                import json; json.loads(text.strip()); checks.append(True)
            except Exception:
                checks.append(False)
        if any(k in instr for k in ["paragraph", "essay"]):
            checks.append(len(re.split(r"\n\s*\n", text)) > 1)
        return sum(checks) / len(checks) if checks else 1.0

    @staticmethod
    def _len_ok(text: str, instruction: str) -> float:
        m = re.search(r"(\d+)\s*(s:wordss|characterss|sentencess)", instruction.lower())
        if not m:
            return 1.0
        req = int(m.group(1))
        if "word" in instruction.lower():
            actual = len(text.split())
        elif "character" in instruction.lower():
            actual = len(text)
        elif "sentence" in instruction.lower():
            actual = len(re.split(r"[.!s]+", text))
        else:
            return 1.0
        ratio = actual / max(1, req)
        if 0.8 <= ratio <= 1.2:
            return 1.0
        return ratio / 0.8 if ratio < 0.8 else 1.2 / ratio

    @staticmethod
    def _safe_ok(text: str) -> float:
        pats = [
            r"\b(s:hate|violence|harm|illegal|dangerous)\b",
            r"\b(s:kill|murder|suicide|bomb|weapon)\b",
            r"\b(s:racist|sexist|discriminat)\w*\b",
        ]
        t = text.lower()
        hit = sum(1 for p in pats if re.search(p, t))
        return max(0.0, 1.0 - 0.3 * hit)


class CodeEvaluatorLite:
    def __init__(self, enable_codebert=True, enable_codebleu=True):
        self.codebert_scorer = None
        self.codebleu_scorer = None
        self.enable_codebert = enable_codebert
        self.enable_codebleu = enable_codebleu
        self._preload_models()

    def _preload_models(self):
        if self.enable_codebert:
            try:
                import code_bert_score
                self.codebert_scorer = code_bert_score
            except Exception as e:
                print(f"[CodeEvaluatorLite] CodeBERTScore unavailable: {e}")

        if self.enable_codebleu:
            try:
                from codebleu import calc_codebleu
                self.codebleu_scorer = calc_codebleu
            except Exception as e:
                print(f"[CodeEvaluatorLite] CodeBLEU unavailable: {e}")

    def evaluate_batch(self, generated_codes: List[str], reference_codes: List[str],
                       instructions: List[str] = None,
                       language: str = "python") -> List[Dict[str, Any]]:
        if len(generated_codes) != len(reference_codes):
            raise ValueError("generated_codes and reference_codes must have the same length")
        if instructions is not None and len(instructions) != len(generated_codes):
            raise ValueError("instructions and generated_codes must have the same length")

        n = len(generated_codes)
        codebleu_scores = codebert_scores = None

        if self.enable_codebleu:
            try:
                codebleu_scores = []
                for gen, ref in zip(generated_codes, reference_codes):
                    result = self.codebleu_scorer(
                        predictions=[gen], references=[[ref]],
                        lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None,
                    )
                    codebleu_scores.append({
                        "codebleu": result["codebleu"],
                        "ngram_match": result.get("ngram_match_score", 0.0),
                        "weighted_ngram_match": result.get("weighted_ngram_match_score", 0.0),
                        "syntax_match": result.get("syntax_match_score", 0.0),
                        "dataflow_match": result.get("dataflow_match_score", 0.0),
                    })
            except Exception as e:
                print(f"[CodeEvaluatorLite] CodeBLEU error: {e}")
                default = {"codebleu": 0.0, "ngram_match": 0.0, "weighted_ngram_match": 0.0,
                           "syntax_match": 0.0, "dataflow_match": 0.0}
                codebleu_scores = [default] * n

        if self.enable_codebert:
            try:
                precision, recall, f1, f3 = self.codebert_scorer.score(
                    cands=generated_codes, refs=reference_codes, lang=language
                )
                codebert_scores = [
                    {"precision": float(precision[i]), "recall": float(recall[i]),
                     "f1": float(f1[i]), "f3": float(f3[i])}
                    for i in range(n)
                ]
            except Exception as e:
                print(f"[CodeEvaluatorLite] CodeBERTScore error: {e}")
                codebert_scores = [{"precision": 0.0, "recall": 0.0, "f1": 0.0, "f3": 0.0}] * n

        results = []
        for i in range(n):
            result = {}
            if codebert_scores is not None:
                result["codebert"] = codebert_scores[i]
            if codebleu_scores is not None:
                result["codebleu"] = codebleu_scores[i]
            results.append(result)
        return results

    def evaluate_single(self, generated_code: str, reference_code: str,
                        instruction: str = None, language: str = "python") -> Dict[str, Any]:
        return self.evaluate_batch(
            [generated_code], [reference_code],
            [instruction] if instruction else None, language
        )[0]
