"""乳腺超声扫查知识库 — 关键词匹配 + 字符 n-gram 相似度搜索"""

import re
import os
import yaml
from typing import Dict, List, Optional


class KnowledgeBase:
    """加载 YAML 知识库，提供关键词匹配和 n-gram 相似度搜索"""

    def __init__(self, yaml_path: str = None):
        if yaml_path is None:
            yaml_path = os.path.join(os.path.dirname(__file__), "knowledge_base.yml")

        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        self.entries: List[Dict] = raw or []
        self._build_keyword_index()

        # 匹配阈值（低于此分数返回兜底回复）
        self.threshold = 0.08

    def _build_keyword_index(self):
        """构建关键词 → 条目索引的映射"""
        self.keyword_map: Dict[str, List[int]] = {}
        for i, entry in enumerate(self.entries):
            for kw in entry.get("keywords", []):
                kw_lower = kw.lower()
                if kw_lower not in self.keyword_map:
                    self.keyword_map[kw_lower] = []
                self.keyword_map[kw_lower].append(i)

    def _tokenize(self, text: str) -> List[str]:
        """简单中文/英文分词：按标点/空格切分，生成字符级 2-gram"""
        # 按非字母数字/非中文字符切分
        tokens = re.findall(r"[一-鿿_a-zA-Z0-9]+", text.lower())
        ngrams = []
        for token in tokens:
            # 字符级 2-gram
            for i in range(len(token) - 1):
                ngrams.append(token[i : i + 2])
            # 单字符也保留（处理英文缩写等）
            if len(token) == 1:
                ngrams.append(token)
        return ngrams

    def _jaccard_similarity(self, query: str, target: str) -> float:
        """计算两个字符串的 Jaccard 相似度（基于 2-gram）"""
        q_grams = set(self._tokenize(query))
        t_grams = set(self._tokenize(target))
        if not q_grams or not t_grams:
            return 0.0
        intersection = q_grams & t_grams
        union = q_grams | t_grams
        return len(intersection) / len(union)

    def search(self, query: str) -> Dict:
        """
        搜索最佳匹配答案
        返回: {"question": str, "answer": str, "score": float}
        """
        if not query or not query.strip():
            return {
                "question": "",
                "answer": "Please type a question.",
                "score": 1.0,
            }

        query_lower = query.strip().lower()

        # ── 第 1 层：关键词精确匹配 ──────────────────────────────
        keyword_hits: Dict[int, int] = {}
        for kw, indices in self.keyword_map.items():
            if kw in query_lower:
                for idx in indices:
                    keyword_hits[idx] = keyword_hits.get(idx, 0) + 1

        if keyword_hits:
            # 返回命中关键词最多的条目
            best_idx = max(keyword_hits, key=keyword_hits.get)
            entry = self.entries[best_idx]
            return {
                "question": entry["question"],
                "answer": entry["answer"],
                "score": 1.0,
            }

        # ── 第 2 层：n-gram Jaccard 相似度 ────────────────────────
        best_score = 0.0
        best_idx = 0

        for i, entry in enumerate(self.entries):
            # 对 keywords 和 question 分别计算相似度，取最大值
            kw_scores = [
                self._jaccard_similarity(query_lower, kw.lower())
                for kw in entry.get("keywords", [])
            ]
            q_score = self._jaccard_similarity(query_lower, entry["question"].lower())
            score = max(kw_scores + [q_score]) if (kw_scores or q_score > 0) else 0.0

            if score > best_score:
                best_score = score
                best_idx = i

        # ── 阈值判断 ────────────────────────────────────────────
        if best_score >= self.threshold:
            entry = self.entries[best_idx]
            return {
                "question": entry["question"],
                "answer": entry["answer"],
                "score": best_score,
            }

        # ── 兜底回复 ────────────────────────────────────────────
        return {
            "question": "",
            "answer": (
                "Sorry, I don't have an answer for that yet. "
                "Try rephrasing your question, for example:<br><br>"
                "• \"What is BI-RADS?\"<br>"
                "• \"How to tell benign from malignant lesions?\"<br>"
                "• \"What is the scanning order for breast ultrasound?\"<br>"
                "• \"How to hold the ultrasound probe?\"<br>"
                "• \"What are the anatomical layers of the breast?\"<br><br>"
                "For further help, please consult your instructor or reference materials."
            ),
            "score": best_score,
        }

    def get_all_questions(self) -> List[str]:
        """返回所有知识条目的标准问法列表"""
        return [e["question"] for e in self.entries]
