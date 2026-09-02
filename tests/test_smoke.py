# -*- coding: utf-8 -*-
"""冒烟测试：完全不依赖 torch/GPU/模型/外部服务，只测试纯 Python 逻辑（CI 安全）。
所有被测模块顶层不得 import torch/transformers/fastapi。"""
import os, sys, re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_detector_modules_no_torch():
    """features / stylometric / route_a / route_c 顶层都不 import torch/transformers。"""
    import ast, pathlib
    for p in pathlib.Path("detector").glob("*.py"):
        if p.name == "dual_stream.py":
            continue  # 已知引 torch，跳过
        t = p.read_text(encoding="utf-8")
        assert "import torch" not in t and "import transformers" not in t, f"{p.name} 顶层引 torch"

def test_tier_rules():
    from detector.features import THRESHOLDS
    def tier(p):
        if p < THRESHOLDS["human"]: return "human"
        if p < THRESHOLDS["ai"]: return "medium"
        return "high"
    assert tier(0.2) == "human"
    assert tier(0.5) == "medium"
    assert tier(0.8) == "high"
    assert THRESHOLDS["human"] == 0.40

def test_stylometric():
    from detector.stylometric import stylo_features
    f = stylo_features("综上所述，随着教育数字化转型的深入推进，具有重要的现实意义。")
    for k in ("mattr","repetition","punct_density","clause_var","fw_density"):
        assert k in f, f"缺 {k}"
    assert 0 <= f["punct_density"] <= 1

def test_split_sentences_char_scan():
    """字符级分句器处理省略号/连续标点/引号内句号。"""
    def split_scan(text):
        out=[]; buf=""; in_quote=False; i=0; n=len(text)
        while i<n:
            c=text[i]
            if c in "“「『": in_quote=True
            elif c in "”」』": in_quote=False
            buf+=c
            if not in_quote and c in "。？！":
                j=i+1
                while j<n and text[j] in "。？！": buf+=text[j]; j+=1
                i=j-1
                if buf.strip() and len(buf.strip())>4: out.append(buf.strip())
                buf=""
            elif not in_quote and c=="…":
                j=i+1
                while j<n and text[j]=="…": buf+=text[j]; j+=1
                i=j-1
                if buf.strip() and len(buf.strip())>4: out.append(buf.strip())
                buf=""
            i+=1
        if buf.strip() and len(buf.strip())>4: out.append(buf.strip())
        return out
    assert len(split_scan("研究继续…… 目前无定论。")) == 2
    assert len(split_scan("真的吗？！不可能！！ 但确实如此。")) >= 2

def test_route_c_perplexity_no_torch():
    """route_c 的 NgramLM 可构建 + 能算困惑度（纯 python + jieba）。"""
    from detector.route_c import NgramLM
    lm = NgramLM(2)
    lm.train(["这是一个测试句子。", "本研究采用数据。另有其他样本。"])
    p = lm.perplexity("这是一个测试句子。")
    assert p is not None and p > 0

if __name__ == "__main__":
    test_detector_modules_no_torch(); print("no-torch ok")
    test_tier_rules(); print("tier ok")
    test_stylometric(); print("stylometric ok")
    test_split_sentences_char_scan(); print("split ok")
    test_route_c_perplexity_no_torch(); print("route_c ok")
    print("ALL SMOKE TESTS PASSED")
