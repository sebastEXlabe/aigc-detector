# -*- coding: utf-8 -*-
"""冒烟测试：不依赖 GPU/torch/模型，只测纯 Python 逻辑正确性（用于 CI）。"""
import os, sys, re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_detector_imports():
    """核心检测器模块应能 import 且不含需要 torch 的顶层调用。"""
    # features / stylometric / route_a / route_c 顶层都不该 import torch
    from detector.features import THRESHOLDS, TEMPLATE_PATTERNS_ZH
    from detector.stylometric import stylo_features
    from detector.route_a import per_sentence_features, document_features
    assert THRESHOLDS["human"] == 0.40
    assert len(TEMPLATE_PATTERNS_ZH) > 0

def test_split_sentences_char_scan():
    """字符级分句器正确处理省略号/连续标点。"""
    # 从 route_c 提取的分句逻辑（server.py 的，但此处独立复现避免加载 torch）
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

def test_stylometric():
    from detector.stylometric import stylo_features
    f = stylo_features("综上所述，随着教育数字化转型的深入推进，具有重要的现实意义。")
    for k in ("mattr","repetition","punct_density","clause_var","fw_density"):
        assert k in f, f"缺 {k}"
    assert 0 <= f["punct_density"] <= 1

def test_tier_rules():
    from detector.features import THRESHOLDS
    # 判定档位
    def tier(p):
        if p < THRESHOLDS["human"]: return "human"
        if p < THRESHOLDS["ai"]: return "medium"
        return "high"
    assert tier(0.2) == "human"
    assert tier(0.5) == "medium"
    assert tier(0.8) == "high"

def test_detect_aigc_help():
    r = os.popen(f'{sys.executable} "{os.path.join(os.path.dirname(__file__),"..","scripts","detect_aigc.py")}" 2>&1').read()
    assert "用法" in r

if __name__ == "__main__":
    test_detector_imports(); print("detector imports ok")
    test_split_sentences_char_scan(); print("split ok")
    test_stylometric(); print("stylometric ok")
    test_tier_rules(); print("tier ok")
    test_detect_aigc_help(); print("cli ok")
    print("ALL SMOKE TESTS PASSED")
