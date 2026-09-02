# -*- coding: utf-8 -*-
"""冒烟测试：单元逻辑（无需GPU）+ 深度流模块覆盖（需 CPU torch/transformers）。
CI 已装 torch(CPU)+transformers，可覆盖 dual_stream；但避免依赖 390MB LFS 深流模型，
用轻量随机模型验证推理路径。
"""
import os, sys, re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---- 纯逻辑测试（无 torch）----
def test_tier_rules():
    from detector.features import THRESHOLDS
    def tier(p):
        if p < THRESHOLDS["human"]: return "human"
        if p < THRESHOLDS["ai"]: return "medium"
        return "high"
    assert tier(0.2) == "human" and tier(0.5) == "medium" and tier(0.8) == "high"
    assert THRESHOLDS["human"] == 0.40

def test_stylometric():
    from detector.stylometric import stylo_features
    f = stylo_features("综上所述，随着教育数字化转型的深入推进，具有重要的现实意义。")
    for k in ("mattr","repetition","punct_density","clause_var","fw_density"):
        assert k in f, f"缺 {k}"
    assert 0 <= f["punct_density"] <= 1

def test_split_sentences_char_scan():
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

def test_route_c_perplexity():
    from detector.route_c import NgramLM
    lm = NgramLM(2)
    lm.train(["这是一个测试句子。", "本研究采用数据。另有其他样本。"])
    p = lm.perplexity("这是一个测试句子。")
    assert p is not None and p > 0

# ---- 深度流模块覆盖（需要 torch/transformers，CPU 足够）----
def test_dual_stream_import():
    """dual_stream 应能 import（torch+transformers 装了），且 fuse 逻辑正确。"""
    from detector.dual_stream import fuse, load_bert, bert_score_per_sentence
    # fuse 纯逻辑
    assert abs(fuse(0.5, 0.5) - 0.5) < 1e-6
    # 分歧偏统计流
    assert fuse(0.2, 0.8) < 0.5, "分歧时应偏向统计流(低值)"

def test_bert_score_lightweight():
    """用一个轻量随机模型验证 bert_score_per_sentence 推理路径（不依赖LFS大模型）。"""
    import torch
    from transformers import AutoTokenizer, BertForSequenceClassification, AutoConfig
    from detector.dual_stream import bert_score_per_sentence
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")  # 轻量英文词典
    model_cfg = AutoConfig.from_pretrained("bert-base-uncased", num_labels=2)
    model = BertForSequenceClassification(model_cfg)  # 随机初始化，无需下载权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 模型必须移到 device（bert_score_per_sentence 假定调用方已做）
    probs = bert_score_per_sentence(tok, model, device, ["This is a test.", "Another sentence."], batch=2)
    assert probs is not None and len(probs) == 2
    assert all(0 <= p <= 1 for p in probs)

if __name__ == "__main__":
    test_tier_rules(); print("tier ok")
    test_stylometric(); print("stylometric ok")
    test_split_sentences_char_scan(); print("split ok")
    test_route_c_perplexity(); print("route_c ok")
    test_dual_stream_import(); print("dual_stream import ok")
    test_bert_score_lightweight(); print("bert_score lightweight ok")
    print("ALL SMOKE TESTS PASSED")
