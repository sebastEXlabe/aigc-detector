# -*- coding: utf-8 -*-
"""模型加载与真实推理测试：统计流 pkl + 深度流 LFS roberta_ft。
依赖 CI checkout 已拉取 LFS（models/roberta_ft/model.safetensors 存在）。
"""
import os, sys, pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def test_stat_model_loads():
    """统计流 classifier.pkl 能加载且含 vec/model/threshold。"""
    p = os.path.join(BASE, "models", "classifier.pkl")
    if not os.path.exists(p):
        import pytest; pytest.skip("classifier.pkl 不存在（未上传/未训练）")
    with open(p, "rb") as f:
        m = pickle.load(f)
    assert "vec" in m and "model" in m
    assert "threshold" in m and 0 <= m["threshold"] <= 1

def test_stat_model_predict():
    """统计流模型能对中文句子输出概率[0,1]。"""
    p = os.path.join(BASE, "models", "classifier.pkl")
    if not os.path.exists(p):
        import pytest; pytest.skip("classifier.pkl 不存在")
    with open(p, "rb") as f:
        m = pickle.load(f)
    X = m["vec"].transform(["综上所述，随着教育数字化转型的深入推进，具有重要的现实意义。"])
    proba = m["model"].predict_proba(X)[:, 1]
    assert 0 <= proba[0] <= 1

def test_deep_model_lfs_exists():
    """深流 LFS 模型文件存在（CI checkout lfs:true 会拉取）。"""
    # 若 LFS 未拉取（shallow），model.safetensors 可能是指针文件
    p = os.path.join(BASE, "models", "roberta_ft", "model.safetensors")
    assert os.path.exists(p), "roberta_ft/model.safetensors 不存在"
    sz = os.path.getsize(p)
    assert sz > 1000, f"model.safetensors 疑似 LFS 指针（{sz}字节），未拉取"

def test_deep_model_real_inference():
    """用真实 LFS roberta_ft 加载并推理一句（较慢，标 xfail 慢测试？不，跑通证明 LFS 可用）。"""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch.nn.functional as F
    path = os.path.join(BASE, "models", "roberta_ft")
    if not os.path.exists(os.path.join(path, "model.safetensors")):
        import pytest; pytest.skip("深流模型不存在")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); model.eval()
    inp = tok(["综上所述，随着教育数字化转型的深入推进。"], return_tensors="pt", truncation=True, max_length=128)
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad():
        out = model(**inp).logits
    p = F.softmax(out, -1)[:, 1].cpu().numpy()
    assert 0 <= p[0] <= 1
