# -*- coding: utf-8 -*-
"""AIGC 检测常驻服务（FastAPI，带完整实时日志）。
启动时加载双流模型常驻内存，提供 /detect /ingest /train /health。
每个请求都输出详细日志（时间/方法/路径/句数/统计流/深流/融合/耗时/三态/错误）。
日志实时打印到终端窗口，便于监控实际使用中的问题。
运行：python server.py [--host 127.0.0.1] [--port 9000]
"""
import os, sys, io, json, re, pickle, subprocess, threading, time, datetime
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

BASE = r"C:\Users\woshi\.dsh\aigc-detector"
SCRIPTS = os.path.join(BASE, "scripts")
LOG_DIR = os.path.join(BASE, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "service.log")
sys.path.insert(0, os.path.join(BASE))
from detector.route_c import score_text, NgramLM
from detector.stylometric import stylo_features
from detector.dual_stream import load_bert, bert_score_per_sentence, fuse, doc_score
try:
    from detector.features import TEMPLATE_PATTERNS_ZH
except Exception:
    TEMPLATE_PATTERNS_ZH = []

app = FastAPI(title="AIGC检测服务", version="1.0")

@app.exception_handler(Exception)
async def global_exc_handler(request, exc):
    """全局异常兜底：任何未捕获异常都返回 JSON，而非 500 HTML。"""
    try:
        log(f"  [异常] {type(exc).__name__}: {str(exc)[:200]}")
    except Exception:
        pass
    return JSONResponse({"error": f"内部错误: {type(exc).__name__}"}, status_code=500)

STATE = {"stat": None, "bert": None, "lm": None, "model_meta": None, "ready": False}
TRAIN_LOCK = threading.Lock()   # 重训互斥锁，防止 /ingest 与 /train 并发重训
TRAIN_RUNNING = False

def ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    line = f"[{ts()}] {msg}"
    # 终端窗口实时显示
    print(line, flush=True)
    # 同步追加到日志文件（供排查/监控）
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

# ---- 模型加载 ----
def load_models():
    from detector.route_c import score_text, NgramLM
    log("═══ 加载模型 ═══")
    p_cls = os.path.join(BASE, "models", "classifier.pkl")
    if os.path.exists(p_cls):
        with open(p_cls, "rb") as f:
            STATE["stat"] = pickle.load(f)
        log(f"  统计流 TF-IDF 加载 ✓  acc={STATE['stat'].get('acc'):.3f} auc={STATE['stat'].get('auc'):.3f}")
    else:
        log("  统计流 缺失 ✗")
    p_lm = os.path.join(BASE, "models", "n-gram-lm.pkl")
    if os.path.exists(p_lm):
        with open(p_lm, "rb") as f:
            STATE["lm"] = pickle.load(f)
        log("  路线C 语言模型 加载 ✓")
    log("  加载深度流 RoBERTa（后台异步，不阻塞启动）...")
    # 深度流异步加载：即使卡住/失败，服务先用统计流可用，就绪后热切换
    threading.Thread(target=_load_bert_async, daemon=True).start()
    STATE["ready"] = True
    if STATE["stat"]:
        STATE["model_meta"] = {"acc": STATE["stat"].get("acc"), "auc": STATE["stat"].get("auc"), "threshold": STATE["stat"].get("threshold")}
    log(f"  ═══ 服务就绪（统计流 {'✓' if STATE['stat'] else '✗'} | 深度流后台加载中）═══")

def _load_bert_async():
    """后台加载深度流，带超时容错。加载完成后热更新 STATE['bert']。"""
    try:
        STATE["bert"] = load_bert(device="cuda")
        log(f"  ═══ 深度流 RoBERTa 加载 {'✓' if STATE['bert'] else '✗'}（后台）═══")
    except Exception as e:
        log(f"  ═══ 深度流加载失败（降级为统计流）: {str(e)[:100]} ═══")
        STATE["bert"] = None

# ---- 工具 ----
# 正文边界标记：检测应只覆盖正文，排除其后的参考文献/致谢/附录（避免污染 AIGC 判定）
# 边界判定：标记词所在段落须短且以标记词开头（独立章节标题），避免正文中出现"参考文献[1]"等词被误判。
BODY_END_MARKERS = ["参考文献", "致谢", "致　谢", "致 谢", "附录", "附　录", "发表论文", "攻读学位"]

def is_body_end(t):
    """判断段落是否是正文结束的独立章节标题（参考文献/致谢/附录等）。"""
    t = t.strip()
    if not t or len(t) > 25:  # 标题类段落很短
        return False
    for m in BODY_END_MARKERS:
        if t.startswith(m) or m in t[:6]:  # 标记词在段首
            return True
    return False

def extract_body(paragraphs):
    """提取正文：找到正文结束标题（参考文献/致谢/附录）并只返回其前的正文。
    若找不到可靠边界，则返回全文。"""
    body = []
    cut = None
    for i, t in enumerate(paragraphs):
        if is_body_end(t):
            cut = i
            break
        body.append(t)
    text = "\n".join(body).strip()
    # 若截断后正文过少（可能误判或正文确实短），回退全文
    if cut is not None and len(text) < 100:
        return "\n".join(paragraphs)
    return text

def docx_ordered_text(path):
    """按文档顺序提取 docx 的段落+表格文本（避免漏掉表格内容）。"""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    from docx.oxml.ns import qn
    d = Document(path)
    items = []
    for child in d.element.body.iterchildren():
        if child.tag == qn('w:p'):
            t = Paragraph(child, d).text
            if t.strip(): items.append(t)
        elif child.tag == qn('w:tbl'):
            tb = Table(child, d)
            for row in tb.rows:
                row_text = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                if row_text: items.append(row_text)
    return items

def read_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        paras = docx_ordered_text(path)
        full = "\n".join(paras)
        text = extract_body(paras)
        # 自适应：若提取的正文 < 全文40%，说明截断点可能异常（如封面/目录先出现"参考文献"），
        # 回退用全文，优先保证不漏掉真正正文。
        if len(text.strip()) < max(100, len(full.strip()) * 0.4):
            text = full
        return text
    if os.path.exists(path) and os.path.getsize(path) < 2_000_000:
        return open(path, encoding="utf-8", errors="ignore").read()
    return path

def split_sentences(text):
    """智能分句：正确处理中文引号内句号、省略号、连续感叹/问号。
    修复：省略号(……)不分句、连续标点(？!！)丢失内容的问题。"""
    out = []
    buf = ""
    in_quote = False
    i = 0; n = len(text)
    while i < n:
        c = text[i]
        if c in "“「『":
            in_quote = True
        elif c in "”」』":
            in_quote = False
        buf += c
        if not in_quote and c in "。？！":
            j = i + 1
            while j < n and text[j] in "。？！":
                buf += text[j]; j += 1
            i = j - 1
            if buf.strip() and len(buf.strip()) > 4:
                out.append(buf.strip())
            buf = ""
        elif not in_quote and c == "…":
            j = i + 1
            while j < n and text[j] == "…":
                buf += text[j]; j += 1
            i = j - 1
            if buf.strip() and len(buf.strip()) > 4:
                out.append(buf.strip())
            buf = ""
        i += 1
    if buf.strip() and len(buf.strip()) > 4:
        out.append(buf.strip())
    return out

# ---- 端到端检测 ----
def detect_pipeline(text, top_k=20):
    t0 = time.time()
    sents = split_sentences(text)
    if not sents:
        return {"error": "无可检测句子"}
    stat = STATE["stat"]
    p_tf = None; t_tf = None; t_bert = None
    if stat:
        t_tf = time.time()
        X = stat["vec"].transform(sents)
        p_tf = stat["model"].predict_proba(X)[:, 1].tolist()
        t_tf = time.time() - t_tf
    p_bert = None
    if STATE["bert"]:
        t_bert = time.time()
        tok, model, dev = STATE["bert"]
        p_bert = bert_score_per_sentence(tok, model, dev, sents)
        t_bert = time.time() - t_bert
    if p_bert:
        fused = [fuse(a, b) for a, b in zip(p_tf or [0]*len(sents), p_bert)]
    else:
        fused = p_tf
    overall = doc_score(sents, fused) if fused else 0.0
    thr = stat.get("threshold", 0.5) if stat else 0.5
    ai_count = sum(1 for p in fused if p >= thr) if fused else 0
    if overall >= 0.5: state_l = "高度疑似AI生成"
    elif overall >= 0.35: state_l = "疑似AI（建议人工复核）"
    elif overall >= 0.2: state_l = "证据不足（倾向人类，存在少量AI痕迹）"
    else: state_l = "基本人类撰写"
    hi = []
    # 命中模板的句子（无论是否AI，都记录，便于人工判断）
    templated = []
    stat_avg = round(sum(p_tf)/max(len(p_tf),1), 4) if p_tf else None
    bert_avg = round(sum(p_bert)/max(len(p_bert),1), 4) if p_bert else None
    # 逐句详细日志（仅当句数不多或AI句/模板命中时打印，避免刷屏）
    detail = []
    if fused:
        idxs = sorted(range(len(sents)), key=lambda i: -fused[i])[:top_k]
        for i in idxs:
            if fused[i] >= thr:
                hits = [d for pat, w, d in TEMPLATE_PATTERNS_ZH if re.search(pat, sents[i])]
                hi.append({"sentence": sents[i][:200], "ai_prob": round(fused[i], 4), "templates": hits})
                detail.append((i, sents[i], fused[i], hits))
        for i, s in enumerate(sents):
            hits = [d for pat, w, d in TEMPLATE_PATTERNS_ZH if re.search(pat, s)]
            if hits:
                templated.append({"sentence": s[:120], "templates": hits})
    # 详细日志
    log(f"  [检测] 句数={len(sents)} 总耗时={time.time()-t0:.2f}s (统计流{t_tf:.2f}s 深流{t_bert:.2f}s) 统计流均值={stat_avg} 深流均值={bert_avg} 融合={overall:.3f} AI句={ai_count} → {state_l}")
    if detail:
        for i, s, p, hits in detail:
            log(f"    · 高AI句[{p:.2f}] {s[:60]}" + (f" 模板:{hits}" if hits else ""))
    if templated:
        for x in templated[:8]:
            log(f"    · 模板命中 {x['templates']} | {x['sentence'][:50]}")
    # 对齐真实AIGC报告口径的参数
    total_chars = len(text)
    # 报告参数对比：对照官方常见判定口径（PaperYY/知网）
    pct = overall * 100
    if pct < 40:
        verdict = "人类创作（0~40%）"
    elif pct < 60:
        verdict = "疑似AI生成（40~60%）"
    else:
        verdict = "AI生成（60~100%）"
    return {
        "overall_ai_prob": round(overall, 4),         # 对应报告"AI特征值"
        "ai_feature_rate": round(pct, 1),             # 百分数版，对齐报告口径
        "stat_prob": stat_avg,
        "bert_prob": bert_avg,
        "n_sentences": len(sents),                    # 对应报告"句子数"
        "total_chars": total_chars,                   # 对应报告"总字符数"
        "ai_chars": int(sum(len(s) for s, p in zip(sents, fused) if p >= thr)),  # 对应报告"AI特征字符数"
        "ai_sentence_count": ai_count,
        "state": state_l,
        "verdict": verdict,                            # 报告参数对比：档位判定
        "elapsed_ms": int((time.time()-t0)*1000),
        "top_ai_sentences": hi,
    }

@app.get("/health")
def health():
    log("  [健康检查] 请求")
    return {"ready": STATE["ready"], "stat": bool(STATE["stat"]), "bert": bool(STATE["bert"]), "model_meta": STATE["model_meta"]}

@app.post("/detect")
async def detect(req: Request):
    try:
        body = await req.json()
    except Exception:
        return JSONResponse({"error": "请求体必须是有效 JSON"}, status_code=400)
    if body is None:
        return JSONResponse({"error": "请求体为空"}, status_code=400)
    text = body.get("text")
    path = body.get("path")
    top_k = body.get("top_k", 20)
    if path:
        log(f"  [检测] 请求 path={path}")
        if not os.path.exists(path):
            log(f"  [检测] ✗ 路径不存在: {path}")
            return JSONResponse({"error": f"文件不存在: {path}"}, status_code=404)
        try:
            text = read_text(path)
        except Exception as e:
            log(f"  [检测] ✗ 读取文件失败: {e}")
            return JSONResponse({"error": f"读取文件失败: {e}"}, status_code=400)
    elif text:
        log(f"  [检测] 请求 text字符数={len(text)}")
    if not text or not str(text).strip():
        log("  [检测] ✗ 缺 text 或 path")
        return JSONResponse({"error": "缺少 text 或 path"}, status_code=400)
    res = detect_pipeline(str(text), top_k)
    if "error" in res:
        log(f"  [检测] ✗ {res['error']}")
        return JSONResponse(res, status_code=200)
    return res

@app.post("/detect_batch")
async def detect_batch(req: Request):
    """批量检测多篇稿件。body: {"paths": [file_path...] 或 "texts": [str...], "top_k": 5}
    返回每篇的检测报告。"""
    try:
        body = await req.json()
    except Exception:
        return JSONResponse({"error": "请求体必须是有效 JSON"}, status_code=400)
    paths = body.get("paths", [])
    texts = body.get("texts", [])
    items = body.get("items", [])   # 支持 [{path} 或 {text}] 混合
    top_k = body.get("top_k", 20)
    # 汇总所有待检测项
    jobs = []
    for p in paths:
        jobs.append({"path": p})
    for t in texts:
        jobs.append({"text": t})
    for it in items:
        jobs.append(it)
    if not jobs:
        return JSONResponse({"error": "需提供 paths/texts/items"}, status_code=400)
    results = []
    log(f"  [批量检测] {len(jobs)} 篇")
    for j in jobs:
        try:
            if j.get("path"):
                p = j["path"]
                if not os.path.exists(p):
                    results.append({"path": p, "error": "文件不存在"}); continue
                text = read_text(p)
                res = detect_pipeline(text, top_k)
                res["path"] = p; res["name"] = os.path.basename(p)
            elif j.get("text"):
                res = detect_pipeline(j["text"], top_k)
                res["text_snippet"] = j["text"][:50]
            results.append(res)
        except Exception as e:
            results.append({"error": str(e), "item": j})
    log(f"  [批量检测] 完成 {len(jobs)} 篇")
    return {"results": results, "n": len(results)}

@app.post("/ingest")
async def ingest(req: Request):
    try:
        body = await req.json()
    except Exception:
        return JSONResponse({"error": "请求体必须是有效 JSON"}, status_code=400)
    path = body.get("path")
    log(f"  [入库] 请求 path={path}")
    if not path or not os.path.exists(path):
        log(f"  [入库] ✗ 路径无效: {path}")
        return JSONResponse({"error": "缺少有效 path"}, status_code=400)
    # 先提取并打印报告参数（对齐真实AIGC报告口径），供日志查看
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("ingest_nr", os.path.join(SCRIPTS, "ingest_new_reports.py"))
        ingest_nr = importlib.util.module_from_spec(spec); spec.loader.exec_module(ingest_nr)
        meta = ingest_nr.extract_report_meta(path)
        log(f"  [入库报告] 平台={meta.get('platform')} | AI特征值={meta.get('ai_rate')}% | "
            f"总字符={meta.get('total_chars')} | AI字符={meta.get('ai_chars')} | "
            f"报告编号={meta.get('report_no')} | 篇名={meta.get('title','')[:35]} | 作者={meta.get('author','')[:12]}")
    except Exception as e:
        log(f"  [入库报告] 参数提取失败: {e}")
    try:
        r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "ingest_new_reports.py"), "--source-dir", os.path.dirname(path), "--target", path], capture_output=True, text=True, timeout=300)
        log(f"  [入库] 完成 rc={r.returncode} | {r.stdout[-200:] if r.stdout else r.stderr[-200:]}")
    except Exception as e:
        log(f"  [入库] ⚠ 异常 {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
    threading.Thread(target=auto_retrain, daemon=True).start()
    log("  [入库] 已自动触发重训")
    return {"ingested": path, "auto_train": "triggered"}

@app.post("/report_bug")
async def report_bug(req: Request):
    """智能体使用中发现BUG，写入 bug_tracker.jsonl 统计文件。
    body: {"description": "...", "severity": "high|med|low", "scene": "..."}
    """
    try:
        body = await req.json()
    except Exception:
        return JSONResponse({"error": "请求体必须是有效 JSON"}, status_code=400)
    desc = body.get("description", "")
    severity = body.get("severity", "med")
    scene = body.get("scene", "general")
    if not desc:
        return JSONResponse({"error": "缺少 description"}, status_code=400)
    return log_bug(desc, severity, scene)

def log_bug(desc, severity="med", scene="general"):
    """写入 bug_tracker 并记录日志。"""
    import datetime as _dt
    rec = {
        "id": f"BUG-{_dt.datetime.now().strftime('%Y%m%d%H%M%S')}",
        "time": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "severity": severity,
        "scene": scene,
        "description": desc,
        "status": "open",
    }
    try:
        with open(os.path.join(BASE, "bug_tracker.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        log(f"  [BUG上报] 写入失败: {e}")
    log(f"  [BUG上报] {rec['id']} severity={severity} scene={scene} | {desc[:60]}")
    return {"bug": rec}

@app.post("/train")
async def train(req: Request):
    log("  [重训] 请求触发")
    threading.Thread(target=auto_retrain, daemon=True).start()
    return {"train": "started"}

def auto_retrain():
    """重训统计流 + 热更新内存模型。用互斥锁防止并发重训竞争。"""
    global TRAIN_RUNNING
    if not TRAIN_LOCK.acquire(blocking=False):
        log(">>> [重训] 重训正在进行中，忽略本次触发（防并发）")
        return
    if TRAIN_RUNNING:
        log(">>> [重训] 重训已在跑，跳过")
        TRAIN_LOCK.release()
        return
    TRAIN_RUNNING = True
    log("═"*40)
    log(">>> [重训] 开始重训统计流...")
    try:
        r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "train_classifier.py")], capture_output=True, text=True, timeout=600)
        if r.returncode == 0:
            p_cls = os.path.join(BASE, "models", "classifier.pkl")
            with open(p_cls, "rb") as f:
                STATE["stat"] = pickle.load(f)
            STATE["model_meta"] = {"acc": STATE["stat"].get("acc"), "auc": STATE["stat"].get("auc"), "threshold": STATE["stat"].get("threshold")}
            log(f">>> [重训] ✓ 完成并热更新: acc={STATE['model_meta']['acc']:.3f} auc={STATE['model_meta']['auc']:.3f}")
            # 重训深流（WSL），若可用
            if os.path.exists(r"C:\Users\woshi\.dsh\aigc-detector\scripts\finetune_roberta_wsl.py"):
                log(">>> [重训] 请求WSL重训深度流...")
                try:
                    r2 = subprocess.run(["wsl", "--", "bash", "-c", ". /home/sebast/aigcenv/bin/activate && python /mnt/c/Users/woshi/.dsh/aigc-detector/scripts/finetune_roberta_wsl.py"], capture_output=True, text=True, timeout=1200)
                    log(f">>> [重训] 深流 rc={r2.returncode} | {r2.stdout[-120:] if r2.stdout else r2.stderr[-120:]}")
                    # 重新加载深流
                    STATE["bert"] = load_bert(device="cuda")
                    log(">>> [重训] 深流已重新加载 ✓")
                except Exception as e:
                    log(f">>> [重训] 深流异常 {e}")
        else:
            log(f">>> [重训] ✗ 失败: {r.stderr[-200:]}")
    except Exception as e:
        log(f">>> [重训] ⚠ 异常 {e}")
    finally:
        TRAIN_RUNNING = False
        TRAIN_LOCK.release()
    log("═"*40)

def find_free_port(start_port, max_attempt=50):
    """从 start_port 起找一个可用端口（socket 绑定测试），返回可用端口号。"""
    import socket
    for p in range(start_port, start_port + max_attempt):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((host, p))
            return p
        except OSError:
            continue  # 端口被占用，试下一个
        finally:
            s.close()
    return None

if __name__ == "__main__":
    host = "127.0.0.1"; port = 9000
    for i, a in enumerate(sys.argv):
        if a == "--host" and i+1 < len(sys.argv): host = sys.argv[i+1]
        if a == "--port" and i+1 < len(sys.argv): port = int(sys.argv[i+1])
    log("═══ AIGC 检测服务启动 ═══")
    # 单实例锁：PID 文件里若已有存活进程则退出，防止多实例竞争
    PID_FILE = os.path.join(BASE, "service.pid")
    try:
        if os.path.exists(PID_FILE):
            old_pid = int(open(PID_FILE, encoding="utf-8").read().strip())
            import subprocess as _sp
            chk = _sp.run(["tasklist", "/FI", f"PID eq {old_pid}"], capture_output=True, text=True, timeout=5)
            if str(old_pid) in chk.stdout:
                log(f"  已有服务实例运行 (PID {old_pid})，本次启动退出（防多实例）")
                sys.exit(0)
        with open(PID_FILE, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except Exception as e:
        log(f"  (单实例锁检查失败: {e})")
    # 端口冲突自动切换：从指定端口起，占用则 +1 递增
    used_port = find_free_port(port)
    if used_port is None:
        log(f"!! 从 {port} 起 {50} 个端口均被占用，无法启动")
        sys.exit(1)
    if used_port != port:
        log(f"  默认端口 {port} 被占用，已自动切换到端口 {used_port}")
    else:
        log(f"  使用端口 {port}")
    port = used_port
    # 把实际端口写入 last_port.txt（供智能体/调用方读取）
    try:
        with open(os.path.join(BASE, "last_port.txt"), "w", encoding="utf-8") as f:
            f.write(str(port))
    except Exception as e:
        log(f"  (写 last_port.txt 失败: {e})")
    log(f"  ═══ AIGC 检测服务启动 ═══")
    log(f"  地址: http://{host}:{port}")
    log("  接口: /detect /ingest /train /health")
    log("  实际端口已写入 last_port.txt")
    load_models()
    log(f"服务就绪 http://{host}:{port}，等待请求...")
    uvicorn.run(app, host=host, port=port, log_level="info", access_log=True)
