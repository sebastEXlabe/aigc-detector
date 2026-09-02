
import os, re, json
from bs4 import BeautifulSoup

def parse(fp):
    t = open(fp, encoding="utf-8", errors="ignore").read()
    soup = BeautifulSoup(t, "lxml")
    md = {}
    txt = soup.get_text(" ", strip=True)
    m = re.search(r"疑似率[：:]\s*([\d.]+)%", txt); md["total_rate"] = float(m.group(1)) if m else None
    m = re.search(r"字数[：:]\s*([\d,]+)", txt); md["words"] = int(m.group(1).replace(",","")) if m else None
    m = re.search(r"句子数[：:]\s*(\d+)", txt); md["n_sents"] = int(m.group(1)) if m else None
    m = re.search(r"段落数[：:]\s*(\d+)", txt); md["n_paras"] = int(m.group(1)) if m else None
    m = re.search(r"检测文献[：:]\s*AIGC文档检测（\s*(.*?)\s*）", txt); md["title"] = m.group(1).strip() if m else None
    sentences = []
    # Each em with class; look up the whole document text for the chance that follows
    for em in soup.find_all("em"):
        cls = (em.get("class") or [""])[0]
        if cls not in ("low","medium","high"): continue
        s_text = em.get_text(" ", strip=True)
        if not s_text: continue
        # The popover div is the next div sibling with class aigc-detection-chance-popover
        prob = None
        for sib in em.next_siblings:
            if getattr(sib, "name", None) == "div" and sib.get("class") and "aigc-detection-chance-popover" in sib.get("class"):
                sm = re.search(r"AI生成的可能性为[：:]\s*<i[^>]*>\s*([\d.]+)%", str(sib))
                if sm: prob = float(sm.group(1))
                break
        if prob is None:
            # fallback: search within next few siblings text
            for sib in list(em.next_siblings)[:8]:
                m2 = re.search(r"AI生成的可能性为[：:]\s*([\d.]+)%", str(sib))
                if m2: prob = float(m2.group(1)); break
        if prob is not None:
            sentences.append({"text": s_text, "level": cls, "prob": prob})
    paras = []
    for m in re.finditer(r"该段落可能为AI生成的概率为[：:]\s*([\d.]+)%", txt):
        paras.append(float(m.group(1)))
    md["sentences"] = sentences
    md["para_probs"] = paras
    return md

base = r"C:\\Users\\woshi\\.dsh\\aigc_reports\\raw"
out = []
for root, dirs, files in os.walk(base):
    for n in files:
        if n.lower().endswith(".html"):
            fp = os.path.join(root, n)
            try:
                t = open(fp, encoding="utf-8", errors="ignore").read()
                if "疑似率" not in t: continue
                md = parse(fp)
                if md["total_rate"] is not None:
                    md["source"] = os.path.basename(root)
                    out.append(md)
                    print(f"{str(md['title'])[:26]:26} rate={md['total_rate']}% sents={len(md['sentences'])} (rep {md['n_sents']})")
            except Exception as e:
                import traceback; print("ERR", fp, traceback.format_exc()[-400:])
with open(r"C:\\Users\\woshi\\.dsh\\aigc_reports\\parsed\\reports.jsonl", "w", encoding="utf-8") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False)+"\n")
print("\\nTOTAL:", len(out), " sents:", sum(len(r['sentences']) for r in out), " paras:", sum(len(r['para_probs']) for r in out))
