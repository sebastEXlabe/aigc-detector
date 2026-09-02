
import fitz, os, re, json

def clean_title(name):
    n = os.path.basename(name)
    n = re.sub(r"\.pdf$", "", n, flags=re.I)
    n = re.sub(r"[_\-]+(全文报告|原文对照报告|AIGC检测报告|AIGC存档报告|AIGC_全文报告单|检查报告|检测报告).*$", "", n)
    n = re.sub(r"^免费_PDF打印版_AIGC检测报告_?\[?", "", n)
    n = re.sub(r"[\]\)]+$", "", n)
    return n.strip() or os.path.basename(name)

def extract_meta(pdf_path):
    meta = {"file": pdf_path, "ok": False, "platform": "unknown", "title": None, "ai_rate": None, "date": None, "has_yuanwen": False, "has_fragments": False, "sent_markers": 0, "pages": 0, "text_sample": ""}
    try:
        doc = fitz.open(pdf_path)
        meta["pages"] = len(doc)
        text = ""
        for i in range(min(3, len(doc))):
            text += doc[i].get_text()
        doc.close()
        meta["ok"] = True
    except Exception as e:
        meta["error"] = str(e)
        return meta
    if "cx.cnki.net" in text or ("知网" in text and "AIGC检测" in text):
        meta["platform"] = "zhihu"
    elif "PaperPass" in text or "PaperYY" in text:
        meta["platform"] = "paperpass"
    elif "原文对照" in text or "人工撰写占比" in text:
        meta["platform"] = "duizhao"
    elif "AIGC检测" in text:
        meta["platform"] = "aigc_detect"
    m = re.search(r"篇名[：:]\s*(.+)", text)
    meta["title"] = (m.group(1).strip() if m else clean_title(pdf_path))
    m = re.search(r"AI特征值[：:]\s*([\d.]+)%", text)
    if not m: m = re.search(r"AIGC总体疑似度[^0-9]{0,8}\s*([\d.]+)%", text)
    if not m: m = re.search(r"疑似度[：:]\s*([\d.]+)%", text)
    meta["ai_rate"] = float(m.group(1)) if m else None
    m = re.search(r"检测时间[：:]\s*([\d-]+ [\d:]+)", text)
    meta["date"] = m.group(1).strip() if m else None
    meta["has_yuanwen"] = "原文内容" in text
    meta["has_fragments"] = "片段" in text
    meta["sent_markers"] = len(re.findall(r"AI生成的可能性为|疑似AIGC生成|该句子为AI生成|人工撰写占比", text))
    meta["text_sample"] = text[:90]
    return meta

tests = [
  "C:/Users/woshi/Documents/WXWork/1688854381822881/Cache/File/2026-04/AIGC全文报告_基于python+vue技术高考志愿智能推荐系统的设计与实现.pdf",
  "D:/xwechat_files/wxid_uo6oalg1zkih22_59d1/msg/file/2025-05/免费_PDF打印版_AIGC检测报告_[王哲毕业设计（论文）1].pdf",
  "D:/xwechat_files/wxid_uo6oalg1zkih22_59d1/msg/file/2026-05/范雨竹+毕业论文二稿5.15_AIGC原文对照报告.pdf",
  "D:/xwechat_files/wxid_uo6oalg1zkih22_59d1/msg/file/2025-04/论文-AIGC检测报告-20250330.pdf",
  "C:/Users/woshi/Downloads/Q26061802/基于数智赋能与具身干预的中职德育新模式实践探索——以中职电商专业舞动赋能课程为例_AIGC_全文报告单.pdf",
]
for p in tests:
    m = extract_meta(p)
    print(json.dumps({k: m[k] for k in ["platform","title","ai_rate","date","has_yuanwen","has_fragments","sent_markers","pages"]}, ensure_ascii=False))
