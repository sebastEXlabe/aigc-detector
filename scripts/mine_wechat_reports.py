# -*- coding: utf-8 -*-
"""微信缓存 PDF 检测报告挖掘（并行）：提取 原文+作者/标题/学号+分章节AI% → 索引。
输出 wechat_reports_index.jsonl。用法：python mine_wechat_reports.py [--limit 0]
"""
import os, sys, re, glob, json, collections, random
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
import fitz
W = r'D:\xwechat_files'
ZH = re.compile(r'[\u4e00-\u9fff]')
FIELD = re.compile(r'(姓名|题目|专业|学校|学号|班级|指导教师|学院)[：:]\s*([^\n]{1,30})')
DOCPCT = re.compile(r'(\d+)/(\d+)\s+\n?([\d.]+)%')
def docx_txt(fp):
    try:
        d=Document(fp); return "\n".join(p.text.strip() for p in d.paragraphs if p.text.strip())
    except Exception: return ""

def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--limit",type=int,default=0); a=ap.parse_args()
    pdfs=[f for f in glob.glob(os.path.join(W,'**','*.pdf'),recursive=True) if os.path.getsize(f)>20000]
    random.seed(5); random.shuffle(pdfs)
    if a.limit: pdfs=pdfs[:a.limit]
    print("缓存 PDF:", len(pdfs))
    from docx import Document
    out=[]
    for f in pdfs:
        try:
            doc=fitz.open(f); txt=''.join(p.get_text() for p in doc); doc.close()
            if len(ZH.findall(txt))<400: continue
            # 提取字段（报告头）
            fields={}
            for m in FIELD.finditer(txt):
                if m.group(1) not in fields: fields[m.group(1)]=m.group(2).strip()
            # 分章节 AI%(取最大的一处作指示)
            pcts=DOCPCT.findall(txt)
            docai = float(pcts[0][2]) if pcts else None
            # 关键词: 是否检测报告
            is_rep = bool(re.search(r'(报告单|全文对照|全文标明引文|AIGC|检测报告|原创性声明)', txt))
            if not is_rep and not fields: continue
            out.append({"path":f,"fields":fields,"doc_ai_pct":docai,"zh":len(ZH.findall(txt))})
        except Exception: continue
    outp=r"C:\Users\woshi\.dsh\aigc-detector\data\wechat_reports_index.jsonl"
    with open(outp,"w",encoding="utf-8") as fh:
        for x in out: fh.write(json.dumps(x,ensure_ascii=False)+"\n")
    with_field=sum(1 for x in out if x.get("fields"))
    print(f"=== 微信PDF报告(含原文)提取: {len(out)} 份 | 含作者/标题字段 {with_field} 份 ===")
    for x in out[:4]:
        f=x.get("fields",{}); print(f"  {f.get('姓名','')} | {f.get('专业','')} | {f.get('题目','')[:30]} | docAI%={x.get('doc_ai_pct')}")
    print("保存", outp)

if __name__=="__main__": main()
