# -*- coding: utf-8 -*-
"""改写语料(降AIGC)质量过滤：剔除参考文献/致谢/摘要/封面/签名等非正文句污染。
被 merge_rewrite_corpus.py 与训练前置共用，保证进入训练语料的都是真正可改写的正文陈述句。
"""
import re

# 参考文献条目：姓名+点+英文/中文+[J/M/D/C/EB/OL] 等
_REF = re.compile(r'\[\d+\]\s|\b\[[JMDCNEK]\]|\b\[EB/OL\]|\b\[J/OL\]|\b(20\d\d)[,，.]\d+.*?[:：]|^(参考文献|references?)\s*[:：]?', re.I)
# 封面/签名/声明/目录/摘要头/致谢/资助等
_META = re.compile(
    r'^(参考文献|致\s*谢|摘\s*要|目\s*录|content|abstract|keywords|作者简介|基金项目|收稿|录用|'
    r'本\s*人\s*签\s*名|指导教师签名|学位论文|原创性声明|知识产权声明|分类号|UDC|论文类别|学校代码|学\s*号|答辩日期)',
    re.I)
_AUTHLINE = re.compile(r'\s*[\u4e00-\u9fff]{2,4}\s*[，,]\s*[\u4e00-\u9fff]{2,4}(?:\s*[，,]\s*[\u4e00-\u9fff]{2,4}){0,4}\s*\n?$')
_TITLEBLK = re.compile(r'^.{10,80}\s*$', re.M)


def is_contaminated(s):
    if not s: return True
    s = (s or "").strip()
    if len(s) < 8: return True
    if _REF.search(s): return True
    if re.match(r'\s*\[\d+\]', s): return True
    if _META.match(s): return True
    # 含"题目：xx(1)"式题号、文件题名后缀
    if re.search(r'(论文模版|论文模板|：?检测报告|(1)(\.docx)?$)', s): return True
    return False


def clean_pairs(rows):
    """rows: list of dict (含 src_ai/tgt_human)。返回 (清洗后, 剔除数)。"""
    keep, drop = [], 0
    for r in rows:
        if is_contaminated((r.get("src_ai") or "")
                           ) or is_contaminated((r.get("tgt_human") or "")):
            drop += 1
            continue
        keep.append(r)
    return keep, drop


if __name__ == "__main__":
    import json, sys
    p = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\woshi\.dsh\aigc-detector\data\rewrite_pairs_wechat.jsonl"
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    keep, drop = clean_pairs(rows)
    print(f"{p}: 总 {len(rows)} | 保留 {len(keep)} | 剔除 {drop}")
