#!/usr/bin/env python3
import sys, os, re

def split_chat_by_date(input_files, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    date_contents = {}
    date_header = re.compile(r"^###\s+(\d{4}-\d{2}-\d{2})\s*$")
    inline_date = re.compile(r"[（(](\d{4}-\d{2}-\d{2})\s+\d{2}:\d{2}[）)]")
    total_lines = 0
    total_dates = set()
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"文件不存在: {input_file}")
            continue
        print(f"正在处理: {input_file}")
        with open(input_file, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        current_date = None
        for line in lines:
            total_lines += 1
            m = date_header.match(line.strip())
            if m:
                current_date = m.group(1)
                total_dates.add(current_date)
                date_contents.setdefault(current_date, []).append(line)
                continue
            m = inline_date.search(line)
            if m:
                current_date = m.group(1)
                total_dates.add(current_date)
                date_contents.setdefault(current_date, []).append(line)
                continue
            if current_date:
                date_contents.setdefault(current_date, []).append(line)
            else:
                date_contents.setdefault("unknown", []).append(line)
    file_count = 0
    for dk in sorted(date_contents.keys()):
        content = date_contents[dk]
        if all(l.strip() == "" for l in content):
            continue
        with open(os.path.join(output_dir, f"{dk}.txt"), "w", encoding="utf-8") as fh:
            fh.writelines(content)
        print(f"  {dk}.txt ({len([l for l in content if l.strip()])} 行)")
        file_count += 1
    print(f"完成! {total_lines}行, {len(total_dates)}个日期, {file_count}个文件")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python split_by_date.py 文件1.txt [文件2.txt ...]")
        sys.exit(1)
    split_chat_by_date(sys.argv[1:])
