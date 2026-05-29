import re, json, sys

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    content = f.read()

scripts = re.findall(r'<script[^>]*>([\s\S]*?)</script>', content)
for i, s in enumerate(scripts):
    if 'content' not in s or len(s) < 2000:
        continue
    esc = re.findall(r'\\\\u[0-9a-f]{4}', s, re.IGNORECASE)
    if len(esc) > 50:
        print(f'Script {i}: {len(s)} chars, {len(esc)} unicode esc', file=sys.stderr)
        # Try to find "content":"..." pattern
        for m in re.finditer(r'\\"content\\"\s*:\s*\\"([\s\S]{500,}?)\\"(?=\s*,\s*\\"|\s*})', s):
            raw = m.group(1)
            chinese = re.findall(r'[一-鿿]{10,}', raw)
            if chinese:
                print('\n'.join(chinese[:30]))
                break
        break
