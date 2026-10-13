import json
from pathlib import Path

path = Path(__file__).with_name("rag_predictions.json")
lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

indices = range(54, 59)
new_lines = []
for i, idx in enumerate(indices):
    raw = lines[idx]
    content = raw.rstrip("\n\r").lstrip()
    escaped = json.dumps(content, ensure_ascii=False)
    suffix = "," if i < 4 else ""
    new_lines.append("      " + escaped + suffix + "\n")

for idx, nl in zip(indices, new_lines):
    lines[idx] = nl

path.write_text("".join(lines), encoding="utf-8")
print("OK")
