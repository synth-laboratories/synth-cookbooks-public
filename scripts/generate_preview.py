"""Write a local static preview of cookbook pages (no Next required)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = json.loads((ROOT / "catalog" / "generated" / "pages.json").read_text())


CSS = """
:root { color-scheme: dark; --bg:#0b0b0c; --ink:#f4f1ea; --muted:#b7b2a7; --line:#ffffff1a; --accent:#e4572e; }
* { box-sizing: border-box; }
body { margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:var(--bg); color:var(--ink); }
main { max-width: 920px; margin: 0 auto; padding: 48px 24px 96px; }
a { color: var(--accent); text-decoration: none; }
.kicker { font-size: 11px; letter-spacing: .18em; text-transform: uppercase; color: var(--accent); }
h1 { font-size: 40px; line-height: 1.1; margin: 16px 0 12px; }
.grid { display:grid; gap:16px; grid-template-columns: repeat(auto-fit,minmax(260px,1fr)); margin-top: 32px; }
.card, .panel { border:1px solid var(--line); border-radius: 16px; padding: 20px; }
.muted { color: var(--muted); }
pre { overflow:auto; background:#00000066; border:1px solid var(--line); padding:16px; border-radius:12px; font-size:12px; }
dt { font-size:11px; letter-spacing:.16em; text-transform:uppercase; color:#ffffff66; }
dd { margin: 8px 0 20px; }
"""


def page_html(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{CSS}</style></head>
<body><main>{body}</main></body></html>
"""


def main() -> int:
    out = ROOT / "preview"
    out.mkdir(exist_ok=True)
    cards = []
    for page in PAGES["pages"]:
        cards.append(
            f'<a class="card" href="{page["slug"]}.html"><p class="kicker">{page["method"]} · {page["task"]}</p>'
            f'<h2>{page["title"]}</h2><p class="muted">{page["verdict"]}</p></a>'
        )
        receipt = json.dumps(page["receipt"], indent=2)
        body = f"""
        <p><a href="index.html">← Cookbooks</a></p>
        <p class="kicker">{page["method"]} · {page["task"]} · {page["execution"]} · {page["status"]}</p>
        <h1>{page["title"]}</h1>
        <dl>
          <dt>Hypothesis</dt><dd>{page["hypothesis"]}</dd>
          <dt>Verdict</dt><dd>{page["verdict"]}</dd>
          <dt>Confidence</dt><dd>{page["confidence"]}</dd>
          <dt>Why</dt><dd class="muted">{page["why"]}</dd>
        </dl>
        <h2>Measured receipt</h2>
        <pre>{receipt}</pre>
        <p class="muted">Source {page["source"]}. Local preview only; frontend route is {page["href"]}.</p>
        """
        (out / f"{page['slug']}.html").write_text(page_html(page["title"], body))
    index = """
    <p class="kicker">Local preview</p>
    <h1>Public cookbooks</h1>
    <p class="muted">Generated from catalog/generated/pages.json. Serve with <code>python3 -m http.server --directory preview</code>.</p>
    <div class="grid">""" + "".join(cards) + "</div>"
    (out / "index.html").write_text(page_html("Cookbooks", index))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
