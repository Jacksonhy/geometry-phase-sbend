"""Render the English Markdown directly via Pandoc MathML and headless Chrome.

Requires Pandoc and Chrome/Chromium on PATH, or explicit executable arguments.
No network resources are used and no browser user profile is opened.
"""
import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = '''
@page { size: A4; margin: 18mm 17mm 18mm; @bottom-center { content: counter(page); font-size: 9pt; } }
html { color: #111; font-family: "Times New Roman", "Liberation Serif", serif; font-size: 10.5pt; }
body { margin: 0; line-height: 1.32; }
h1 { font-size: 19pt; text-align: center; line-height: 1.15; margin: 0 0 13pt; }
h2 { font-size: 13pt; margin: 15pt 0 6pt; break-after: avoid; }
h3 { font-size: 11.3pt; margin: 11pt 0 5pt; break-after: avoid; }
p { margin: 6pt 0; orphans: 3; widows: 3; }
.author-block { text-align: center; margin: 0 0 14pt; break-inside: avoid; }
.author-block p { margin: 4pt 0; }
img { display: block; max-width: 100%; max-height: 150mm; margin: 7pt auto; object-fit: contain; }
figure { margin: 9pt 0 4pt; break-inside: avoid; }
.figure-block, .table-block { break-inside: avoid; }
figcaption { display: none; }
table { border-collapse: collapse; width: 100%; font-size: 8.6pt; line-height: 1.18; margin: 8pt 0 5pt; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th { border-top: 1pt solid #222; border-bottom: .6pt solid #444; padding: 5pt 3pt; text-align: left; }
td { padding: 5pt 3pt; vertical-align: top; }
tbody tr:last-child td { border-bottom: 1pt solid #222; }
table+p { break-before: avoid; font-size: 9.5pt; }
math { font-family: "Cambria Math", "STIX Two Math", math; }
math[display="block"] { margin: 9pt 0; font-size: 10.5pt; break-inside: avoid; }
a { color: #111; text-decoration: none; overflow-wrap: anywhere; }
li { margin-bottom: 4pt; break-inside: avoid; }
ol { padding-left: 19pt; }
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pandoc', default=shutil.which('pandoc'))
    p.add_argument('--chrome', default=shutil.which('google-chrome') or shutil.which('chromium') or shutil.which('chrome'))
    p.add_argument('--output', type=Path, default=ROOT/'paper/manuscript_v4_3_en.pdf')
    a = p.parse_args()
    if not a.pandoc or not a.chrome:
        p.error('Supply --pandoc and --chrome executables')
    a.output = a.output.resolve()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='phase-manuscript-') as td:
        tmp = Path(td); css = tmp/'style.css'; css.write_text(CSS, encoding='utf-8')
        html = tmp/'manuscript.html'
        subprocess.run([a.pandoc, 'manuscript_v4_3_en.md', '--from=markdown', '--to=html5',
                        '--standalone', '--mathml', '--embed-resources', '--css', str(css),
                        '--metadata', 'pagetitle=Geometry-derived phase references for S-bends',
                        '-o', str(html)], cwd=ROOT/'paper', check=True)
        content = html.read_text(encoding='utf-8')
        content = re.sub(r'<colgroup>.*?</colgroup>', '', content, flags=re.S)
        content = re.sub(r'(<figure>.*?</figure>)(\s*<p><strong>Figure .*?</p>)',
                         r'<div class="figure-block">\1\2</div>', content, flags=re.S)
        content = re.sub(r'(<table>.*?</table>)(\s*<p><strong>Table .*?</p>)',
                         r'<div class="table-block">\1\2</div>', content, flags=re.S)
        html.write_text(content, encoding='utf-8')
        subprocess.run([a.chrome, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                        '--no-first-run', '--no-default-browser-check',
                        '--user-data-dir='+str(tmp/'profile'), '--print-to-pdf='+str(a.output),
                        html.as_uri()], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if not a.output.is_file() or a.output.stat().st_size < 10000:
        raise RuntimeError('PDF export failed')
    print('EXPORTED', a.output)


if __name__ == '__main__':
    main()
