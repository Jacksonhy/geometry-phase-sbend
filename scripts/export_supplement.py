"""Export the current English supplement with original images and MathML."""
import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from export_manuscript import CSS, ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pandoc',default=shutil.which('pandoc'))
    p.add_argument('--chrome',default=shutil.which('google-chrome') or shutil.which('chromium') or shutil.which('chrome'))
    p.add_argument('--output',type=Path,default=ROOT/'paper/supplement_v4_3_aligned.pdf')
    a=p.parse_args()
    if not a.pandoc or not a.chrome:
        p.error('Supply --pandoc and --chrome executable paths')
    a.output=a.output.resolve();a.output.parent.mkdir(parents=True,exist_ok=True)
    css=CSS.replace('content: counter(page)', 'content: "Supplement 1 - " counter(page)')
    css+='''
img { max-height: 180mm; }
table { font-size: 8pt; }
td, th { padding: 4pt 2.5pt; }
.table-block > p { font-size: 9.2pt; }
.figure-block > p { font-size: 9.5pt; }
code { font-size: .9em; overflow-wrap: anywhere; }
'''
    with tempfile.TemporaryDirectory(prefix='phase-supplement-') as td:
        tmp=Path(td);style=tmp/'style.css';style.write_text(css,encoding='utf-8')
        html=tmp/'supplement.html'
        result=subprocess.run([a.pandoc,'supplement_v4_3_aligned.md',
                    '--from=markdown+tex_math_single_backslash','--to=html5','--standalone',
                    '--mathml','--embed-resources','--css',str(style),
                    '--metadata','pagetitle=Supplement 1 - Geometry-derived phase references',
                    '-o',str(html)],cwd=ROOT/'paper',check=True,capture_output=True,text=True,encoding='utf-8')
        if 'Could not convert TeX math' in result.stderr or 'Could not fetch resource' in result.stderr:
            raise RuntimeError(result.stderr)
        content=html.read_text(encoding='utf-8')
        content=re.sub(r'<colgroup>.*?</colgroup>','',content,flags=re.S)
        content=re.sub(r'(<figure>.*?</figure>)(\s*<p><em>Figure S\d+\..*?</p>)',
                       r'<div class="figure-block">\1\2</div>',content,flags=re.S)
        content=re.sub(r'(<table\b[^>]*>.*?</table>)(\s*<p><em>Table S\d+\..*?</p>)',
                       r'<div class="table-block">\1\2</div>',content,flags=re.S)
        assert content.count('class="figure-block"')==7
        if content.count('class="table-block"')!=23:
            diagnostic=ROOT/'outputs/supplement_render.html'
            diagnostic.parent.mkdir(parents=True,exist_ok=True)
            diagnostic.write_text(content,encoding='utf-8')
            raise RuntimeError('Table grouping count: '+str(content.count('class="table-block"')))
        assert len(re.findall(r'<math\s[^>]*display="block"',content))==9
        html.write_text(content,encoding='utf-8')
        subprocess.run([a.chrome,'--headless=new','--disable-gpu','--no-pdf-header-footer',
                        '--no-first-run','--no-default-browser-check','--user-data-dir='+str(tmp/'profile'),
                        '--print-to-pdf='+str(a.output),html.as_uri()],check=True,capture_output=True)
    assert a.output.is_file() and a.output.stat().st_size>10000
    print(json.dumps(dict(pdf=str(a.output),bytes=a.output.stat().st_size,figures=7,tables=23,
                          display_math_groups=9,all_resources_embedded=True)))


if __name__=='__main__':
    main()
