"""Export the final manuscript to editable Word with native OMML equations.

Requires Pandoc and python-docx. Uses the same Markdown and original figures
as export_manuscript.py; does not regenerate or rewrite scientific content.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}


def element(parent, tag, **attrs):
    node = parent.find(qn(tag))
    if node is None:
        node = OxmlElement(tag)
        parent.append(node)
    for key, value in attrs.items():
        node.set(qn('w:' + key), str(value))
    return node


def body_content(path):
    with ZipFile(path) as z:
        body = etree.fromstring(z.read('word/document.xml')).find('w:body', NS)
        return ([n.text or '' for n in body.xpath('.//w:t | .//m:t', namespaces=NS)],
                len(body.xpath('.//m:oMathPara', namespaces=NS)),
                len(body.xpath('.//m:oMath', namespaces=NS)),
                len(body.xpath('./w:tbl', namespaces=NS)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pandoc', default=shutil.which('pandoc'))
    parser.add_argument('--output', type=Path, default=ROOT/'paper/manuscript_v4_3_en.docx')
    args = parser.parse_args()
    if not args.pandoc:
        parser.error('Supply --pandoc executable')
    source = ROOT/'paper/manuscript_v4_3_en.md'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='manuscript-word-') as tmp:
        raw = Path(tmp)/'raw.docx'
        subprocess.run([args.pandoc, str(source), '--from=markdown-implicit_figures',
                        '--to=docx', '--standalone', '--resource-path='+str(source.parent),
                        '-o', str(raw)], check=True)
        original = body_content(raw)
        doc = Document(raw)
        section = doc.sections[0]
        section.page_width, section.page_height = Mm(210), Mm(297)
        section.top_margin = section.bottom_margin = Mm(18)
        section.left_margin = section.right_margin = Mm(17)
        section.header_distance = section.footer_distance = Mm(8)
        for name in ['Normal','Body Text','First Paragraph','Compact','Caption','Image Caption',
                     'Table Caption','Bibliography','Title','Heading 1','Heading 2','Heading 3']:
            if name not in doc.styles:
                continue
            st = doc.styles[name]
            st.font.name = 'Times New Roman'
            fonts = st.element.get_or_add_rPr().find(qn('w:rFonts'))
            if fonts is not None:
                for key in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme','csTheme']:
                    fonts.attrib.pop(qn('w:'+key),None)
            st.font.color.rgb = RGBColor(0,0,0)
            st.font.size = Pt(11)
            st.font.bold = False
            st.font.italic = False
            st.font.underline = False
            pf = st.paragraph_format
            pf.space_before = Pt(0)
            pf.space_after = Pt(6)
            pf.line_spacing = 1.15
            pf.widow_control = True
        for name, size, before, after in [('Heading 1',13,15,6),('Heading 2',11.3,11,5)]:
            st = doc.styles[name]
            st.font.size = Pt(size)
            st.font.bold = True
            st.paragraph_format.space_before = Pt(before)
            st.paragraph_format.space_after = Pt(after)
            st.paragraph_format.keep_with_next = True
        title = doc.styles['Title']
        title.font.size = Pt(19)
        title.font.bold = True
        title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title.paragraph_format.space_after = Pt(13)
        title.paragraph_format.keep_with_next = True
        if 'Hyperlink' in doc.styles:
            doc.styles['Hyperlink'].font.color.rgb = RGBColor(0,0,0)
            doc.styles['Hyperlink'].font.underline = False

        # Pandoc starts the manuscript title at Heading 1. Preserve all text and
        # shift only the hierarchy to Title / Heading 1 / Heading 2.
        paragraphs = doc.paragraphs
        for i,p in enumerate(paragraphs):
            oldstyle = p.style.name
            if i == 0:
                p.style = 'Title'
            elif oldstyle == 'Heading 2':
                p.style = 'Heading 1'
            elif oldstyle == 'Heading 3':
                p.style = 'Heading 2'
            elif i <= 3:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after = Pt(4 if i < 3 else 14)
                p.paragraph_format.keep_with_next = True
            else:
                p.paragraph_format.keep_with_next = False
            p.paragraph_format.widow_control = True
            if p._p.xpath('.//m:oMathPara'):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.keep_together = True
                p.paragraph_format.space_before = Pt(7)
                p.paragraph_format.space_after = Pt(7)
            if p._p.xpath('.//w:drawing'):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.keep_with_next = True
                p.paragraph_format.keep_together = True
                p.paragraph_format.space_before = Pt(7)
            if re.match(r'^(Figure|Table) \d+\.', p.text):
                p.style = 'Caption'
                p.paragraph_format.keep_together = True
                p.paragraph_format.keep_with_next = False
                p.paragraph_format.space_after = Pt(8)
                for run in p.runs:
                    run.font.size = Pt(9.5)
            if p._p.xpath('./w:pPr/w:numPr'):
                p.paragraph_format.keep_together = True

        # Scale original images proportionally to the existing PDF's usable area.
        max_width, max_height = Mm(176), Mm(150)
        for shape in doc.inline_shapes:
            factor = min(1, max_width/shape.width, max_height/shape.height)
            shape.width = int(shape.width*factor)
            shape.height = int(shape.height*factor)

        widths = [[.58,.42], [.22,.12,.66], [.22,.195,.195,.195,.195],
                  [.07,.16,.20,.19,.19,.19], [.22,.195,.195,.195,.195],
                  [.78,.22], [.31,.23,.23,.23]]
        for ti, table in enumerate(doc.tables):
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            table.autofit = False
            pr = table._tbl.tblPr
            element(pr,'w:tblW',w=9978,type='dxa')
            borders = element(pr,'w:tblBorders')
            for edge in ['top','left','bottom','right','insideH','insideV']:
                element(borders,'w:'+edge,val='single',sz=4,color='D9D9D9')
            margins = element(pr,'w:tblCellMar')
            for edge,value in [('top',80),('bottom',80),('left',70),('right',70)]:
                element(margins,'w:'+edge,w=value,type='dxa')
            for j, column in enumerate(table.columns):
                column.width = Mm(176*widths[ti][j])
            for ri,row in enumerate(table.rows):
                trpr = row._tr.get_or_add_trPr()
                element(trpr,'w:cantSplit')
                if ri == 0:
                    element(trpr,'w:tblHeader')
                for ci,cell in enumerate(row.cells):
                    cell.width = Mm(176*widths[ti][ci])
                    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                    if ri == 0:
                        element(cell._tc.get_or_add_tcPr(),'w:shd',fill='F2F2F2',val='clear')
                    for p in cell.paragraphs:
                        p.paragraph_format.space_before = Pt(0)
                        p.paragraph_format.space_after = Pt(1)
                        p.paragraph_format.line_spacing = 1.08
                        # These seven short tables fit on one page each. Keep
                        # their final row with the following explicit caption.
                        p.paragraph_format.keep_with_next = True
                        p.paragraph_format.keep_together = True
                        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci==0 or ti==1 else WD_ALIGN_PARAGRAPH.CENTER
                        for run in p.runs:
                            run.font.name = 'Times New Roman'
                            run.font.size = Pt(8.5 if ti==3 else 9)
                            if ri == 0:
                                run.font.bold = True

        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        field = OxmlElement('w:fldSimple')
        field.set(qn('w:instr'),'PAGE')
        run = OxmlElement('w:r')
        rp = OxmlElement('w:rPr')
        element(rp,'w:sz',val=18)
        run.append(rp)
        t = OxmlElement('w:t'); t.text='1'; run.append(t); field.append(run)
        footer._p.append(field)
        doc.core_properties.title = paragraphs[0].text
        doc.core_properties.author = 'Hengyi Ji'
        doc.core_properties.subject = 'Editable English manuscript for Applied Optics submission'
        doc.core_properties.comments = ''
        doc.save(args.output)
        final = body_content(args.output)
        assert original == final, 'Formatting altered manuscript text, equations or tables'
        assert final[1] == 11 and final[3] == 7
        with ZipFile(args.output) as z:
            images = [n for n in z.namelist() if n.startswith('word/media/')]
            assert len(images)==4
            sha=lambda b:hashlib.sha256(b).hexdigest()
            source_images=[source.parent/p for p in re.findall(r'!\[[^]]*\]\(([^)]+)\)',source.read_text(encoding='utf-8'))]
            assert sorted(sha(z.read(n)) for n in images)==sorted(sha(p.read_bytes()) for p in source_images)
        print(json.dumps({'docx':str(args.output),'editable_display_equations':final[1],
                          'total_editable_equations':final[2],'native_tables':final[3],
                          'original_images':len(images),'formatting_preserves_content':True}))


if __name__=='__main__':
    main()
