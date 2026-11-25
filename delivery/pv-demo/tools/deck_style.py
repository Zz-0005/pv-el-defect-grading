# -*- coding: utf-8 -*-
"""Shared visual primitives for the defense deck (flat, editorial, technical)."""
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from pptx.oxml import parse_xml

FONT = "微软雅黑"

INK      = RGBColor.from_string("152430")
INK_SOFT = RGBColor.from_string("3D5261")
MUTED    = RGBColor.from_string("7C8C99")
LINE     = RGBColor.from_string("D9E1E7")
SOFT     = RGBColor.from_string("F4F7F9")
SOFT2    = RGBColor.from_string("EAF0F4")
WHITE    = RGBColor.from_string("FFFFFF")
ACCENT   = RGBColor.from_string("0E6F73")   # deep teal
ACCENT_L = RGBColor.from_string("E3F0F0")
AMBER    = RGBColor.from_string("C9762A")   # signal amber
AMBER_L  = RGBColor.from_string("FBEEDF")
DARK     = RGBColor.from_string("10222A")

NO_STYLE = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"   # No Style, No Grid


# ---------------------------------------------------------------- text helpers
def _apply_font(run, size, bold, color, font=FONT, italic=False):
    f = run.font
    f.name = font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set("typeface", font)


def textbox(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP, pad=(0, 0, 0, 0), wrap=True):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left, tf.margin_top, tf.margin_right, tf.margin_bottom = (Inches(v) for v in pad)
    return tb


def para(tf, first, text=None, *, size=13, bold=False, color=INK, align=PP_ALIGN.LEFT,
         before=0, after=0, line=1.28, font=FONT):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(before)
    p.space_after = Pt(after)
    p.line_spacing = line
    if text is not None:
        _apply_font(p.add_run(), size, bold, color, font)
        p.runs[0].text = text
    return p


def rich(p, segments):
    """segments: [(text, size, bold, color)] appended to paragraph p."""
    for text, size, bold, color in segments:
        r = p.add_run()
        r.text = text
        _apply_font(r, size, bold, color)


def simple(slide, x, y, w, h, text, *, size=13, bold=False, color=INK, align=PP_ALIGN.LEFT,
           anchor=MSO_ANCHOR.TOP, line=1.25, pad=(0, 0, 0, 0)):
    tb = textbox(slide, x, y, w, h, anchor=anchor, pad=pad)
    p = para(tb.text_frame, True, text, size=size, bold=bold, color=color, align=align, line=line)
    return tb


# ---------------------------------------------------------------- shapes
def rect(slide, x, y, w, h, fill=SOFT, line=None, shape=MSO_SHAPE.RECTANGLE, radius=None,
         line_w=0.75, shadow=False):
    shp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(line_w)
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        shp.adjustments[0] = radius
    if not shadow:
        shp.shadow.inherit = False
    shp.text_frame.word_wrap = True
    return shp


def hline(slide, x, y, w, color=LINE, thick=0.75):
    return rect(slide, x, y, w, thick / 72.0, fill=color)


def arrow(slide, x, y, w, h, color=RGBColor.from_string("B9C7CF")):
    shp = rect(slide, x, y, w, h, fill=color, shape=MSO_SHAPE.RIGHT_ARROW, shadow=True)
    return shp


# ---------------------------------------------------------------- chrome
def slide_header(slide, eyebrow, title, *, ey_color=ACCENT, sub=None, w=11.9):
    """Standard page header: time-coded eyebrow, headline, hairline rule."""
    simple(slide, 0.72, 0.46, w, 0.26, eyebrow, size=10.5, bold=True, color=ey_color)
    simple(slide, 0.72, 0.74, w, 0.62, title, size=25, bold=True, color=INK, line=1.1)
    y = 1.44 if sub is None else 1.36
    if sub:
        simple(slide, 0.72, y, w, 0.3, sub, size=11.5, color=MUTED, line=1.2)
        y += 0.34
    hline(slide, 0.72, y + 0.06, w, LINE)
    return y + 0.22


def slide_footer(slide, page, total, tag="AI 机器视觉工程师订单班 · 项目答辩"):
    hline(slide, 0.72, 7.02, 11.9, LINE)
    simple(slide, 0.72, 7.12, 7.0, 0.24, tag, size=8.5, color=MUTED)
    simple(slide, 10.4, 7.12, 2.2, 0.24, f"{page:02d} / {total:02d}", size=8.5, color=MUTED,
           align=PP_ALIGN.RIGHT)


def set_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# ---------------------------------------------------------------- table
def add_table(slide, x, y, w, rows, col_w, *, head_fill=DARK, head_color=WHITE,
              body_fill=WHITE, zebra=False, size=10.5, head_size=10, row_h=0.36,
              head_h=0.38, bold_cols=(0,), align=None, first_col_color=INK,
              rule_rows=(), rule_color=LINE, pad=0.07):
    """rows[0] is the header. col_w in inches. align: list of PP_ALIGN per column."""
    nrows, ncols = len(rows), len(rows[0])
    total_h = head_h + row_h * (nrows - 1)
    gf = slide.shapes.add_table(nrows, ncols, Inches(x), Inches(y), Inches(w), Inches(total_h))
    tbl = gf.table
    tblPr = tbl._tbl.tblPr
    tblPr.set("firstRow", "0")
    tblPr.set("bandRow", "0")
    for el in tblPr.findall(qn("a:tableStyleId")):
        tblPr.remove(el)
    tblPr.append(parse_xml(f'<a:tableStyleId xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">{NO_STYLE}</a:tableStyleId>'))

    scale = w / sum(col_w)
    for i, cw in enumerate(col_w):
        tbl.columns[i].width = Inches(cw * scale)
    tbl.rows[0].height = Inches(head_h)
    for r in range(1, nrows):
        tbl.rows[r].height = Inches(row_h)

    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.margin_left = Inches(pad)
            cell.margin_right = Inches(pad)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if r == 0:
                cell.fill.fore_color.rgb = head_fill
            elif r in rule_rows:
                cell.fill.fore_color.rgb = SOFT2
            elif zebra and r % 2 == 0:
                cell.fill.fore_color.rgb = SOFT
            else:
                cell.fill.fore_color.rgb = body_fill
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = (align[c] if align else (PP_ALIGN.LEFT if c else PP_ALIGN.LEFT))
            p.line_spacing = 1.12
            run = p.add_run()
            run.text = str(val)
            if r == 0:
                _apply_font(run, head_size, True, head_color)
            elif c in bold_cols:
                _apply_font(run, size, True, first_col_color)
            else:
                _apply_font(run, size, False, INK_SOFT)
    return gf


# ---------------------------------------------------------------- cards
def card(slide, x, y, w, h, *, fill=WHITE, line=LINE, accent=None, pad=0.22):
    """White card with hairline border and optional left accent bar."""
    rect(slide, x, y, w, h, fill=fill, line=line)
    if accent is not None:
        rect(slide, x, y, 0.045, h, fill=accent)
    return x + pad, y + pad * 0.85
