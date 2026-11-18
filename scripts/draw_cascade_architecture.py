import argparse
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.tile_geometry import TILES

NAME = "cascade_audit_v2_architecture"

FRAME = ROOT / "data/processed/TACK_Tunnel_B/full_frames/images/TB_Camera8_001935.png"
TILE = ROOT / "data/TACK_Tunnel_Data/3_img/TB_Camera8_001935_G.png"

# Display sizes are schematic. Each segmenter stage lists its output resolution,
# encoder channels, decoder channels and drawn height/depth.
SCALES = (
    (256, 16, 24, 15),
    (128, 24, 40, 12),
    (64, 40, 64, 9.5),
    (32, 80, 96, 7),
    (16, 160, None, 5),
)


def slab(name, x, y, height, widths, channels, color, band=None):
    widths_tex = ",".join(f"{width:g}" for width in widths)
    # A second entry keeps PGF from indexing a single channel label as a string.
    labels = channels if len(channels) > 1 else (*channels, "")
    channel_labels = ",".join(f'"{channel}"' for channel in labels)
    shape = "RightBandedBox" if band else "Box"
    options = [f"name={name}", f"width={{{widths_tex}}}", f"height={height}",
               f"depth={height}", rf"fill=\{color}", f"xlabel={{{{{channel_labels}}}}}"]
    if band:
        options.append(rf"bandfill=\{band}")
    return rf"""\pic at ({x:.2f},{y:.2f},0) {{{shape}={{{','.join(options)}}}}};
\coordinate ({name}-in) at ({name}-nearwest |- {name}-west);
\coordinate ({name}-out) at ({name}-fareast |- {name}-east);"""


def caption(x, y, title, tensor=""):
    return rf"\node[caption] at ({x:.2f},{y:.2f}) {{{title}\\\detail{{{tensor}}}}};"


def connect(source, target, label=""):
    annotation = rf" node[op,pos=.50] {{{label}}}" if label else ""
    return rf"\draw[connection] ({source}) -- node {{\midarrow}}{annotation} ({target});"


def feature_caption(x, height, widths, title, tensor="", shift=0):
    return caption(x + .1 * sum(widths) - .071 * height + shift, -.171 * height - .80, title, tensor)


def segmenter_layers():
    parts = []
    previous = "tile-out"
    previous_edge = 1.39
    skips = []
    for level, (size, encoder, _, height) in enumerate(SCALES, 1):
        name = f"e{level}"
        widths = (2 + .72 * level,)
        gap = 1.10 if level == 1 else .68
        x = previous_edge + gap + .071 * height
        parts.append(slab(name, x, 0, height, widths, (encoder,), "ConvColor", "ConvReluColor"))
        title = "Bottleneck" if level == len(SCALES) else f"Encoder {level}"
        parts.append(feature_caption(x, height, widths, title, f"{size} × {size}", shift=.16))
        if level > 1:
            parts.append(rf"\path ({previous}) -- coordinate[pos=.40] (tap{level - 1}) ({name}-in);")
            skips.append(f"tap{level - 1}")
        parts.append(connect(previous, f"{name}-in", "Normalize" if level == 1 else ""))
        previous = f"{name}-out"
        previous_edge = x + .2 * sum(widths) + .071 * height

    channels = SCALES[-1][1]
    for index, (size, skip, outputs, height) in enumerate(reversed(SCALES[:-1]), 1):
        name = f"d{index}"
        merge = f"cat{index}"
        merge_x = previous_edge + 1.36
        widths = (2.4, 2.4)
        x = merge_x + 1.04 + .071 * height
        parts.append(rf"\pic at ({merge_x:.2f},0,0) {{Ball={{name={merge},fill=\SumColor,opacity=.62,radius=2.25,logo=C}}}};")
        parts.append(slab(name, x, 0, height, widths, (outputs, outputs), "DcnvColor", "DcnvBand"))
        parts.append(feature_caption(x, height, widths, "SepBlock", f"{channels + skip} → {outputs} ch · {size}×{size}", shift=.42))
        parts.append(connect(previous, f"{merge}-west", "Up 2×"))
        parts.append(connect(f"{merge}-east", f"{name}-in"))
        lane = -3.25 - .68 * (index - 1)
        parts.append(rf"""\draw[connection,skip]
  ({skips[-index]}) -- ({skips[-index]} |- 0,{lane:.2f})
  -- node {{\midarrow}} ({merge}-south |- 0,{lane:.2f})
  -- node {{\midarrow}} ({merge}-south);""")
        channels = outputs
        previous = f"{name}-out"
        previous_edge = x + .2 * sum(widths) + .071 * height

    height = 17
    up_x = previous_edge + 1.25 + .071 * height
    parts.append(slab("up", up_x, 0, height, (1.5,), (channels,), "DcnvColor"))
    parts.append(feature_caption(up_x, height, (1.5,), "Bilinear upsample", "512 × 512"))
    parts.append(connect(previous, "up-in", "Up 2×"))
    head_x = up_x + .30 + .142 * height + .60
    parts.append(slab("head", head_x, 0, height, (1.65,), (1,), "SoftmaxColor"))
    parts.append(feature_caption(head_x, height, (1.65,), "1×1 head", "512 × 512 logits"))
    parts.append(connect("up-out", "head-in"))
    parts.append(rf"\coordinate (right-edge) at ({head_x + .33 + .071 * height:.2f},0);")
    return "\n".join(parts)


def ranker_layers():
    parts = [
        slab("features", 5.0, 7.60, 11, (2.35, 2.35), ("", 576), "ConvColor", "ConvReluColor"),
        caption(4.78, 4.92, "Shared encoder", "12 × 576 × 16 × 16"),
        slab("avg", 9.27, 8.57, 3.0, (1.5,), (576,), "PoolColor"),
        slab("max", 9.27, 6.63, 3.0, (1.5,), (576,), "PoolColor"),
        r"\node[op] at (9.38,9.28) {Global avg};",
        r"\node[op] at (9.38,7.27) {Global max};",
        r"\pic at (11.80,7.60) {Ball={name=pool-cat,fill=\SumColor,opacity=.62,radius=2.25,logo=C}};",
        slab("linear", 14.28, 7.60, 6.2, (3.0,), (128,), "ConvColor", "ConvReluColor"),
        caption(14.15, 4.92, "Projection · 1152 → 128", "ReLU, dropout 0.1"),
        slab("grid", 18.52, 7.60, 4.8, (1.8,), (128,), "ConvColor"),
        caption(18.58, 4.92, "Reshape + permute", "128 × 3 × 4"),
        r"\pic at (21.60,7.60) {Ball={name=position,fill=\SumColor,opacity=.62,radius=2.25,logo=+}};",
        r"\node[op] (learned) at (21.60,9.20) {Learned position · 128 × 3 × 4};",
        slab("context", 24.3, 7.60, 4.8, (2.9, 2.2), (128, 64), "ConvColor", "ConvReluColor"),
        caption(24.55, 4.92, "Grid context", "2 × (3×3 conv + BN + ReLU)"),
        slab("score", 28.23, 7.60, 4.8, (1.4,), (1,), "SoftmaxColor"),
        caption(28.68, 4.92, "1×1 score head", "1 × 3 × 4"),
        r"\node[caption] (logits) at (32.10,7.60) {Flatten\\\detail{12 tile logits}};",
        r"\node[caption] (topk) at (37.2,7.60) {Top-6\\\detail{Gather RGB tiles for Stage 2}};",
        connect("frame-out", "features-in", "Normalize"),
        r"\coordinate (fork) at (7.48,7.60);",
        r"\draw[connection] (features-out) -- (fork);",
        r"\draw[connection] (fork) |- node[pos=.82] {\midarrow} (avg-in);",
        r"\draw[connection] (fork) |- node[pos=.82] {\midarrow} (max-in);",
        connect("avg-out", "pool-cat-west"),
        connect("max-out", "pool-cat-west"),
        connect("pool-cat-east", "linear-in"),
        connect("linear-out", "grid-in"),
        connect("grid-out", "position-west"),
        connect("learned.south", "position-north"),
        connect("position-east", "context-in"),
        connect("context-out", "score-in"),
        connect("score-out", "logits.west"),
        connect("logits.east", "topk.west"),
    ]
    return "\n".join(parts)


def make_layout():
    tile_names = ",".join(f"{label}/{col}/{row}" for label, (col, row) in TILES.items())
    return (LAYOUT.replace("FRAME_IMAGE", FRAME.as_posix()).replace("TILE_IMAGE", TILE.as_posix())
            .replace("TILE_NAMES", tile_names).replace("RANKER_LAYERS", ranker_layers())
            .replace("SEGMENTER_LAYERS", segmenter_layers()))


LAYOUT = r"""\usetikzlibrary{calc}
\DeclareFontFamily{TU}{diagramroman}{}
\DeclareFontShape{TU}{diagramroman}{m}{n}{<-> \UnicodeFontFile{lmroman10-regular}{\UnicodeFontTeXLigatures}}{}
\DeclareFontShape{TU}{diagramroman}{b}{n}{<-> \UnicodeFontFile{lmroman10-bold}{\UnicodeFontTeXLigatures}}{}
\DeclareFontShape{TU}{diagramroman}{bx}{n}{<-> ssub * diagramroman/b/n}{}
\renewcommand{\rmdefault}{diagramroman}
\def\DcnvColor{rgb:blue,5;green,2.5;white,5}
\def\DcnvBand{rgb:blue,5;cyan,3;white,2}
\definecolor{ink}{HTML}{202524}
\definecolor{note}{HTML}{435553}
\newcommand{\detail}[1]{\textcolor{note}{\footnotesize #1}}
\begin{document}
\begin{tikzpicture}[
    z={(-.71cm,-.71cm)},text=ink,
    connection/.style={ultra thick,draw=\edgecolor,opacity=.72,
        every node/.style={sloped,allow upside down}},
    skip/.style={draw=\edgecolor,line width=1.15pt},
    title/.style={anchor=west,font=\bfseries\Large,inner sep=0pt},
    subtitle/.style={anchor=west,font=\small,inner sep=0pt},
    caption/.style={align=center,font=\bfseries\small,inner xsep=3pt,inner ysep=0pt},
    op/.style={above=3pt,font=\footnotesize,text=note,inner sep=1pt},
]
\node[title] at (-1.52,12.03) {Fixed-grid crack segmentation};
\node[subtitle] at (-1.52,11.43)
    {Cascade audit v2 · 2048 × 1536 frame · Ranked selects six tiles · Full bypasses ranking and segments all twelve};

\node[title] at (-1.52,10.70) {MobileNetV3-Small grid ranker};
\node[subtitle] at (-1.52,10.15) {Stage 1 · 1,297,761 parameters};
\begin{scope}[cm={.71,.71,0,1,(.15,7.60)},transform shape]
    \node[inner sep=0pt] at (0,0) {\includegraphics[width=3.20cm,height=2.40cm]{FRAME_IMAGE}};
    \foreach \column in {-0.8,0,0.8}
      \draw[white,opacity=.62,line width=.45pt] (\column,-1.2) -- (\column,1.2);
    \foreach \row in {-.4,.4}
      \draw[white,opacity=.62,line width=.45pt] (-1.6,\row) -- (1.6,\row);
    \foreach \tile/\col/\row in {TILE_NAMES}
      \node[white,font=\tiny\bfseries] at ({-.8*1.5+.8*\col},{.8-.8*\row}) {\tile};
\end{scope}
\coordinate (frame-out) at (1.42,7.60);
\node[caption] at (-.16,4.92) {3×4 RGB tile grid\\\detail{12 × 3 × 512 × 512}};
RANKER_LAYERS

\node[title] at (-1.52,3.94) {MobileNetV3-Large U-Net};
\node[subtitle] at (-1.52,3.38) {Stage 2 · 1,282,913 parameters};
\begin{scope}[cm={.71,.71,0,1,(0,0)},transform shape]
    \node[inner sep=0pt] at (0,0) {\includegraphics[width=3.70cm,height=3.70cm]{TILE_IMAGE}};
\end{scope}
\coordinate (tile-out) at (1.39,0);
\node[caption] at (-.25,-3.51) {RGB tile\\\detail{3 × 512 × 512}};
SEGMENTER_LAYERS

\node[subtitle] at (-1.52,-6.04) {C: channel concatenation\quad
    Up 2×: bilinear interpolation\quad
    SepBlock: two depthwise 3×3 + pointwise 1×1 pairs, each convolution followed by BN and ReLU6};
\node[subtitle,anchor=east] at (right-edge |- 0,-6.04)
    {Sigmoid threshold 0.5, then stitch masks into the frame. Unselected tiles stay zero.};
"""


def main():
    parser = argparse.ArgumentParser(description="Build the cascade audit diagram with PlotNeuralNet.")
    parser.add_argument("--preview", action="store_true", help="Render to tmp without updating the report.")
    args = parser.parse_args()

    backend = ROOT / ".tools/PlotNeuralNet"
    compiler = ROOT / ".tools/bin/tectonic"
    if not (backend / "pycore/tikzeng.py").is_file() or not compiler.is_file():
        parser.error("Run scripts/setup_architecture_tools.py first.")
    for sample in (FRAME, TILE):
        if not sample.is_file():
            parser.error(f"Missing diagram input image: {sample.relative_to(ROOT)}")
    sys.path.insert(0, str(backend))
    from pycore.tikzeng import to_cor, to_end, to_generate

    temporary = ROOT / "tmp/pdfs"
    temporary.mkdir(parents=True, exist_ok=True)
    output = (temporary if args.preview else ROOT / "outputs/reports") / "model_architecture"
    with TemporaryDirectory(dir=temporary, prefix="cascade-") as directory:
        build = Path(directory)
        tex = build / f"{NAME}.tex"
        pdf = build / f"{NAME}.pdf"
        preamble = rf"""% Generated by scripts/draw_cascade_architecture.py.
\documentclass[border=8pt,tikz]{{standalone}}
\usepackage{{import}}
\subimport{{{(backend / 'layers').as_posix()}/}}{{init}}
"""
        with redirect_stdout(StringIO()):
            to_generate([preamble, to_cor(), make_layout(), to_end()], tex)
        subprocess.run([str(compiler), "--outdir", str(build), str(tex)], check=True)
        subprocess.run(["pdftoppm", "-singlefile", "-png", "-r", "300", str(pdf), str(build / NAME)], check=True)
        formats = [".tex", ".pdf", ".png"]
        if not args.preview:
            subprocess.run(["pdftocairo", "-svg", str(pdf), str(build / f"{NAME}.svg")], check=True)
            formats.append(".svg")

        output.mkdir(parents=True, exist_ok=True)
        for suffix in formats:
            artifact = f"{NAME}{suffix}"
            (build / artifact).replace(output / artifact)
            print(output / artifact)


if __name__ == "__main__":
    main()
