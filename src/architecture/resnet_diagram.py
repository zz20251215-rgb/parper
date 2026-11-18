from pathlib import Path


def make_tex(tool_root: Path) -> str:
    layers = (tool_root.resolve() / "layers").as_posix()
    parts = [r"""\documentclass[border=10pt,tikz]{standalone}
\usepackage{import}
\subimport{""" + layers + r"""/}{init}
\usetikzlibrary{calc,positioning}
\renewcommand{\familydefault}{\sfdefault}
\definecolor{encoder}{HTML}{0072B2}
\definecolor{decoder}{HTML}{009E73}
\definecolor{upsample}{HTML}{D55E00}
\definecolor{ink}{HTML}{263238}
\begin{document}
\begin{tikzpicture}[font=\sffamily\small,text=ink,
 flow/.style={-Stealth,line width=.8pt,draw=ink},
 up/.style={-Stealth,line width=.9pt,draw=upsample},
 skip/.style={-Stealth,dashed,line width=.8pt,draw=encoder},
 concat/.style={circle,draw=decoder,line width=.8pt,fill=white,inner sep=2.5pt,font=\sffamily\scriptsize\bfseries},
 stage/.style={align=center,font=\sffamily\small,text width=2.8cm,fill=white,inner sep=2pt},
 tensor/.style={font=\sffamily\scriptsize,fill=white,inner sep=1pt}]
\node[font=\sffamily\Large\bfseries] at (11.2,13.7) {ResNet34--U-Net};
\node[font=\sffamily\small] at (11.2,13.05) {24,436,241 parameters\quad |\quad ImageNet-pretrained encoder + randomly initialized decoder};
"""]

    def volume(name, x, y, size, color, title, tensor, banded=True):
        shape = "RightBandedBox" if banded else "Box"
        band = f",bandfill={color}!85!black,bandopacity=.7" if banded else ""
        parts.append(
            rf"\pic at ({x},{y},0) {{{shape}={{name={name},fill={color},opacity=.35,"
            rf"height={size},depth={size},width=2.5,scale=.2{band}}}}};" + "\n"
        )
        if color == "encoder":
            position = rf"stage,anchor=east,align=right] at ({x-.5},{y-.3})"
        elif color == "decoder" and name != "df":
            position = rf"stage,anchor=north west,align=left] at ({x+1.65},{y-.5})"
        else:
            position = rf"stage,anchor=south] at ({x+.25},{y+size*.1+.34})"
        parts.append(rf"\node[{position} {{{title}}};" + "\n")
        parts.append(rf"\node[tensor,anchor=north] at ({x+.25},{y-size*.1-.27}) {{$ {tensor} $}};" + "\n")

    volume("input", 0, 11, 9, "black!30", "RGB input", r"3\times512\times512", False)
    enc = [
        ("stem", 2, 9, 8, r"Stem: $7\!\times\!7$, s2\\BN + ReLU", r"64\times256\times256"),
        ("e1", 4, 7, 7, r"Layer 1 (3 blocks)\\MaxPool s2", r"64\times128\times128"),
        ("e2", 6, 5, 6, r"Layer 2 (4 blocks)\\s2", r"128\times64\times64"),
        ("e3", 8, 3, 5, r"Layer 3 (6 blocks)\\s2", r"256\times32\times32"),
        ("e4", 10, 1, 4, r"Layer 4 (3 blocks)\\s2", r"512\times16\times16"),
    ]
    for name, x, y, size, title, tensor in enc:
        volume(name, x, y, size, "encoder", title, tensor)
    dec = [
        ("d3", 13, 3, 5, 768, 256, 32, "e3"),
        ("d2", 15, 5, 6, 384, 128, 64, "e2"),
        ("d1", 17, 7, 7, 192, 64, 128, "e1"),
        ("d0", 19, 9, 8, 128, 32, 256, "stem"),
        ("df", 21, 11, 9, 32, 16, 512, None),
    ]
    for name, x, y, size, cin, cout, spatial, skip in dec:
        volume(name, x, y, size, "decoder", rf"Double conv\\${cin}\rightarrow{cout}$", rf"{cout}\times{spatial}\times{spatial}")
        if skip:
            parts.append(rf"\node[concat] ({name}cat) at ({x-.65},{y}) {{C}};" + "\n")
            parts.append(rf"\draw[skip] ({skip}-east) -- ({name}cat);" + "\n")
            parts.append(rf"\draw[flow] ({name}cat) -- ({name}-west);" + "\n")
    volume("out", 23.4, 11, 9, "upsample", r"$1\!\times\!1$ head\\Logits", r"1\times512\times512", False)
    parts.append(r"""
\draw[flow] (input-east) -- (stem-west) node[midway,sloped,above,font=\scriptsize] {Normalize};
\draw[flow] (stem-east) -- (e1-west);
\draw[flow] (e1-east) -- (e2-west);
\draw[flow] (e2-east) -- (e3-west);
\draw[flow] (e3-east) -- (e4-west);
\draw[up] (e4-east) -- (d3cat);
\draw[up] (d3-east) -- (d2cat);
\draw[up] (d2-east) -- (d1cat);
\draw[up] (d1-east) -- (d0cat);
\draw[up] (d0-east) -- (df-west);
\draw[flow] (df-east) -- (out-west);
\draw[skip] (2,-1.1) -- (3.2,-1.1);
\node[anchor=west,font=\small] at (3.3,-1.1) {Encoder skip};
\node[concat] at (7.3,-1.1) {C};
\node[anchor=west,font=\small] at (7.65,-1.1) {Channel concatenation};
\draw[up] (13,-1.1) -- (14.2,-1.1);
\node[anchor=west,font=\small] at (14.3,-1.1) {Bilinear upsample $\times2$};
\node[align=center,font=\small,text width=24cm] at (12,-2.15) {Double conv: $(3\!\times\!3\ \mathrm{Conv}+\mathrm{BN}+\mathrm{ReLU})\times2$.\quad BasicBlock: two $3\!\times\!3$ convolutions with an internal residual shortcut.\\Four encoder-to-decoder skips; the final upsample has no skip. ImageNet channel normalization is inside the model.};
\node[align=center,font=\scriptsize,text width=24cm] at (12,-3.0) {Encoder: 21,284,672 parameters. Decoder + head: 3,151,569 parameters. Tensor labels are $C\times H\times W$; volumes are schematic.\\No classification pooling or fully connected layer. Sigmoid and prediction thresholding are applied outside the network.};
\end{tikzpicture}
\end{document}
""")
    return "".join(parts)
