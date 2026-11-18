from pathlib import Path


def make_tex(tool_root: Path) -> str:
    layers = (tool_root.resolve() / "layers").as_posix() + "/"
    parts = [r"""\documentclass[border=9pt,tikz]{standalone}
\usepackage{import}
\subimport{""" + layers + r"""}{init}
\usepackage{amsmath}
\usetikzlibrary{positioning,calc,arrows.meta}
\definecolor{encoder}{HTML}{0072B2}
\definecolor{decoder}{HTML}{009E73}
\definecolor{upsample}{HTML}{D55E00}
\begin{document}
\begin{tikzpicture}[font=\sffamily\small,
 flow/.style={-{Stealth[length=2mm]},line width=.8pt,draw=black!65},
 skip/.style={flow,dashed,draw=black!45},
 up/.style={flow,draw=upsample},
 concat/.style={circle,draw=black!65,fill=white,inner sep=2.7pt,font=\sffamily\small},
 info/.style={font=\sffamily\footnotesize,fill=white,inner sep=2pt}]
\node[font=\sffamily\bfseries\Large] at (7,2.3) {Basic U-Net};
\node[font=\sffamily\small] at (7,1.65) {7,760,097 parameters \quad $512\times512$ input};
"""]

    def block(name, x, y, channels, resolution, size, color, title):
        parts.append(rf"""
\pic at ({x},{y},0) {{RightBandedBox={{name={name},width={{2.4,2.4}},height={size},depth={size},fill={color},bandfill={color}!75!black,opacity=.6,bandopacity=.85,xlabel={{{{"",""}}}},ylabel=,zlabel=,caption=}}}};
\node[above=5pt,font=\sffamily\bfseries\small] at ({name}-north) {{{title}}};
\node[below=16pt] at ({name}-south) {{${channels}\times {resolution}\times {resolution}$}};
""")

    for i, (ch, res, size) in enumerate([(32, 512, 9), (64, 256, 7.6), (128, 128, 6.4), (256, 64, 5.4)]):
        block(f"e{i}", 0, -4.2 * i, ch, res, size, "encoder", f"Encoder {i + 1}")
        block(f"d{i}", 12, -4.2 * i, ch, res, size, "decoder", f"Decoder {4 - i}")
        parts.append(rf"""
\node[concat] (c{i}) at (10.3,{-4.2 * i}) {{C}};
\draw[skip] (e{i}-east) -- node[info,above] {{skip: ${ch}$ channels}} (c{i}.west);
\draw[flow] (c{i}.east) -- node[info,above] {{${2 * ch}$ ch.}} (d{i}-west);
""")
        if i:
            parts.append(rf"""
\draw[flow] (e{i - 1}-southwest) -- ++(-1.6,0) |- node[info,near end,above] {{$2\times2$ pool}} (e{i}-west);
\draw[up] (d{i}-east) -- (14.2,{-4.2 * i}) -- (14.2,{-4.2 * i + 2.0}) -- (10.3,{-4.2 * i + 2.0}) -- (c{i - 1}.south);
\node[info,text=upsample] at (11.5,{-4.2 * i + 2.0}) {{$2\times2$ up}};
""")
    block("b", 5.4, -16.5, 512, 32, 4.6, "encoder", "Bottleneck")
    parts.append(r"""
\draw[flow] (e3-southwest) -- ++(-1.6,0) |- node[info,near end,above] {$2\times2$ pool} (b-west);
\draw[up] (b-east) -- (10.3,-16.5) -- node[info,right=3pt,text=upsample] {$2\times2$ up} (c3.south);
\pic at (-3.6,0,0) {Box={name=input,width=.9,height=9,depth=9,fill=black!25,opacity=.5,xlabel={{"",""}},ylabel=,zlabel=,caption=}};
\node[above=5pt,font=\sffamily\bfseries] at (input-north) {Input};
\node[below=16pt] at (input-south) {$3\times512\times512$};
\draw[flow] (input-east) -- (e0-west);
\pic at (16,0,0) {Box={name=out,width=.9,height=9,depth=9,fill=upsample,opacity=.7,xlabel={{"",""}},ylabel=,zlabel=,caption=}};
\node[above=5pt,font=\sffamily\bfseries] at (out-north) {Logits};
\node[below=16pt] at (out-south) {$1\times512\times512$};
\draw[flow] (d0-east) -- node[info,above] {$1\times1$ conv} (out-west);
\node[anchor=west,align=left,font=\sffamily\footnotesize] at (-3.6,-19.5) {
\textcolor{encoder}{\rule{8pt}{8pt}} Encoder / bottleneck \qquad
\textcolor{decoder}{\rule{8pt}{8pt}} Decoder \qquad
\textcolor{upsample}{\rule{8pt}{8pt}} Transposed convolution / output head\\[5pt]
Each paired cuboid: $(3\times3\ \mathrm{Conv}+\mathrm{ReLU})\times2$; no batch normalization.\\[4pt]
C: channel concatenation. Up: $2\times2$ transposed convolution, stride 2. Pool: max pooling, stride 2.\\[4pt]
Tensor labels: channels $\times$ height $\times$ width. Cuboid dimensions are schematic.\\[4pt]
Inference only: sigmoid followed by threshold 0.5; these are outside the trainable model.
};
\end{tikzpicture}
\end{document}
""")
    return "".join(parts)
