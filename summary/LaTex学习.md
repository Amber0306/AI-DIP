部分内容参考https://zhuanlan.zhihu.com/p/108779897

整体内容为https://www.bilibili.com/video/BV15b411j7Au?p=9&spm_id_from=pageDriver学习笔记

# 一.命令行LaTex

## 1.查看版本

tex -v

latex -v

xelatex -v

## 2.过程

notepad test.tex

新建test.tex文件;

latex test.tex

生成test.dvi中间文件；

dvipdfmx test.dvi

生成test.pdf文件。

### 中文的处理

xelatex test.tex

编译并直接生成pdf文件，可以处理unicode编码文件，并没有处理中文信息。

文件需保存为UTF-8编码格式，才能处理中文，同时引入宏包。

\usepackage{ctex}

# 二、IDE 实现文档编写

texStudio

修改命令 xeLatex

# 三、LaTex源文件结构

导言区和正文区

```latex
%导言区
\documentclass{article}
% book, report letter

\title{my first document}
\author{Amber}
\date{\today}


% 正文区,表示注释
\begin{document}
	
	\maketitle
	% 文本模式
	hello,world!
	
	Let $f(x)$ % 数学模式
	be defined by the fumula
	$$f(x)=3x^2+x-1$$
	% 两个$$表示单行公式，而不是行内公式
	which is a polynomial of degree 2.
	
\end{document}
```

# 四、中文处理方法

```latex
%导言区 全局设置
\documentclass{ctexart}
% ctexbook,ctexrep

%\usepackage{ctex}

\newcommand\degree{^\circ}

\title{\heiti 勾股定理}
\author{Amber}
\date{\today}


% 正文区,表示注释
\begin{document}
	
	\maketitle
	
	直角三角形斜边的平方等于两腰的平方和。
	
	可以用符号语言表述为：设直角三角形$ABC$，其中$\angle C=90\degree$，则有：
	%用于产生带编号的行间公式
	\begin{equation}
	AB^2 = BC^2 + AC^2
	\end{equation}
	
	
\end{document}
```

编码模式需要设置成utf-8

## 查看文档

C:\Users\Administrator>texdoc ctex

C:\Users\Administrator>texdoc lshort-zh

# 五、字体字号设置

**五种属性**

## 字体编码

正文字体编码：OT1,T1,EU1等

数学字体编码：OML, OMS, OMX等

## 字体族

罗马字体：笔画起始处有装饰

无衬线字体：笔画起始处无装饰

打字机字体：每个字符宽度相同，等宽字体

## 字体系列

粗细

宽度

## 字体形状

直立

斜体

伪斜体

小型大写

## 字体大小

无

```latex
\documentclass[10pt]{article}

\usepackage{ctex}

\newcommand{\myfont}{\textit{\textbf{\textsf{Fancy Text}}}}

\begin{document}
	%字体族设置（罗马字体、无衬线字体、打字机字体
	\textrm{Roman Family}
	\textsf{Sans Serif Family}
	\texttt{Typewriter Family}
	
	% 后续所有为该字体
	\rmfamily Roman Family
	\sffamily Sans Serif Family
	\ttfamily Typewriter Family
	% 限定范围
	{\rmfamily Roman Family}
	hello hello hello
	
	%字体系列设置（粗细、宽度）
	\textmd{Medium Series}
	\textbf(Boldface Series)
	{\mdseries Medium Series}
	{\bfseries Boldface Series}
	
	%字体形状设置（直立、斜体、伪斜体、小型大写）
	\textup{Upright Shape}
	\textit{Italic Shape}
	\textsl{Slanted Shape}
	\textsc{Small Caps Shape}
	{\upshape Upright Shape}
	{\itshape Italic Shape}
	{\slshape Slanted Shape}
	{\scshape Small Caps Shape}
	
	%中文字体
	{\songti 宋体}\quad
	{\heiti 黑体}\quad
	{\fangsong 仿宋}\quad 
	{\kaishu 楷书}
	
	中文字体的\textbf{粗体}和\textit{斜体}
	
	%字体大小设置
	{\tiny Hello}\\
	{\scriptsize Hello}\\
	{\footnotesize Hello}\\
	{\small Hello}\\
	{\normalsize Hello}\\
	
	%中文字号设置
	\zihao{5} 你好
	
	% 格式与内容分离
	\myfont 我的字体
	
\end{document}
```

# 六、文档基本结构

```latex
\documentclass{ctexbook}

\usepackage{ctex}

% 可对格式进行设置
\ctexset{}

\begin{document}
	
	\tableofcontents
	
	\chapter{引言}
	作为图像处理领域的一个分支，去雾技术在卫星拍摄、航空拍摄以及水下拍摄等方面具有重要的应用意义。\par 一方面，由于自然景深的存在，距离较远的物体在拍摄时，很可能会被雾气覆盖，另一方面，有雾天气条件下，雾气的遮罩会导致图像不清晰。这是由于光线传播时，大气中粒子具有散射现象，图像的细节信息丧失了很多，这不利于对图像做出后续的处理，因此图像去雾也具有很大的研究意义。
	
	本文所述的暗通道去雾则是属于复原除雾算法，是利用先验信息由物理模型进行去雾的方法。\\其实现简单，效果良好，在不同场景下的表现相对优异，但存在一些缺点，因此该算法仍待优化。
	
	\section{实验方法}
	\section{实验结果}
	\subsection{数据}
	\subsection{图标}
	\subsubsection{实验条件}
	\subsubsection{实验过程}
	\subsection{结果分析}
	\chapter{结论}
	\section{致谢}
\end{document}
```

# 七、特殊字符

空行分段，多个空行等于1个

自动缩进，绝对不能使用空格代替

英文中多个空格处理为一个空格，中文中空格将被自动忽略

汉字与其他字符的间距会自动由XeLaTex处理

禁止使用中文全角空格

```latex
%文档基本结构

\documentclass{article}
\title{First Tex File}
\author{Andy}
\date{\today}

\usepackage{ctex}
%正文区
\begin{document}
	\section{空白符号}
	实在学不了唱歌，        能把主子伺候好，跟老母亲我一起去淘宝直播 卖宠物用品也是极好的啊，我学生家里就是卖宠物用品的，        中日混血，家里有一群猫狗，一墙的仓鼠笼子（我看过照片，大概有几十个的样子）
	\section{\ LaTex 控制符}
	\# \$ \{  \} \~{} \_{} \^{} \textbackslash
	\section{排版符号}
	\S \P \dag \ddag \copyright \pounds
	\section{\ Tex 标志符号}
	\ TeX{} \ LaTeX{}  \ LaTeXe{}
	\section{引号}
	`' `` ''   ``被引号包裹'' %  `表示单引号的左边，'表示单引号的右边
	\section{连字符}
	- -- ---
	\section{非英文字符}
	\section{重音符号}
	
	a\quad b 1em的空白\\
	a\qquad b 2em的空白\\
	a\,b a\thinspace b 1/6em空白\\
	a\enspace b 0.5em\\
	a\ b\\
	a~b 硬空格
	a\kern 1pc b
	a\kern -1em b
	a\hskip 1em b
	%a\hskip{35pt}b
	
	%占位宽度
	a\hphantom{xyz}b
	
	%弹性长度
	a\hfill b
	
	
\end{document}
```

# 八、插图

```latex
%文档基本结构
\documentclass{ctexart}

\usepackage{ctex}
\usepackage{graphicx}

\graphicspath{{figures/}}
%表示图片在当前目录下的figures目录

%正文区
\begin{document}
	\LaTeX{}中的插图：
	\includegraphics[scale=0.3]{one.jpg}
	\includegraphics{two}%two是figures文件夹下的文件(图像)
\end{document}
```

可选参数可修改其大小旋转角度等等。

# 九、表格

texdoc booktab

texdoc longtab

```latex
\documentclass{ctexart}

\title{First Tex File}
\author{Andy}
\date{\today}

\usepackage{ctex}
%正文区
\begin{document}
	\begin{tabular}
		{||l|c|c|p{1.5cm}| r|}
		%会有5列，指定每列的对其形式,|表示每列中间有竖线分开
		\hline
		%每行之间由横线分开
		姓名&语文&数学&外语&政治\\
		%\\表示换行
		\hline\hline
		张三&87&120&补考另行通知&10\\
		\hline
		张1&87&120&25&36\\
		\hline
		张2&87&120&25&36\\
		\hline
	\end{tabular}
	
\end{document}
```

# 十、浮动体

插入图片和表格，设置格式，与添加引用

```latex
%文档基本结构
\documentclass{ctexart}

\usepackage{graphicx}

\usepackage{ctex}

\graphicspath{{E:/docs/letax/figure}}

%正文区
\begin{document}
	\LaTeX{} 中的插图：
	%交叉引用
	\ref{fig-ppp}

	\begin{figure}[htbp]
		%h,表示此处，代码所在上下文位置
		%t，表示顶部，代码所在页面或者之后页面的顶部
		%b，页面底部，代码所在页面或之后页面的底部
		%p，独立一页，浮动页面
		\centering
		\includegraphics[scale=0.3]{ppp.jpg}
		\caption{JPG}
		\label{fig-ppp}
	\end{figure}


	\begin{table}
		\centering
		\caption{考试成绩单}
		\begin{tabular}{|l|c|c|c|r|}
			\hline
			姓名　&　语文&数学&外语&备注\\
			\hline
			张三&87&93&100&优秀\\
			\hline
			王一&80&80&20&补考另行通知\\
			\hline
		\end{tabular}
	\end{table}
\end{document}
```

# 十一、数学公式初步

```latex
\documentclass{article}
\usepackage{ctex}

\usepackage{amsmath}


%正文区
\begin{document}
	\section{简介}
	\LaTeX{}
	排版内容分为文本模式和数学模式。
	\section{行内公式}
	\subsection{美元符号}
	交换律是$a+b=b+a$
	\subsection{小括号}
	交换律是\(a+b=b+a\)
	\subsection{math环境}
	交换律是
		\begin{math}
			a+b=b+a
		\end{math}
	\section{上下标}
	\subsection{上标}
	$3x^{20} - x + 2 = 0$\\
	$3x^{3x^{20} - x + 2 = 0} - x + 2 = 0$
	\subsection{下标}
	$a_0,a_1,...,a_{3x^{20}}$
	\section{希腊字母}
	$\alpha$ $\pi$
	\section{数学函数}
	$\log$ $\arccos$\\
	$\sqrt[3]{x^2+y^2}	$
	\section{分式}
	$3/4$
	$\frac{3}{4}$
	\section{行间公式}
	\subsection{美元符号}
	$$a+b=b+a$$
	\subsection{中括号}
	交换律是\[a+b=b+a\]
	\subsection{displaymath环境}
		\begin{displaymath}
		a+b=b+a
		\end{displaymath}
	\subsection{自动编号的equation环境}
	交换律公式 \ref{commutative}
		\begin{equation}
		a+b=b+a \label{commutative}
		\end{equation}
	\subsection{不编号公式的equation环境}
		\begin{equation*}
		a+b=b+a
		\end{equation*}
\end{document}
```



# 十二、数学公式-矩阵

```latex
\documentclass{ctexart}

\usepackage{amsmath}

\begin{document}
	\[
	\begin{matrix}
		0&1\\
		1&0
	\end{matrix}
	\]
	%pmatrix, bmatrix,Bmatrix, vmatrix
	%省略号 \dots, \vdots, \ddots
	
	\[\begin{pmatrix}%括号包裹的矩阵
		0&1\\
		1&0
	\end{pmatrix}
	\]	
	
	\[\begin{vmatrix}%长竖线包裹的矩阵
		0&1\\
		1&0
	\end{vmatrix}
	\]
	
	\[\begin{bmatrix}%长中括号包裹的矩阵
		0&1\\
		1&0
	\end{bmatrix}
	\]
	
	
	\[\begin{pmatrix}%括号包裹的矩阵
		a_{11}^2&a_{12}^2&a_{13}^2\\
		0&a_{22}&a_{33}
	\end{pmatrix}
	\]
	
	\[\begin{bmatrix}%长中括号包裹的矩阵
		a_{11}&\dots&a_{1n}\\
		&\ddots&\vdots\\
	\end{bmatrix}_{n \times n}
	\]
	
	\[\begin{pmatrix}%分块矩阵(矩阵嵌套)
		\begin{matrix}
			1&0\\0&1
		\end{matrix}
		& \text{\Large 0}\\
		\text{\Large 0}&\begin{matrix}
			1&0\\0&1
		\end{matrix}
	\end{pmatrix}
	\]
	
	\[\begin{pmatrix}%括号包裹的矩阵
		a_{11}&a_{12}&\cdots&a_{ln}\\
		&a_{22}&\cdots&a_{2n}\\
		&		&\dots &\vdots \\
		\multicolumn{2}{c}{\raisebox{1.3ex}[0pt]{\Huge 0}}
		&		&a_{nn}
	\end{pmatrix}
	\]
	
	
	\[\begin{pmatrix}%跨列的省略号：\hdotsfor{<列数>}
		1&\frac 12 &\dots &\frac ln \\
		\hdotsfor{4}\\
		m&\frac m2& \dots &\frac mn
	\end{pmatrix}
	\]
	
	%行内小矩阵(smallmatrix)环境
	复数$z=(x,y)$也可以用矩阵
	\begin{math}
		\left(%需手动加上左括号
		\begin{smallmatrix}
			x& -y\\y&x
		\end{smallmatrix}
		\right)%需手动加上右括号
	\end{math}来表示
	
	%array环境(类似表格环境tabular)
	\[
	\begin{array}{r|r}
		\frac 12&0\\
		\hline
		0& -\frac abc\\
	\end{array}
	\]
\end{document}
```

# 十三、多行公式

```latex
\documentclass{article}

\usepackage{ctex}
\usepackage{amsmath}
\usepackage{amssymb}
%正文区
\begin{document}
	%gather和gather*环境(可以使用\\换行)
	%带编号
	\begin{gather}
		a+b=b+a\\
		ab  ba
	\end{gather}
	
	%不带编号
	\begin{gather*}
		3+5=5+3\\
		3 \times 5=5\times 3
	\end{gather*}
	
	%在\\前使用\notetag阻止编号
	\begin{gather}
		3^2+4^2=5^2 \notag
	\end{gather}
	
	%align和align*环境(用&对齐)
	%带编号
	\begin{align}
		x &=t+\cos t+1\\
		y &=2 \sin t
	\end{align}
	%不带编号
	\begin{align*}
		x &=t+\cos t+1\\
		y &=2 \sin t
	\end{align*}
	
	%split环境(对齐采用align环境的方式，编号在中间)
	\begin{equation}
		\begin{split}
			\cos 2x &=\cos^2 x- \sin^2 x\\
			&=2\cos^2 x-1
		\end{split}
	\end{equation}		
	
	%case环境
	%每行公式中使用&分隔为两部分
	%通常表示值和后面的条件
	\begin{equation}
		D(x)=\begin{cases}
			1,& \text{如果} x \in \mathbb{Q};\\
			0,& \text{如果} x \in \mathbb{R}\setminus\mathbb{Q}
		\end{cases}
	\end{equation}
\end{document}
```



# 十四、参考文献BibTex

```latex
\documentclass{article}
\usepackage{ctex}

\begin{document}
	
	引用一篇文章\cite{article1},引用一本书\cite{book1}
	
	\begin{thebibliography}{99}
		\bibitem{article1}马化腾，雷军，李彦宏，张一鸣.\emph{基于LaTex的Web数学公式提取方法研究}[J].计算机科学.2014(06)
		\bibitem{book1}Andy H,Bob,Cat,\emph{what does the fox say}
	\end{thebibliography}	
\end{document}
```

```latex
@BOOK{mittelbach2004,
title={腾讯传},
publisher={广东教育出版社},
year={2004},
author={Frank Mittelbach and Michel Goossens},
series={Tools and Techniques},
address={广东},
edition={First}
}
```

```latex
引用一篇文章\cite{article1},引用一本书\cite{book1}
\begin{thebibliography}{99}
\bibitem{article1}马化腾，雷军，李彦宏，张一鸣.\emph{基于LaTex的Web数学公式提取方法研究}[J].计算机科学.2014(06)
\bibitem{book1}Andy H,Bob,Cat,\emph{what does the fox say}
这是一个文献引用：\cite{mittelbach2004}
\bibliography{cite1}
\end{thebibliography}
```



# 十五、参考文献BibLaTex

```latex

```



# 十六、自定义命令和环境

```latex

```



