# 電子情報通信実験IV ～数理・データサイエンス実習～

## １．	全体の目的
電子情報通信工学に必要な数理・データサイエンスの基礎理論を学び，実習を通してデータ処理・分析・予測の技法を身に付け，電気エネルギー，電子物性デバイス，光エレクトロニクス，通信システムの各専門分野の基本的な事項の理解を深めることを目的とする。

## ２．	テーマの概要
本実習で扱うテーマについては，これまでに学習した電子情報通信工学に関わる専門知識を役立てる技術を養うとともに，それら知識への理解をより深め，社会に起きている変化や課題に取り組む力を身に着けることを狙いとして設定している。
　データサイエンスの基礎となる「データを読む」「データを説明する」「データを扱う」ことに取り組むテーマと電子情報通信に関わる現象への数理の応用や数値計算への理解を深めるシミュレーション実験のテーマに取り組んでもらう。以下に本実験で取り組むテーマをまとめる。

### データサイエンスの基礎
-	IV-1 多変量解析と可視化
-	IV-2 時系列解析と予測

### 電子情報通信シミュレーション
-	IV-3 デジタル変復調回路シミュレーション
-	IV-4 発振回路シミュレーション
-	IV-5 3相誘導電動機の特性

## ３．	実験・実習環境
テーマ毎に必要となる実験・実習環境が異なるため，詳細については各テーマ担当者からの指示に従って課題に取り組むこと。
ここでは，汎用的に役立つプログラミング言語Pythonの利用環境について簡単に紹介する。なお，すべてのテーマでPythonを利用するとは限らない。また，Python のプログラミング環境の構築についてはインターネットや書籍[1]などから多くの情報を入手可能で，その内容も多岐に渡る。利用の際は以下の例にとらわれず，各自に適した環境を用意すればよい。


### Google Colaboratory
Google Colaboratoryは，Google社が機械学習の教育や研究用に提供しているJupyter Notebook環境のクラウドサービスで，ブラウザ上でのPython実行を可能としている。ChromeブラウザとインターネットがあればPythonプロジェクトを進めることができる。接続して現れる「Colaboratoryへようこそ」のページにてColaboratory の概要などを参照してほしい。

- URL：https://colab.research.google.com/
- 利用方法：Googleアカウントでログイン
  -「ファイル」→「ノートブックを新規作成」

### Microsoft Visual Studio Code
Visual Studio (VS) Codeはソースコードエディタである。Windows，Linux，macOS上で動作し，無償で利用できる。別途，Python の環境を準備する必要があるが，VS Code にPython拡張機能をインストールすることで，Jupyter Notebook 機能もエディタ内で動作するようになる。

- URL：https://code.visualstudio.com/
- URL：https://code.visualstudio.com/docs/python/python-tutorial
- URL：https://www.python.org/downloads/
- 利用方法：ローカルのPCにダウンロードしてインストール
　　

### Mathworks MATLAB/Simulink
[![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=msiplab/EicLabIV)

新潟大学では，数値解析ソフトウェアMATLAB の大学包括ライセンス（ Campus-Wide License）を導入しており，すべての学生，教職員，および研究者が，キャンパスの内外を問わず，あらゆるデバイスでMATLAB，Simulink，およびオンライン学習ツールを無制限に使用することができる。

- URL：https://www.cais.niigata-u.ac.jp/service/software/matlab/

## ４．	レポート形式
レポート提出は学務情報システムからテーマ毎に設定する期日までに行ってもらう。A4サイズ縦置きPDF形式とする。PDFファイルは，MS Word やLaTeXなどのワープロソフトで作成すればよい。以下では，オンラインで利用できるLaTeX原稿執筆環境としてOverleafを，手書きのレポートをPDF化できるMicrosoft Office Lensを紹介する。

### Overleaf
Overleafは，オンラインで使える LaTeX 原稿執筆のクラウド環境である[8]。ブラウザ上でのLaTeXの実行環境を提供している。ブラウザとインターネットがあればプロジェクトを進めることができる。

- URL：https://www.overleaf.com/
- 利用方法：Googleアカウントなどでログイン

### Microsoft Office Lens
Microsoft Office Lens は写真をPDFに変更するアプリである。パソコンやスマートフォンで手書きのレポートをPDFに変換することができる。

- 利用方法：スマートフォンなどにアプリをインストール

## ５．	参考文献

1.	Pythonによるはじめての機械学習プログラミング／島田達朗ほか／技術評論社
1.	PythonユーザのためのJupyter[実践]入門／池内孝啓ほか／技術評論社
1.	Pythonデータサイエンスハンドブック／Jake VanderPlas／オライリージャパン
1.	東京大学のデータサイエンティスト育成講座／塚本邦尊ほか／マイナビ出版
1.	データサイエンスのための数学／椎名　洋ほか／講談社
1.	データサイエンスの基礎／浜田悦生ほか／講談社
1.  最適化手法入門／寒野善博ほか／講談社
1.  多変量解析入門―線形から非線形へ―／小西貞則／岩波書店
1.  スパース回帰分析とパターン認識／梅津佑太ほか／講談社
1.  時系列解析入門／北川源四郎／岩波書店
1.  時系列解析／島田直希／共立出版
1.  時系列解析入門〔第２版〕／宮野尚哉・後藤田浩／サイエンス社
1.  インストールいらずのLATEX入門 ―Overleafで手軽に文書作成／坂東慶太ほか／東京図書

## 付録

### 参考サイト

Python の自学習に役立つサイトを以下にまとめる。

1.  [新潟大学BDARC学生勉強会](https://bdarc.net/)

### 他のプログラミング言語

Pythonの代替となる他のプログラミング言語を以下にまとめる。

1. [MATLAB/Simulink](https://jp.mathworks.com/)
1. [Julia](https://julialang.org/)
1. [R](https://www.r-project.org/)

## リンク

- [プログラミングBI/BII](https://github.com/msiplab/EicProgLab)
- [電子情報通信設計製図](https://github.com/msiplab/EicDesignLab)

---
新潟大学工学部工学科　電子情報通信プログラム　村松正吾
