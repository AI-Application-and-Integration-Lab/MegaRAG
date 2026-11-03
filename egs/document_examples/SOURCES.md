# Example document sources

These five two-page excerpts are included in MegaRAG and were selected from the MMLongBench-Doc 2k subset. They preserve the original page content; only page selection, PDF compression, and the PDF title metadata were changed. No full dataset download is required.

| Excerpt | Original document / creator | Original PDF pages (1-based) |
| --- | --- | --- |
| [paper.pdf](files/paper.pdf) | [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc/blob/main/documents/2401.18059v1.pdf), Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D. Manning | 3, 4 |
| [finance.pdf](files/finance.pdf) | [2015 Form 10-K](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc/blob/main/documents/NETFLIX_2015_10K.pdf), Netflix, Inc. | 40, 43 |
| [manual.pdf](files/manual.pdf) | [WH-1000XM4 Help Guide](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc/blob/main/documents/stereo_headset.pdf), Sony Corporation | 19, 20 |
| [brochure.pdf](files/brochure.pdf) | [2024 Camry brochure](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc/blob/main/documents/camry_ebrochure.pdf), Toyota | 1, 6 |
| [slides.pdf](files/slides.pdf) | [Vietnam Mobile Apps Report Q3 2015](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc/blob/main/documents/reportq32015-151009093138-lva1-app6891_95.pdf), Appota Inc. | 7, 10 |

Page numbers above count from the first PDF page, regardless of printed page labels. Excerpt pages are numbered 1–2 in the same order. [sources.json](sources.json) records the original filenames, download URLs, page selection, and SHA-256 checksums.

## Attribution and usage

Dataset: **MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations**, Yubo Ma et al., 2024. [Project](https://github.com/mayubo2333/MMLongBench-Doc) · [Paper](https://arxiv.org/abs/2407.01523) · [Dataset](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc).

The dataset's [usage notice](https://github.com/mayubo2333/MMLongBench-Doc#-license) specifies research use and [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). These third-party excerpts retain their source attribution and usage terms; they are not relicensed under MegaRAG's code license. Original document copyrights remain with their respective authors and publishers.

Tutorial questions adapt the dataset's tasks or ask directly about the selected pages. Reference answers were checked against these excerpts; they describe the source documents, including their historical dates.

Start with the [document tutorial](../../docs/examples.md).
