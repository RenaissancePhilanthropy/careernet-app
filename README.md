# CareerNet app

Two browser pages for exploring [CareerNet](https://github.com/RenaissancePhilanthropy/careernet-data),
a labelled dataset of 6,000 career questions from CareerVillage.org and 16,130 volunteer answers.

**Live site:** https://renaissancephilanthropy.github.io/careernet-app/

- **Explore the data** (`docs/index.html`): a dashboard that links the questions' occupation
  (SOC) codes to U.S. Bureau of Labor Statistics employment and wage figures, filterable by
  state and domain.
- **Search** (`docs/search/`): search the questions and answers by meaning rather than by
  keyword, and filter by occupation, scenario and goal. The companion
  [RAG notebook](https://github.com/RenaissancePhilanthropy/careernet-data/blob/main/Notebooks/careernet_retrieval_rag.ipynb)
  in careernet-data runs the same search in Google Colab and adds a chat model that answers
  career questions from the retrieved CareerNet answers.

Search runs entirely in your browser. On first use the page downloads the embedding model
(EmbeddingGemma, about 300 MB) from Hugging Face; the browser keeps it afterwards. Nothing you
type is sent anywhere.

## Running it locally

The pages load their data with `fetch()`, so they must be served over http. Opening
`index.html` directly from disk will not work.

```
cd docs
python -m http.server 8000
```

Then open http://localhost:8000.

## What's here

| Path | Holds |
|---|---|
| `docs/` | The site, published by GitHub Pages. `data.json` feeds the dashboard; `search/data/` holds the search index |
| `pipeline/` | Scripts that build the corpus, embeddings and search index, plus evaluation results. See [pipeline/README.md](pipeline/README.md) |
| `pipeline/packaging/` | Launchers for the offline zip version, which serves the app to your own machine only |

To rebuild the search index from the source data, clone
[careernet-data](https://github.com/RenaissancePhilanthropy/careernet-data), set
`CAREERNET_SRC` to its `Datasets/` folder and follow the steps in
[pipeline/README.md](pipeline/README.md). You only need to do this if the data or model
changes, not to run the site.

## Data and license

The CareerNet data is from Renaissance Philanthropy and The Learning Agency. Its labels,
sampling and collection are described in the
[careernet-data repository](https://github.com/RenaissancePhilanthropy/careernet-data).
Questions about the data: ulrich@renphil.org.

This repository is licensed under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE.txt), the same
license as the data.
