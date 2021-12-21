### Code and data for: Ambiguity of Emojis: How Do we Interpret Emojis Outside of Contexts?


### Repository structure

------------

    ├── README.md          <- The top-level README
    ├── settings.py        <- Project constants: data paths used in the notebook
    ├── data               
    │   ├── emoji_imgs                        <- .png images of emojis used for plotting
    │   ├── final_dataset.csv.gz              <- dataset containing human annotations for emojis in the context-free setting
    │   ├── glove-twitter-200-ambiguity.bin   <- subset of glove gensim twitter embeddings for words from our dataset
    │   └── emoji_categories.pkl              <- scraped emojipedia categories
    │
    ├── notebooks          <- Jupyter notebooks
    │   └── final_notebook.ipynb   <- Notebook with plotting code for: Fig 1, 2, 3 and table 1
    │
    ├── figures            <- Generated figures, figure: 1, 2, 3
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to generate data
    │   │   ├── assign_emoji_categories           <- Functions to assign emoji categories
    │   │   ├── emoji_categorization.py           <- Dictionary with hand-crafted emoji categorization
    │   │   └── utils.py                          <- Helper functions to save files
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── emoji_plotting.py <- Functions to plot emoji scatter plots
    └──
 