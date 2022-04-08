### Code and data for: On the Context-Free Ambiguity of Emoji


### Repository structure

------------

    ├── README.md          <- The top-level README
    ├── settings.py        <- Project constants: data paths used in the notebook
    ├── data               
    │   ├── emoji_imgs                        <- .png images of emojis used for plotting
    │   ├── emoji_annotations.csv.gz          <- Dataset containing human annotations for emojis in the context-free setting along with assigned fine-grained emoji categories
        ├── ambiguity_scores.csv.gz           <- Dataset containing emojis' semantic variations and vocabularies
    │   ├── glove-twitter-200-ambiguity.bin   <- Subset of glove gensim twitter embeddings for words from our dataset
    │   ├── emoji_categories_x.csv            <- Annotations of symbolic levels from author x
    │   ├── emoji_categories_y.csv            <- Annotations of symbolic levels from author y
    │   └── emoji_categories.pkl              <- Scraped emojipedia categories
    │
    ├── notebooks          <- Jupyter notebooks
    │   └── final_notebook.ipynb   <- Notebook with plotting code for: Fig 1, 2, 3 and table 1
    │
    ├── figures            <- Generated figures, figure: 1
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
 
 
###  System requirements and installation guide

We recommend a local installation of new Python virtual environment. The code was tested on Ubuntu 18.04.
Please use the packages versions provided in requirements.txt


1. Open the terminal. To avoid any incompatibility issue,
 please create a new virtual environment. This project was created using [virtualenvwrapper](]https://virtualenvwrapper.readthedocs.io/en/latest/)

`pip install virtualenvwrapper` <br>
`mkvirtualenv emoji-ambiguity -r requirements.txt -p python3.7` <br>

The environment should be activated automatically, if not use: <br>
`workon emoji-ambiguity`

To deactivate the environment simply use: <br>
`deactivate`

2. Start a Jupyter notebook server. <br>
`cd notebooks` <br>
`jupyter notebook`

4. Select "Cells -> Run all". The figures and table with emoji categories and their descriptions will be displayed in the browser, figures will be saved in emoji-ambiguity/figures folder.

### Cite us
Please cite as:
On the Context-Free Ambiguity of Emoji. Justyna Częstochowska, Kristina Gligorić, Maxime Peyrard, Yann Mentha, Michal Bien, Andrea Grütter, Anita Auer, Aris Xanthos and Robert West.
In Proc. of the International AAAI Conference on Web and Social Media ICWSM, 2022.
