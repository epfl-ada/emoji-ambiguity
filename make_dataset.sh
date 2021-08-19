#!/usr/bin/env bash
python src/data/scrape_emojipedia_categories.py
python src/data/em2ambiguity.py
python src/data/ambiguity2final.py
