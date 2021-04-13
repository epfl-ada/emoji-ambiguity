from xml.etree import ElementTree as et

import pandas as pd


def cldr_anns_to_df(anns_path):
    xtree = et.parse(anns_path)
    xroot = xtree.getroot()
    df = {"emoji": [], "cldr_description": []}
    for ann in xroot.find("annotations"):
        if 'type' not in ann.attrib:
            description = list(map(str.strip, ann.text.split('|')))
            df["emoji"].append(ann.attrib['cp'])
            df["cldr_description"].append(description)
    return pd.DataFrame(df)
