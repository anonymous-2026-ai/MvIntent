import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The baseline score, shown in bold, is obtained with no context window and is comparable to the results reported by CITATION0 .")

svg = displacy.render(doc, style="dep", jupyter=False)
with open("dep.svg", "w", encoding="utf-8") as f:
    f.write(svg)
