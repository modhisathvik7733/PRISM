"""PRISM-Lang pretraining — Path B of the language thesis test.

Reuses the v3.0 architecture (prism/lang/) to learn English FROM
SCRATCH on TinyStories, then fine-tunes on reasoning tasks. This is
the thesis-pure path: no pretrained weights are loaded; everything
the model "knows" comes from gradient signal on text.

Also pretrains a matched-param vanilla AR baseline on the same data
+ compute. The downstream comparison isolates the JEPA-middle's
contribution from the effects of pretraining itself.

Components (all new — nothing under prism/ outside this package or
prism/lang/ is modified):
  - corrupt.py    : T5-style span corruption (the structured-model
                    pretraining objective)
  - vanilla_ar.py : matched-param decoder-only baseline (next-token
                    pretraining objective)
"""
