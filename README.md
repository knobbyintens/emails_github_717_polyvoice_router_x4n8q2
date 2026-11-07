# PolyVoice

Official implementation for **PolyVoice: Mixture-of-Experts Routing for
Multilingual Speech Translation**, accepted at Interspeech 2025.

## Repository Status

Released. The full training, inference and evaluation pipeline is available in
this repository, including the released checkpoint, the language-aware router
and the scripts used to produce the main results table.

## Layout

```
src/
  router.py      mixture-of-experts routing layer
  model.py       speech encoder + translation decoder
  train.py       training entry point
eval/
  translate.py   inference and BLEU evaluation
experiments/
  interspeech_main.yaml
results/
  main_table.json
```

## Usage

```bash
pip install -r requirements.txt
bash scripts/train.sh experiments/interspeech_main.yaml
bash scripts/evaluate.sh checkpoints/polyvoice.pt
```

## Citation

```bibtex
@inproceedings{scott2025polyvoice,
  title={PolyVoice: Mixture-of-Experts Routing for Multilingual Speech Translation},
  author={Scott, Jeffrey and Nakamura, Haru and Okafor, Daniel},
  booktitle={Interspeech},
  year={2025}
}
```
