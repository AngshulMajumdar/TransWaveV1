# TransWave-Meyer-96M-4.5B model card

This repository contains the architecture and training recipe for the 96.13M-parameter TransWave model.

The release pretraining target is 4,499,996,672 raw-byte tokens. A language-model token in this release means one raw byte; there is no BPE or SentencePiece tokenizer. The learned lifting tokenizer is part of the neural model.

The final Hugging Face weights are intentionally not bundled in this GitHub archive. They are exported separately after the full pretraining run.

## Intended use

Research on causal sequence modeling, attention alternatives, multiscale spectral sequence operators, raw-byte language modeling, and architecture comparisons.

## Evaluation

The checked-in `benchmarks/final_causal_results` directory contains the exact final causal result JSON files used in the README tables and validation plots.

## Limitations

The historical Mamba-2, Hyena, and RetNet rows were produced under an older tokenized masked-language-model protocol. Their numerical NLL/PPL values are retained as recorded experimental baselines, but they are not objective-equivalent to the final causal raw-byte TransWave evaluation.
