# Architecture

TransWave is a causal raw-byte language model built around learned multiscale lifting and causal Meyer spectral mixing.

## Final configuration

| Component | Value |
|---|---:|
| Architecture ID | `causal_transwave_blocklifting_meyer_byte_lm_g4_v2` |
| Parameters | 96,130,040 |
| Byte vocabulary | 256 |
| Raw context | 32,768 bytes |
| Bytes per block | 8 |
| Block context | 4,096 |
| Model width | 768 |
| Layers | 12 |
| Learned scales | 8 |
| Meyer bands | 12 |
| Controls per band | 8 |
| Causal Meyer FIR | 512 block positions |
| Local causal convolution | kernel 7 |
| FFN | SwiGLU, width 2,816 |
| Within-block decoder | autoregressive GRU |

## Causal factorization

For each eight-byte block, the global representation is shifted by one completed block before decoding. The within-block decoder then predicts the eight bytes autoregressively with teacher forcing during training. The resulting likelihood is a genuine next-byte likelihood rather than a masked-reconstruction score.

## Layer structure

Each TransWave layer combines:

1. content-dependent cross-scale mixing;
2. a strictly causal Meyer FIR implemented as zero-padded FFT linear convolution;
3. a strictly left-causal depthwise local convolution;
4. residual gating; and
5. a SwiGLU feed-forward block.

The local convolution provides a short finite-memory path while the causal Meyer operator supplies substantially longer multiscale memory.
