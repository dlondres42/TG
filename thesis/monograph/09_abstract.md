# Abstract / Resumo

> **Submission note.** CIn/UFPE conventionally requires both a Portuguese
> *Resumo* and an English *Abstract*. Keep both ~250 words; strict TG limit
> for either is usually 500 words. The Resumo and Abstract should mirror
> each other in structure (context — problem — method — findings —
> conclusion); they are not direct translations but matched summaries.

## Resumo (PT-BR)

<<TODO: 250-word resumo cobrindo:

- **Contexto** — serviço web de inferência de ML para dados tabulares;
  latência de cauda (P99) como métrica operacional dominante.
- **Problema** — para inferência barata de árvores de decisão em CPU
  (XGBoost, ~14 µs por linha), o batching adaptativo compensa o seu
  custo de plumbing?
- **Método** — comparação quantitativa entre FastAPI + ONNX Runtime e
  BentoML + batching adaptativo num matriz de 219 células (workers ×
  threads × RPS) sob carga aberta (Vegeta), decomposição em três camadas
  (L1 inferência, L2 HTTP no host, L3 containerizado), métrica de
  goodput-under-SLA para justa comparação entre sistemas que abandonam
  carga e os que enfileiram, evidência de mecanismo via CPU
  observado por `docker stats`.
- **Achados** — FastAPI sustenta 99.7–100 % de goodput com SLA de 50 ms
  em todo o matrix; BentoML colapsa entre 350 e 600 RPS por workers; o
  P99 cresce duas ordens de grandeza no mesmo intervalo. A causa é a
  corrotina dispatcher do BentoML, que serializa o caminho de cada
  requisição: a evidência direta de CPU confirma que o BentoML consome
  ~2.5× mais CPU que o FastAPI sob carga idêntica e ainda assim falha.
- **Conclusão** — o batching adaptativo é a ferramenta certa para os
  casos para os quais foi desenhada (GPU, redes profundas), e não para
  inferência tabular barata em CPU. A escolha entre os stacks deve
  considerar o custo *intrínseco* da inferência.>>

## Abstract (EN)

<<TODO: 250-word abstract covering:

- **Context** — web-tier ML inference serving for tabular data; tail
  latency (P99) as the dominant operational metric.
- **Problem** — for cheap tree-ensemble inference on CPU (XGBoost,
  ~14 µs per row), does adaptive batching pay off, or does its per-
  request plumbing cost exceed the amortisation benefit?
- **Method** — quantitative comparison between FastAPI + ONNX Runtime
  and BentoML + adaptive batching across a 219-cell matrix (workers ×
  threads × RPS) under open-loop Vegeta load, three-layer latency
  decomposition (L1 inference, L2 host HTTP, L3 containerised),
  fairness-corrected goodput-within-SLA metric, and direct
  CPU-utilisation evidence for the mechanism.
- **Findings** — FastAPI sustains 99.7–100 % goodput-within-50ms-SLA
  across the entire matrix; BentoML collapses between 350 and 600 RPS
  scaling with worker count, with P99 climbing by two orders of
  magnitude over the same RPS range. The cause is BentoML's per-worker
  dispatcher coroutine, which serialises the request-handling path: CPU
  evidence shows BentoML using ~2.5× the CPU of FastAPI at identical
  load and still failing.
- **Conclusion** — adaptive batching is the right tool for the workloads
  it was designed for (GPU, deep networks), not for cheap tabular CPU
  inference. The choice between serving stacks must account for the
  *intrinsic* cost of inference.>>

## Keywords / Palavras-chave

**EN.** ML inference serving; tail latency; P99; FastAPI; BentoML; ONNX
Runtime; adaptive batching; open-loop benchmarking; Coordinated Omission;
XGBoost; tabular data.

**PT.** servidores de inferência de ML; latência de cauda; P99; FastAPI;
BentoML; ONNX Runtime; batching adaptativo; benchmarking de carga aberta;
omissão coordenada; XGBoost; dados tabulares.
