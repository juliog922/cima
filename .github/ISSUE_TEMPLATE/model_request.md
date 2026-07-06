---
name: Model request
about: Propose a model for the certified registry
labels: model-request
---

**Repository** (ORG/REPO, quant tag if GGUF):

**Preflight output** — run `cima vet ORG/REPO[:TAG] --preflight` and paste
the full report. Requests without it are converted to a question.

```
paste here
```

**Family status**: is a smaller member of this family already certified?
If yes, name the registry row. If no, this request needs a full on-GPU
`cima vet` run — state the hardware you can test on.

**Capabilities expected**: generate / vision / audio / embed
