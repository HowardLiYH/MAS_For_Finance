# NeurIPS 2026 Paper Draft

## Population-Based Continual Learning for Multi-Agent LLM Trading Systems

This directory contains the LaTeX source for the NeurIPS 2026 submission.

## Files

- `neurips_2026.tex` - Main paper source
- `neurips_2024.sty` - NeurIPS style file (download from conference website)

## Building

```bash
# Download NeurIPS style file
wget https://media.neurips.cc/Conferences/NeurIPS2024/Styles/neurips_2024.sty

# Build PDF
pdflatex neurips_2026.tex
bibtex neurips_2026
pdflatex neurips_2026.tex
pdflatex neurips_2026.tex
```

## TODO: Sections to Complete

### Experimental Results (Section 5)
- [ ] Run backtesting experiments on 2022-2024 data
- [ ] Fill in Table 3: Main results
- [ ] Fill in Table 4: Per-asset analysis
- [ ] Fill in Tables 5-7: Ablation studies
- [ ] Add learning curve figures

### Analysis (Section 6)
- [ ] Analyze parameter transfer patterns
- [ ] Document pipeline synergies
- [ ] Test across market regimes (bull/bear/sideways)
- [ ] Compare GPT-4 vs DeepSeek vs Claude

### Figures Needed
1. Population score over iterations (learning curve)
2. Diversity over iterations
3. Best pipeline evolution
4. Variant selection heatmap by market regime
5. Shapley value distribution

## Paper Structure

1. **Introduction** (1 page) - Problem, insight, contributions
2. **Related Work** (0.5 page) - LLM agents, PBT, MARL, continual learning
3. **Method** (2.5 pages) - Architecture, transfer, scoring, diversity
4. **Experimental Setup** (1 page) - Data, baselines, metrics
5. **Results** (1.5 pages) - Main results, ablations
6. **Analysis** (1 page) - Insights
7. **Discussion** (0.5 page) - Limitations
8. **Conclusion** (0.25 page)

Total: ~8 pages (NeurIPS limit)

## Key Contributions

1. **Population-based multi-agent framework** - Heterogeneous agent populations
2. **Knowledge transfer strategies** - Soft update, distillation, selective
3. **Shapley value credit assignment** - Fair evaluation in coupled pipelines
4. **Diversity preservation** - Prevent population collapse

## Citation

```bibtex
@inproceedings{author2026popagent,
  title={Population-Based Continual Learning for Multi-Agent LLM Trading Systems},
  author={Author, First and Author, Second},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```
