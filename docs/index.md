![](assets/header.jpg)

# Neural Bidirectional Texture Function Compression and Rendering

[Luca Quartesan](luca@traverseresearch.nl), [Carlos Santos](santos.c@buas.nl)

## Context
Bidirectional Texture Function (BTF) is a data-driven representation of surface materials that can encapsulate complex behaviors, such as self-shadowing and interreflections, that would otherwise be too expensive to render. BTFs can be captured by taking samples from a combination of light and view directions.
Neural Networks can learn from BTF data; the current SOTA is called NeuMIP. We propose two changes (Sin, Cat) to improve this technique in terms of memory, performance and quality.
Finally, we show that Neural BTF can be easily integrated into rendering engines by implementing it in Mitsuba 2.

## Method
We propose two changes. We replace the ReLUactivations with a Sine (Sin) as presented in Siren and use a different multi-level sampling like the one proposed in Instant-NGP, where we concatenate (Cat) results from different levels instead of reducing them through interpolation.
We perform an ablation study over the two proposed changes to evaluate their impact on memory, performance and quality.
We run the experiments on two datasets:
1. UBO2014, real captures.
2. MBTF from NeuMIP, synthetic.



## Results
![](assets/table.jpg)

+ Our results show that our method achieves better compression quality while using less memory and performance than the baseline.
+ Neural BTF is part of the broader field of neural scene representation.

<iframe frameborder="0" class="juxtapose" width="100%" height="1280" src="https://cdn.knightlab.com/libs/juxtapose/latest/embed/index.html?uid=f580e43c-70ea-11ed-b5bd-6595d9b17862"></iframe>

## Discussion
+ When used in a simulation the material only behaves as a diffuse PDF.
+ We do not report performance results from a real-time implementation.

## Paper
**Neural Bidirectional Texture Function Compression and Rendering**
Luca Quartesan and Carlos Pereira Santos
+ [Paper preprint]()
+ [BibTeX](assets/quartesan22neubtf.bib)
+ [Code](https://github.com/Traverse-Research/NeuBTF)

## Citation
```bibtex
@article{quartesan2022neubtf,
  title   = "Neural Bidirectional Texture Function Compression and Rendering",
  author  = "Luca Quartesan and Carlos Pereira Santos",
  journal = "SIGGRAPH Asia 2022 Posters (SA '22 Posters), December 06-09, 2022",
  year = {2022},
  month = dec,
  numpages = {2},
  url = {https://doi.org/10.1145/3550082.3564188},
  doi = {10.1145/3550082.3564188},
  publisher = {ACM},
}

```