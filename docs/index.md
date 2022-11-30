## NeuBTF
> Neural Bidirectional Texture Function Compression and Rendering

Luca Quartesan             |  Carlos Pereira Santos
:---:|:---:
luca@traverseresearch.nl |  santos.c@buas.nl

### Context
Bidirectional Texture Function (BTF) is a data-driven representation of surface materials that can encapsulate complex behaviors, such as self-shadowing and interreflections, that would otherwise be too expensive to render. BTFs can be captured by taking samples from a combination of light and view directions.
Neural Networks can learn from BTF data; the current SOTA is called NeuMIP. We propose two changes (Sin, Cat) to improve this technique in terms of memory, performance and quality.
Finally, we show that Neural BTF can be easily integrated into rendering engines by implementing it in Mitsuba 2.

### Method
We propose two changes. We replace the ReLUactivations with a Sine (Sin) as presented in Siren and use a different multi-level sampling like the one proposed in Instant-NGP, where we concatenate (Cat) results from different levels instead of reducing them through interpolation.
We perform an ablation study over the two proposed changes to evaluate their impact on memory, performance and quality.
We run the experiments on two datasets:
1. UBO2014, real captures.
2. MBTF from NeuMIP, synthetic.


### Results
![](/media/table.png)

+ Our results show that our method achieves better compression quality while using less memory and performance than the baseline.
+ Neural BTF is part of the broader field of neural scene representation.

### Discussion
+ When used in a simulation the material only behaves as a diffuse PDF.
+ We do not report performance results from a real-time implementation.