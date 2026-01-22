## RAG KV Precomputation & Cross-Query Reuse

- (EMNLP'2025) [**TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text**](https://aclanthology.org/2025.emnlp-main.334/) (precomputed KV cache, chunk-level reuse, TTFT reduction, position remapping): 通过**离线预计算并存储文档 chunk 的 KV cache**，在线检索时直接加载 KV，显著降低 RAG 的 prefill 计算与 TTFT；同时设计 attention mask / position 机制与轻量微调，使多 chunk 拼接后仍能保持可用的跨 chunk 推理质量。

- (EuroSys'2025) [**CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion**](https://github.com/YaoJiayi/CacheBlend) (cached knowledge fusion, multi-chunk KV stitching, 100% cache hit): 面向 RAG 中“检索到的 chunk 经常复用但顺序不同”的现实问题，提出**多段 KV cache 融合/拼接**方案，让命中缓存的 chunk 不必重新 prefill；核心在于处理多 chunk 的**位置偏移、跨段依赖**与融合策略，从系统角度减少重复算力并降低首 token 延迟。

- (arXiv'2025) [**CacheClip: Accelerating RAG with Effective KV Cache Reuse**](https://arxiv.org/abs/2510.10129) (selective recomputation, attention-guided token selection, inter-chunk attention recovery): 针对“直接复用离线 KV 会丢失跨 chunk 注意力而掉点”的矛盾，CacheClip 用**辅助模型预测主模型注意力分布**，只对少量关键 token 做选择性重算来恢复跨 chunk 注意力，同时复用其余 KV；兼顾 TTFT 与跨 chunk 推理质量，适合多文档/多跳问答的 RAG。

- (arXiv'2025) [**KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse**](https://arxiv.org/abs/2502.16002) (document KV precompute, KV concatenation, position adjustment, special tokens): 将“文档块”作为独立单元**预计算 KV**并持久化，在线将检索到的多个文档 KV **拼接复用**以避免重复 prefill；通过位置嵌入对齐、特殊 token 恢复跨块自注意力、以及混合数据微调，缓解独立编码导致的性能下降。

- (arXiv'2025) [**Shared Disk KV Cache Management for Efficient Multi-Instance Inference in RAG-Powered LLMs**](https://arxiv.org/abs/2504.11765) (disk KV cache, multi-instance sharing, prefill offload, queue-aware caching): 面向多实例 RAG 服务，将文档相关 KV 预生成并放到**共享磁盘 KV cache**，多个推理实例可复用同一份 KV；结合查询局部性与排队延迟做主动生成/调度，在资源约束下同时改善吞吐与延迟，偏工程落地与集群部署场景。

- (arXiv'2025) [**Parallel Key-Value Cache Fusion for Position Invariant RAG**](https://arxiv.org/abs/2501.07523) (position-invariant RAG, parallel KV fusion, multi-segment robustness): 面向 RAG 中输入段落可交换/顺序不稳定的问题，提出**位置不变**的 KV 融合思路，使多段落组合时对顺序更鲁棒；通过并行融合策略降低多段输入带来的重复开销并缓解位置偏置导致的性能波动。

- (arXiv'2024) [**RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation**](https://arxiv.org/abs/2404.12457) (multi-level caching, order-sensitive reuse, RAG pipeline optimization): 从系统视角做 RAG 多级缓存（检索结果/文档块/中间表示等），重点解决“文档顺序敏感导致难复用”的问题；通过缓存组织与一致性策略提升命中率，并在端到端链路上优化延迟与成本（更偏系统与工程）。

---

## RAG Decoding & Speculation

- (arXiv'2024) [**Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting**](https://arxiv.org/abs/2402.10410) (speculative decoding, draft-and-verify, RAG pipeline acceleration): 用小模型并行生成带检索增强的“草稿”输出，大模型进行验证/纠错，减少大模型在长输入上的解码负担；适合检索 chunk 很长、prefill+decode 都昂贵的 RAG 场景，通过“先快后准”提升整体吞吐。

- (arXiv'2025) [**REFRAG: Rethinking RAG Based Decoding**](https://arxiv.org/abs/2509.01092) (chunk embedding decoding, KV footprint reduction, first-token latency): 重新设计 RAG 解码，把长上下文压缩为更紧凑的表示进行解码，显著降低 KV cache 压力与首词延迟；核心是让解码器能“理解”压缩后的 chunk 表示，并在必要时局部恢复细节，从而在效率与质量之间做结构化权衡。

---

## Knowledge / Structure Injection via KV

- (ICLR'2026 Submission) [**AtlasKV: Augmenting LLMs with Billion-Scale Knowledge Graphs in 20GB VRAM**](https://openreview.net/forum?id=6i1jVAYbHs) (knowledge graph to KV, hierarchical KV pruning, VRAM-efficient knowledge augmentation): 将知识图谱三元组转成可被模型直接使用的 Q/K/V 风格数据（如 KG2KV），并用分层剪枝/管理（如 HiKVP）把十亿级 KG 以较小显存开销接入推理；适合“无需外部检索器但要强结构知识”的问答/推理场景。

- (NeurIPS'2025) [**Graph-KV: Breaking Sequence via Injecting Structural Biases into Large Language Models**](https://openreview.net/forum?id=J4w4RtwLyB) (graph-structured attention mask, structural inductive bias, RAG multi-hop reasoning): 把多段文本（例如检索到的 chunk）视为图节点，利用 KV cache 作为段落级表示，并用图结构约束注意力（block mask）实现类似 message passing 的交互；适合多跳推理与结构化文档（引用网络/段落依赖）下的 RAG，减少序列化带来的位置偏置与上下文浪费。

---

## KV Cache Compression / Eviction for Long Context (RAG-Compatible)

- (ICLR'2026 Submission) [**SparseCache: Extreme Sparse Coding for KV Cache Compression**](https://openreview.net/forum?id=43zTdoRqY4) (dictionary learning, sparse coding, OMP reconstruction, extreme compression): 用字典学习+稀疏编码压缩 KV：离线学习全局共享字典，在线用 OMP 得到稀疏系数并重构 KV；适合 KV 成本极高的长上下文/RAG 场景，思路是把 KV 表示映射到可高效存储的稀疏系数空间。

- (ICLR'2026 Submission) [**RACC: Retrieval-Augmented KV Cache Compression in Long-Context Generation**](https://openreview.net/forum?id=y2xi9ouYcg) (retrieval-aware importance, token ranking, selective retention): 引入“检索相关性”来评估 token 的重要性，对 KV 做检索感知压缩：更偏向保留与检索证据强相关的上下文；适用于 RAG 长文生成中“输入很长但真正关键证据很稀疏”的场景，可与其他系统优化正交叠加。

- (arXiv'2025) [**OjaKV: Context-Aware Online Low-Rank KV Cache Compression with Oja’s Rule**](https://arxiv.org/abs/2501.07137) (online low-rank, Oja’s rule, drift control, FlashAttention compatible): 通过 Oja 规则在线更新低秩子空间，对 KV 做**上下文感知的在线低秩压缩**；核心在于动态更新压缩基以避免上下文漂移，并保持对高效注意力实现的兼容性，适合流式/长对话/在线 RAG 推理。

- (ICML'2025) [**LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models**](https://proceedings.mlr.press/v267/shi25b.html) (ladder-shaped caching, cross-layer KV storage, distance-aware compression): 采用“梯形”KV 缓存结构与距离相关的动态压缩策略，在持续生成时避免 OOM；适合长上下文生成与长文 RAG，把缓存结构设计与压缩策略结合来降低峰值显存。

- (ICML'2025) [**SpeCache: Speculative Key-Value Caching for Efficient Generation of LLMs**](https://proceedings.mlr.press/v267/jie25a.html) (CPU offload, speculative prefetch, top-k KV fetch, VRAM reduction): 将完整 KV cache 卸载到 CPU 内存，GPU 只保留低精度摘要并按步**动态取回 top-k 关键 KV**；为避免 CPU↔GPU 传输带来额外时延，引入投机预测与预取并行，适合显存紧张但主机内存充足的长上下文/RAG 服务。

- (ACL'2025) [**RefreshKV: Updating Small KV Cache During Long-form Generation**](https://aclanthology.org/2025.acl-long.1211.pdf) (periodic refresh, attention-pattern-driven update, long-form generation): 在长文生成中交替执行“全上下文注意力”和“小缓存注意力”，并根据全注意力的模式周期性重建小 KV；适合长篇写作/长文 RAG，目标是在接近驱逐法的速度收益下减少遗忘带来的质量劣化。

- (ICLR'2025) [**RazorAttention: Efficient KV Cache Compression Through Retrieval Heads**](https://openreview.net/forum?id=tkiZQlL04w) (retrieval heads, head-wise caching, compensation token, training-free): 发现少量 attention head 负责全局检索式注意力，其余多为局部注意力；据此对不同 head 采用差异化缓存：关键 head 保留全量 KV，非关键 head 丢弃远端 token，并用补偿 token 恢复信息；适合无需训练、希望与高效注意力内核兼容的压缩部署。

- (NeurIPS'2024) [**ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification**](https://openreview.net/forum?id=5t4ZAkPiJs) (KV quantization, salient token identification, FlashAttention-friendly): 在 KV 量化中结合显著 token 识别以提升高压缩比下的精度稳定性，并设计与快速注意力实现兼容的近似计算；适合“量化优先”的推理栈，在不大改系统结构的情况下压低 KV 占用与时延。

- (NeurIPS'2025) [**ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context Inference**](https://openreview.net/forum?id=20JDhbJqn3) (chunk-based compression, semantic grouping, context fragmentation mitigation): 将压缩基本单元从 token 提升到“语义 chunk”，减少逐 token 重要性评估造成的语义碎片；适合长文与 RAG 场景中需要跨句/跨段语义一致性的任务，在高压缩率下更稳。

- (NeurIPS'2025) [**MUSTAFAR: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference**](https://openreview.net/forum?id=C69741fMFX) (unstructured sparsity, bitmap sparse format, sparse attention kernel): 用非结构化稀疏直接剪枝 KV，并配套 bitmap 稀疏格式与自定义 attention kernel 在压缩态上计算；适合 decode 阶段显存/带宽受限的服务，把“压缩收益”与“内核加速”绑定以抵消运行时开销。

- (NeurIPS'2022025) [**KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction**](https://openreview.net/forum?id=JFygzwx8SJ) (query-agnostic eviction, context reconstruction scoring, multi-query reuse): 针对多查询复用的场景，KVzip 用“能否从压缩 KV 重建原始上下文”来评估 KV 重要性，从而实现**与查询无关**的驱逐；适合 RAG 中同一文档块会被不同问题复用的情况，避免 query-aware 策略在多查询下不稳定。

- (ICLR'2024) [**Efficient Streaming Language Models with Attention Sinks (StreamingLLM)**](https://openreview.net/forum?id=NG7sS51zVF) (streaming inference, attention sinks, long context stabilization): 发现“attention sink”现象并提出流式推理框架：即便采用滑窗也保留关键 sink token 的 KV，使模型在超长流式输入下更稳定；适合多轮对话与持续检索更新的 RAG（流式上下文不断增长），降低重算与漂移风险。

- (NAACL'2025) [**A Systematic Study of Cross-Layer KV Sharing for Efficient LLM Inference**](https://aclanthology.org/2025.naacl-short.34.pdf) (cross-layer KV sharing, unified framework, configuration sweep): 系统梳理并统一不同跨层 KV sharing 方案（哪些层共享、Q 与哪层 KV 配对等），给出在不同提示长度/压缩比下的吞吐与效果规律；适合做工程选型与配置扫描，为“跨层共享是否值得”提供经验边界。

---

## KV Cache Management (Submissions & Emerging Directions)

- (ICLR'2026 Submission) [**ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs**](https://openreview.net/forum?id=TcymFvT03t) (zero-shot compression, importance allocation, residual merge): 面向无需训练/迁移成本的部署，按多维重要性动态分配缓存预算，并通过残差合并保留关键上下文；适合频繁更换模型或任务的 RAG 服务，强调“即插即用”的压缩策略。

- (ICLR'2026 Submission) [**Hierarchical Adaptive Eviction (HAE)**](https://openreview.net/forum?id=RlH8muWuiY) (hierarchical eviction, prefill pruning, decode eviction, multimodal KV): 在预填充阶段做双注意力剪枝、解码阶段动态驱逐，形成分层驱逐框架；适合多模态/长上下文推理（例如图文检索增强），在控制精度损失的前提下降低 KV 峰值占用。

- (ICLR'2026 Submission) [**HOLD ONTO THAT THOUGHT: Assessing KV Cache Compression on Reasoning**](https://openreview.net/forum?id=vE8dQvDh2l) (reasoning sensitivity, compression trade-off, evaluation suite): 系统评估 KV 压缩对推理类任务的影响，强调不同压缩策略会以不同方式伤害链式推理/多步依赖；适合需要“保推理不掉点”的 RAG（尤其多跳 QA），为选择压缩策略提供推理敏感性视角。

- (ICLR'2026 Submission) [**ORACLEKV: Oracle Guidance for Question-Independent KV Cache Eviction**](https://openreview.net/forum?id=pRO3R2MUka) (question-independent eviction, offline importance learning, online eviction): 通过离线学习 token 重要性（类似 oracle 指导），在线执行与问题无关的动态驱逐；适合多查询复用与开放域问答式 RAG，减少 query-conditioned 评分在实际系统中的抖动与额外算力。

- (ICLR'2026 Submission) [**CAKE: Cascading and Adaptive KV Cache Eviction for Efficient LLM Inference**](https://openreview.net/forum?id=WFLxYozcZn) (cascaded eviction, adaptive budget, latency-aware): 用级联与自适应策略在不同阶段/层次逐步淘汰不重要 KV，并根据预算/时延压力动态调参；适合在线服务的“压力自适应”场景，在吞吐与质量间更平滑地调节。

- (ICLR'2026 Submission) [**Taming the Fragility of KV Cache Eviction (DefensiveKV)**](https://openreview.net/forum?id=zZPksxLgdV) (eviction robustness, adversarial/fragile cases, defensive policies): 针对驱逐策略在特定输入模式下容易突然崩溃的问题，提出更鲁棒的驱逐设计；适合生产 RAG 中“分布外检索文本/噪声 chunk”较多的情况，目标是避免质量断崖式下降。

- (ICLR'2026 Submission) [**A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints**](https://openreview.net/forum?id=AWLJJRgvbA) (queueing theory, stability region, KV memory constraints, capacity planning): 把计算资源与 KV cache 内存约束统一进排队论模型，推导稳定性条件并指导资源规划；适合做大规模 RAG 部署容量评估（GPU 显存是硬瓶颈），为调度/限流/缓存策略提供理论化工具。

- (ICLR'2026 Submission) [**Lexico: Extreme KV Cache Compression via Sparse Coding over Universal Dictionaries**](https://openreview.net/forum?id=p5rLOe9USf) (universal dictionary, sparse coding, extreme compression): 以通用字典+稀疏编码为核心，把 KV 映射到稀疏系数表示以实现极端压缩；适合显存极度紧张或需要把 KV 持久化到更廉价介质的 RAG 服务，与 “SparseCache” 属于相近但可互补的方向。

---

## LLM Serving Systems (KV/Prefix/Cache-Centric)

- (FAST'2025) [**Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot**](https://www.usenix.org/conference/fast25/presentation/qin) (KVCache-centric serving, multi-tier storage, precomputation): 以 KV cache 为中心重新组织推理服务：更多使用存储换取更少计算，通过分层存储与预计算策略降低重复 prefill；适合多轮对话/高复用前缀的聊天服务，也可迁移到 RAG 中复用频繁的文档块场景。

- (FAST'2025) [**IMPRESS: An Importance-Informed Multi-Tier Prefix KV Storage System for Large Language Model Inference**](https://www.usenix.org/system/files/fast25-chen-weijian-impress.pdf) (multi-tier prefix KV, hot/cold separation, prefetching): 用重要性评估把前缀 KV 分成冷热并放到多层介质，结合预取减少 I/O；适合 prefix 命中率高的生产推理（包括 RAG 中“固定模板+检索段”模式），核心是把 KV 当作可管理的数据对象来做存储系统优化。

- (arXiv'2025) [**ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference**](https://arxiv.org/abs/2410.21465) (shadow cache, tiered KV, throughput-oriented): 用“主缓存+影子缓存”机制：主缓存保留高频 token，影子缓存保留关键低频 token，在吞吐与质量之间做分层折中；适合长上下文与长文 RAG 的高并发服务，强调吞吐优先但避免关键证据被驱逐。

- (OSDI'2024) [**DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving**](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf) (prefill/decode disaggregation, KV memory pressure isolation, goodput optimization): 将 prefill 与 decode 解耦到不同资源/流水线，减少相互干扰并提高整体 goodput；适合 RAG 中 prefill 占比极高的负载，把“长输入 prefill”与“短步 decode”分离后更利于做 KV/显存与算力的独立优化。

- (OSDI'2024) [**Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve**](https://www.usenix.org/conference/osdi24/presentation/agrawal) (scheduler, throughput-latency tradeoff, batch/interleave control): 通过更精细的调度与批处理/交织策略，缓解吞吐与延迟的矛盾；适合 RAG 服务在高峰期做动态调度，间接影响 KV cache 峰值占用与请求排队抖动。

- (OSDI'2024) [**ServerlessLLM: Low-Latency Serverless Inference for Large Language Models**](https://www.usenix.org/conference/osdi24/presentation/fu) (serverless inference, cold-start mitigation, elastic serving): 面向弹性伸缩与多租户，降低 serverless 推理的冷启动与调度开销；对 RAG 来说适合“潮汐流量+多模型/多版本”场景，KV/前缀缓存更依赖外部化与跨实例共享策略。

- (USENIX ATC'2025) [**KVCache Cache in the Wild: Characterizing and Optimizing KVCache Cache at a Large Cloud Provider**](https://www.usenix.org/conference/atc25/presentation/wang-jiahao) (workload characterization, KV reuse patterns, eviction policy design): 基于真实生产 trace 系统刻画 KV cache 复用模式（复用倾斜、类别内可预测等），并据此设计更匹配负载的驱逐策略；适合做生产系统的策略落地与容量规划，尤其对 RAG 这种复用强但分布复杂的业务很有参考价值。

- (arXiv'2025) [**METIS: Fast Quality-Aware RAG Systems with Configuration Adaptation**](https://arxiv.org/abs/2412.10543) (quality-aware scheduling, configuration adaptation, chunk count tuning): 在 RAG 端到端链路上联合做调度与配置自适应（例如每个请求检索 chunk 数、合成策略），在质量不降的情况下显著降低延迟；它不是直接的 KV 算法，但通过减少无效 chunk 与优化合成流程，能间接降低 KV 压力与 prefill 成本。

---

## Hardware / Disaggregated Acceleration for RAG

- (VLDB'2025) [**Chameleon: A Heterogeneous and Disaggregated Accelerator System for Retrieval-Augmented Language Models**](https://www.vldb.org/pvldb/vol18/p42-jiang.pdf) (heterogeneous accelerators, disaggregation, retrieval+LLM co-design): 将检索与 LLM 推理分别映射到更匹配的硬件（如 FPGA/专用检索加速 + GPU 推理），并通过解耦式架构独立扩展两类资源；适合大规模 RAG 服务，把检索侧吞吐/能效与推理侧 KV/显存瓶颈分开优化。

- (ISCA'2025) [**REIS: A High-Performance and Energy-Efficient Retrieval System with In-Storage Processing**](https://arxiv.org/abs/2506.16444) (in-storage processing, ANNS acceleration, data movement reduction): 在存储侧做近数据处理以降低向量检索的数据搬运成本；对 RAG 来说能降低检索端延迟并释放 host 资源，让更多预算留给 LLM 推理（包括 KV cache 管理与更大上下文窗口）。

---

## Surveys

- (OpenReview'2025) [**A Survey on Large Language Model Acceleration based on KV Cache**](https://openreview.net/forum?id=z3JZzu9EA3) (taxonomy, KV management, serving optimization): 对 KV cache 相关的加速方法做系统分类与对比（压缩/驱逐/共享/系统实现等），适合用来快速建立领域地图，并据此定位“RAG+KV”优化在方法谱系中的位置。
