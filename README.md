## RAG KV Precomputation & Cross-Query Reuse (Reuse + Repair / Fusion)

- ⭐⭐⭐ (arXiv'2026) [**From Prefix Cache to Fusion RAG Cache: Accelerating LLM Inference in Retrieval-Augmented Generation**](https://arxiv.org/abs/2601.12904) (RAG, chunk-KV reuse, offline fusion, selective recomputation, cross-chunk context repair): 面向“复用 chunk-KV 会丢失跨 chunk 语境导致质量下降”的核心矛盾，提出 FusionRAG：离线阶段把相关 chunk 信息融合/注入到每个 chunk 的缓存表征里，在线阶段仅对模型关注的少量 token 重算，从而在较低重算比下显著提升质量-效率折中并降低 TTFT。

- ⭐⭐⭐ (arXiv'2025) [**$A^3$: Attention-Aware Accurate KV Cache Fusion for Fast Large Language Model Serving**](https://arxiv.org/abs/2511.17560) (KV cache fusion, attention-aware selection, selective recomputation, TTFT): 发现现有重算式复用常“重算了不该重算的 token”，导致更新与 query 相关内容错配；用 query→document 的注意力来选择/融合最关键的缓存段，并在极小重算预算下提升长上下文与 RAG 场景的质量与 TTFT。

- ⭐⭐⭐ (arXiv'2025) [**HyperRAG: Enhancing Quality-Efficiency Tradeoffs in Retrieval-Augmented Generation with Reranker KV-Cache Reuse**](https://arxiv.org/abs/2504.02921) (RAG pipeline, reranker KV reuse, document-side KV, I/O shift): 把 KV 复用落到 RAG pipeline 的 reranker 环节：缓存 document-side KV 让 reranker 主要处理 query 部分，并结合系统级优化把瓶颈从 GPU compute 转移到存储/I/O，在吞吐与端到端质量上取得更稳的收益。

- ⭐⭐⭐ (EuroSys'2025) [**CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion**](https://arxiv.org/abs/2405.16444) (RAG, cached KV fusion, selective recomputation, cross-chunk attention repair): 面向 RAG 输入由多个检索 chunk 组成且顺序/位置频繁变化导致 prefix caching 难命中的问题，允许直接复用离线预计算的 chunk-KV，并对少量 token 做选择性重算修复缺失的 cross-chunk attention，在低重算比例下逼近 full prefill 质量并提升吞吐与 TTFT。

- ⭐⭐⭐ (ICML'2025) [**EPIC: Efficient Position-Independent Context Caching for Serving Large Language Models**](https://proceedings.mlr.press/v267/hu25j.html) (position-independent caching, AttnLink/LegoLink, selective recomputation, KV reuse): 面向同一文档块在不同请求中出现但位置不同的 PIC 场景，通过 AttnLink/LegoLink 利用注意力稀疏性，只对极少量关键 token 做“链接式重算”补齐跨块依赖，并显式处理重复 attention sink，在显著降低 KV footprint 的同时保持推理质量。

- ⭐⭐⭐ (SIGMOD'2025) [**Cache-Craft: Managing Chunk-Caches for Efficient Retrieval-Augmented Generation**](https://arxiv.org/abs/2502.15734) (chunk-cache, reuse detection, partial recomputation, eviction policy): 面向生产 RAG 中 chunk 高频复用但非前缀对齐导致“可复用但直接拼接会掉点”的场景，先识别可复用 chunk-KV，再对少量被新 query/新上下文“污染”的 token 做部分重算修复质量，并配套缓存组织/淘汰策略让真实 workload 下收益可持续。

- (ICLR'2025) [**APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding**](https://openreview.net/forum?id=yUC8pU508S) (parallel encoding, KV precompute, distribution alignment, training-free): 面向“离线并行预计算多个 context 的 KV、在线组合使用”的范式，指出 naive parallel encoding 会因注意力分布失配而掉点；提出 shared prefix、adaptive temperature、scaling factor 等推理期对齐策略，使并行编码更接近 sequential encoding，常作为“复用+重算修复”类方法的重要基线。

- (arXiv'2025) [**CacheClip: Accelerating RAG with Effective KV Cache Reuse**](https://arxiv.org/abs/2510.10129) (auxiliary-model-guided selection, selective recomputation, inter-chunk attention, attention sinks): 面向 RAG 跨 chunk 推理中“直接复用 KV 会丢失 inter-chunk attention，且重复 attention sink 影响质量”的问题，用辅助小模型近似主模型注意力分布来更精准选 token 做选择性重算，并结合 shared prefixes 与分组更新提升局部一致性，在固定重算预算下更稳地恢复跨 chunk reasoning。

- (EMNLP'2025) [**TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text**](https://aclanthology.org/2025.emnlp-main.334/) (precomputed chunk-KV, mask/position redesign, TTFT reduction): 主打离线预计算并存储 chunk-KV，在线检索后直接加载 KV 以减少 prefill；为缓解 chunk 拼接带来的注意力/位置错配，提出 attention mask 与位置处理（并可结合轻量调优）尽量保持质量，更偏“拼接范式”而非大比例重算。

- (arXiv'2025) [**Parallel Key-Value Cache Fusion for Position Invariant RAG**](https://arxiv.org/abs/2501.07523) (position-invariant RAG, parallel KV fusion, multi-segment robustness): 面向 RAG 中输入段落可交换/顺序不稳定的问题，提出位置不变的 KV 融合，使多段落组合对顺序更鲁棒；通过并行融合降低多段输入的重复开销并缓解位置偏置导致的性能波动。

- (NeurIPS'2025) [**KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse**](https://neurips.cc/virtual/2025/poster/116061) (document KV precompute, position adjustment, special tokens, KV reuse): 面向多请求共享同一检索文档/背景材料导致重复编码的问题，将文档独立预计算 KV，在线按检索结果拼接复用，并通过 special tokens、注意力约束与位置调整缓解跨段依赖缺失，适合文档复用率高的 RAG 服务以减少重复 prefill。


## KV Cache Systems (Tiering / Sharing / Offloading / I/O-Aware Recomputation)

- ⭐⭐⭐ (arXiv'2025) [**KVShare: Semantic-Aware Key-Value Cache Sharing for Efficient Large Language Model Serving**](https://arxiv.org/abs/2503.16525) (multi-tenant KV reuse, semantic alignment, differential editing, serving system): 面向多租户服务中“严格前缀复用难命中、语义缓存又可能损伤多样性/一致性”的痛点，通过语义对齐与差分编辑实现更细粒度的 KV 共享，并配合服务系统设计在不显著伤害准确性的前提下提升复用收益。

- ⭐⭐⭐ (arXiv'2025) [**EVICPRESS: Joint KV-Cache Compression and Eviction for Efficient LLM Serving**](https://arxiv.org/abs/2512.14946) (multi-tier KV, joint compression+eviction, utility modeling, latency-quality tradeoff): 将“压缩”和“驱逐/多层放置”统一为联合决策：对压缩敏感上下文保守压缩、对可压缩部分激进压缩，并跨多存储层做自适应放置以降低平均生成时延，适合作为线上 KV 管理的系统骨架。

- (Findings ACL'2025) [**KVPR: Efficient LLM Inference with I/O-Aware KV Cache Partial Recomputation**](https://aclanthology.org/2025.findings-acl.997/) (KV offloading, partial recomputation, compute–communication overlap, scheduling): 面向 KV offload 到 CPU 后 PCIe 传输成为瓶颈的问题，用“部分重算换带宽”：先传一部分激活让 GPU 立即开始重算部分 KV，同时并行传输剩余 KV，实现通信与计算重叠，并结合切分/调度策略在端到端延迟与精度间做权衡。

- (arXiv'2025) [**Shared Disk KV Cache Management for Efficient Multi-Instance Inference in RAG-Powered LLMs**](https://arxiv.org/abs/2504.11765) (disk KV cache, multi-instance sharing, prefill offload, queue-aware caching): 面向多实例 RAG 服务，将文档相关 KV 预生成并放到共享磁盘 KV cache，多推理实例复用同一份 KV；结合查询局部性与排队延迟做主动生成/调度，偏工程落地与集群部署。

- (FAST'2025) [**Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot**](https://www.usenix.org/conference/fast25/presentation/qin) (KVCache-centric serving, multi-tier storage, precomputation): 以 KV cache 为中心重组推理服务，通过分层存储与预计算“以存储换计算”降低重复 prefill；对聊天服务与 RAG 文档块复用都具迁移价值。

- (FAST'2025) [**IMPRESS: An Importance-Informed Multi-Tier Prefix KV Storage System for Large Language Model Inference**](https://www.usenix.org/system/files/fast25-chen-weijian-impress.pdf) (multi-tier prefix-KV, hot/cold separation, prefetching): 用重要性评估将前缀 KV 分冷热并放到多层介质，结合预取减少 I/O；适合 prefix 命中率高的在线推理，也可增强“固定模板+检索段”式 RAG 的 KV 存储体系。

- (arXiv'2025) [**ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference**](https://arxiv.org/abs/2410.21465) (shadow cache, tiered KV, throughput-oriented): 用“主缓存+影子缓存”分层：主缓存保留高频 token，影子缓存保留关键低频 token，在吞吐与质量之间折中；适合长上下文与长文 RAG 的高并发吞吐优先场景。

- (ICML'2025) [**SpeCache: Speculative Key-Value Caching for Efficient Generation of LLMs**](https://proceedings.mlr.press/v267/jie25a.html) (CPU offload, speculative prefetch, top-k KV fetch, VRAM reduction): 将完整 KV cache 卸载到 CPU，GPU 仅保留摘要并按步动态取回 top-k 关键 KV；引入投机预测与预取并行降低传输额外时延，适合显存紧张但主机内存充足的长上下文/RAG。


## Multimodal KV Reuse & Position-Independent Caching (Optional but Practical)

- (arXiv'2025) [**MPIC: Position-Independent Multimodal Context Caching System for Efficient MLLM Serving**](https://arxiv.org/abs/2502.01960) (multimodal caching, position-independent reuse, reuse+recompute, MLLM serving): 面向多模态（文本-图像交错）与 multimodal RAG 中 prefix caching 更难命中的问题，把 PIC 扩展到多模态 KV：支持 KV 在本地/远端介质存储与并行加载，并集成 reuse+recompute 机制控制精度损失，代表“位置无关复用+重算修复”在 MLLM 的工程化路线。

- (arXiv'2025) [**MEPIC: Memory Efficient Position Independent Caching for LLM Serving**](https://arxiv.org/abs/2512.16822) (memory-efficient PIC, paged KV layout, block-level recomputation, RoPE fusion): 面向 PIC 在显存节省有限的痛点，通过 paged KV layout 提升跨请求共享度，并把重算从 token-level 提升到 block-level；同时融合 RoPE/内核级优化降低位置处理开销，扩大 PIC 在长提示与高复用服务中的收益。

- ⭐⭐⭐ (arXiv'2025) [**VLCache: Computing 2% Vision Tokens and Reusing 98% for Vision-Language Inference**](https://arxiv.org/abs/2512.12977) (VLM caching, encoder+KV reuse, non-prefix reuse error, layer-aware recomputation): 同时复用视觉 encoder cache 与 KV cache，并形式化分析非前缀复用误差的累积效应；通过按层动态分配的重算策略在极低计算比例下逼近 full recompute，并已集成到 SGLang，适合多模态 RAG/多次复用同图像输入的服务形态。


## Cache in RAG/Agent Pipelines: Semantic / Tool / Knowledge Caching

- (NeurIPS'2025) [**Generative Caching for Structurally Similar Prompts and Responses (GenCache)**](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/09/GenCache_NeurIPS25.pdf) (generative cache, structurally similar prompts, agent workflows, correctness): 面向 agent/workflow 中“结构相似但细节不同”的 prompts，GenCache 不直接返回旧 response（避免语义缓存忽略关键差异导致错误），而是从同簇 prompt-response 对中抽取“生成模式”，以可执行 program 形式缓存；命中时本地执行 program 生成差异感知的新响应，在提高命中率的同时尽量控制 negative hit，并降低端到端时延。


- (EuroMLSys'2024) [**RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation**](https://arxiv.org/abs/2404.12457) (RAG pipeline caching, multi-level caching, order-sensitive reuse, consistency): 从系统视角做 RAG 多级缓存（检索结果/文档块/中间表示等），重点解决文档顺序敏感导致难复用的问题；通过缓存组织与一致性策略提升命中率，并在端到端链路上优化延迟与成本。

- (NLPOSS'2023) [**GPTCache: An Open-Source Semantic Cache for LLM Applications Enabling Faster Answers and Cost Savings**](https://aclanthology.org/2023.nlposs-1.24.pdf) (semantic matching, embedding cache, modular design): 通过 embedding 语义相似度检索历史问答对以复用答案，降低重复调用导致的延迟与 token 成本，适合作为应用层语义缓存的工程基线。

- (VLDB'2025 Demo) [**ContextCache: Context-Aware Semantic Cache for Multi-Turn Queries in Large Language Models**](https://arxiv.org/abs/2506.22791) (context-aware keying, multi-turn dialogue, hit-rate optimization): 面向多轮对话构建上下文感知缓存键并语义匹配，缓解传统语义缓存忽略对话历史导致的低命中与不一致问题。

- (arXiv'2025) [**vCache: Verified Semantic Prompt Caching**](https://arxiv.org/abs/2502.03771v4) (verification, semantic equivalence, correctness): 针对语义匹配缓存容易出现 false positive 的风险，引入轻量验证机制确保命中在语义逻辑上严格等价，在保持低延迟优势的同时提升正确性与安全性。

- (NeurIPS'2025) [**SmartCache: Context-aware Semantic Cache for Efficient Multi-turn LLM Inference**](https://openreview.net/pdf/5bc13f5689dfb66b132abd36782eb71e1da88f36.pdf) (context-aware cache, sub-structure matching, dynamic eviction): 面向复杂多轮推理，利用更强的上下文/结构感知匹配与动态淘汰策略识别高价值语义片段并复用，降低长上下文推理 FLOPs 并提高效率。

- (arXiv'2025) [**Asteria: Semantic-Aware Cross-Region Caching for Agentic LLM Tool Access**](https://arxiv.org/abs/2509.17360) (distributed caching, tool-result reuse, cross-region latency): 面向 agent 工具调用的跨地域延迟与重复调用成本，通过语义感知分布式缓存复用昂贵的工具执行结果与中间态，降低跨地域通信延迟并减少工具重复调用。

- (arXiv'2024) [**MeanCache: User-Centric Semantic Caching for LLM Web Services**](https://arxiv.org/abs/2403.02694) (user-centric semantic cache, federated learning, privacy, query similarity): 提出用户侧本地语义缓存并用联邦学习训练相似度模型，在不集中存储用户数据的前提下提升相似查询匹配精度，适合面向终端/个性化服务的语义缓存形态。

- (arXiv'2024) [**GPT Semantic Cache: Reducing LLM Costs and Latency via Semantic Embedding Caching**](https://arxiv.org/abs/2411.05276) (semantic embedding cache, Redis, cost reduction): 用工程化的 embedding 语义缓存（如 Redis）检索相似问题并复用已有回答，强调降低重复 API 调用成本与响应延迟，适合作为轻量语义缓存 baseline。

- (arXiv'2026) [**Semantic Caching and Intent-Driven Context Optimization for Multi-Agent Natural Language to Code Systems**](https://arxiv.org/abs/2601.11687) (semantic caching, intent classifier, prompt assembly, multi-agent system): 将语义缓存与意图识别驱动的上下文裁剪/拼装结合，用结构化决策减少 prompt token 消耗并控制命中误差，适合作为“Agent 侧缓存 + Context 优化”类产品/系统参考。

- (arXiv'2025) [**Category-Aware Semantic Caching for Heterogeneous LLM Workloads**](https://arxiv.org/abs/2510.26835) (semantic caching, category-specific thresholds, TTL/quota, hybrid storage): 面向异构工作负载用“按类别差异化阈值/TTL/配额”的方式管理语义缓存命中与淘汰，并采用内存向量索引 + 外部存储的混合架构降低 miss 代价，适合线上语义缓存的工程化策略设计。

- (arXiv'2025) [**Semantic Caching for Low-Cost LLM Serving: From Offline Learning to Online Adaptation**](https://arxiv.org/abs/2508.07675) (semantic cache eviction, online learning, cost-aware policies, theory-to-system): 将语义缓存淘汰建模为带不确定分布的学习问题，提供离线最优化与在线自适应框架与算法保证，适合做“有理论支撑的 cache eviction/placement”方向。


## Cache-Augmented Generation (CAG) / Retrieval-Free Caching (Optional)

- (arXiv'2024) [**Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks**](https://arxiv.org/abs/2412.15605) (cache-augmented generation, preload knowledge, retrieval-free QA): 提出用长上下文把可控规模知识库“整包预加载”并缓存其运行时状态以绕开实时检索，作为 RAG 的反向对照范式；适合检索链路复杂/延迟敏感且知识库规模可控的业务场景。


## KV Cache Compression / Eviction / Quantization (Long-Context, RAG-Compatible)

- ⭐⭐⭐ (arXiv'2023) [**Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs**](https://arxiv.org/abs/2310.01801) (adaptive KV compression, head-wise policy, plug-and-play): 通过轻量 profiling 识别注意力头结构差异（局部头/特殊 token 头/全局头），并按 head 采用差异化 KV 构建与丢弃策略，以近乎零训练成本显著降低 KV 显存并尽量保持生成质量，是 head-wise/结构化驱逐路线的强基线。

- (NeurIPS'2023) [**H$_2$O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models**](https://arxiv.org/abs/2306.14048) (heavy-hitter eviction, dynamic cache, sparse attention): 基于累计注意力分数识别高频关键 token，只保留最重要的少量 KV，以较小精度损失换取显著显存下降与吞吐提升，适合长上下文/RAG 的 decode 压力缓解。

- ⭐⭐⭐ (ICLR'2024) [**Efficient Streaming Language Models with Attention Sinks (StreamingLLM)**](https://openreview.net/forum?id=NG7sS51zVF) (streaming inference, attention sinks, long-context stabilization): 发现 attention sink 现象并提出流式推理框架：滑窗下保留关键 sink token 的 KV，使超长流式输入更稳定，适合多轮对话与持续检索更新的 RAG。

- ⭐⭐⭐ (NeurIPS'2024) [**SnapKV: LLM Knows What You Are Looking for before Generation**](https://proceedings.neurips.cc/paper_files/paper/2024/file/28ab418242603e0f7323e54185d19bde-Paper-Conference.pdf) (pattern selection, token clustering, training-free compression): 利用模型注意力模式自动识别关键信息簇并仅保留关键片段 KV，在长文/RAG 中同时提升生成速度与显存效率，且无需微调即可应用。

- ⭐⭐⭐ (COLM'2025) [**PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling**](https://arxiv.org/abs/2406.02069) (layer-wise compression, pyramidal allocation, information density): 利用不同层的信息密度差异做金字塔式预算分配，浅层保留更多细节 KV、深层逐级减少 KV，在极高压缩比下维持更稳的长文理解与检索能力。

- ⭐⭐⭐ (NeurIPS'2025) [**Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference**](https://arxiv.org/abs/2407.11550) (adaptive budget, head-wise allocation, dynamic capacity): 动态为不同注意力头分配 KV 预算以适配“全局头/局部头”差异，在相同显存预算下获得更高精度与更稳的长程依赖。

- ⭐⭐⭐ (NeurIPS'2025) [**KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction**](https://openreview.net/forum?id=JFygzwx8SJ) (query-agnostic scoring, context reconstruction, multi-query reuse): 用“能否从压缩 KV 重建原始上下文”评估 token 重要性，实现与查询无关的驱逐，适合 RAG 中同一文档块被不同问题反复复用的多查询场景，减少 query-aware 评分抖动。

- (NeurIPS'2025) [**AttentionPredictor: Temporal Patterns Matter for KV Cache Compression**](https://arxiv.org/abs/2502.04077) (future relevance prediction, lightweight predictor, temporal awareness): 用轻量预测器预判 token 未来的重要性，缓解仅依赖历史注意力统计的滞后性，减少误删未来关键 token 的风险并提升压缩后生成连贯性。

- (ICLR'2026 Submission) [**RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression**](https://arxiv.org/abs/2502.14051) (two-stage compression, kernel optimization, end-to-end speedup): 结合两阶段压缩与底层系统/内核优化，不仅压缩 KV 大小，还直接优化访存与计算流水线，实现端到端推理延迟降低与吞吐提升。

- (ICLR'2026 Submission) [**SparseCache: Extreme Sparse Coding for KV Cache Compression**](https://openreview.net/forum?id=43zTdoRqY4) (dictionary learning, sparse coding, OMP reconstruction): 通过离线学习全局共享字典、在线稀疏编码与重构实现极端压缩，将 KV 映射到稀疏系数空间，适合 KV 成本极高的长上下文/RAG。

- (ICLR'2026 Submission) [**RACC: Retrieval-Augmented KV Cache Compression in Long-Context Generation**](https://openreview.net/forum?id=y2xi9ouYcg) (retrieval-aware importance, token ranking, selective retention): 引入检索相关性评估 token 重要性并做检索感知压缩，更偏向保留与证据强相关的 KV，适合“输入很长但关键证据稀疏”的 RAG 长文生成。

- (arXiv'2025) [**OjaKV: Context-Aware Online Low-Rank KV Cache Compression with Oja’s Rule**](https://arxiv.org/abs/2501.07137) (online low-rank, Oja’s rule, drift control, FlashAttention compatible): 用 Oja 规则在线更新低秩子空间做上下文感知的在线低秩压缩，并控制漂移，保持对高效注意力实现的兼容，适合流式/在线 RAG 推理。

- (ICML'2025) [**LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models**](https://proceedings.mlr.press/v267/shi25b.html) (ladder-shaped caching, cross-layer KV storage, distance-aware compression): 采用梯形缓存结构与距离相关的动态压缩策略，在持续生成时避免 OOM 并降低峰值显存，适合长上下文生成与长文 RAG。

- (ACL'2025) [**RefreshKV: Updating Small KV Cache During Long-form Generation**](https://aclanthology.org/2025.acl-long.1211.pdf) (periodic refresh, attention-pattern-driven update, long-form generation): 在长文生成中交替执行“全上下文注意力”和“小缓存注意力”，并根据全注意力模式周期性重建小 KV，在接近驱逐法速度收益下减少遗忘导致的质量劣化。

- (ICLR'2025) [**RazorAttention: Efficient KV Cache Compression Through Retrieval Heads**](https://openreview.net/forum?id=tkiZQlL04w) (retrieval heads, head-wise caching, compensation token, training-free): 发现少量 head 负责全局检索式注意力，其余多为局部注意力；据此对不同 head 采用差异化缓存，并用补偿 token 恢复信息，适合无需训练且希望与高效 attention kernel 兼容的压缩部署。

- (NeurIPS'2024) [**ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification**](https://openreview.net/forum?id=5t4ZAkPiJs) (KV quantization, salient token identification, FlashAttention-friendly): 在 KV 量化中结合显著 token 识别以提升高压缩比下的精度稳定性，并设计与快速注意力实现兼容的近似计算，适合量化优先的推理栈在不大改系统结构下压低 KV 占用与时延。

- (NeurIPS'2025) [**ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context Inference**](https://openreview.net/forum?id=20JDhbJqn3) (semantic chunking, compression, fragmentation mitigation): 将压缩基本单元从 token 提升到语义 chunk，减少逐 token 重要性评估造成的语义碎片，在高压缩率下更稳地保持跨句/跨段语义一致性，适合长文与 RAG 证据链场景。

- (NeurIPS'2025) [**MUSTAFAR: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference**](https://openreview.net/forum?id=C69741fMFX) (unstructured sparsity, bitmap sparse format, sparse attention kernel): 用非结构化稀疏直接剪枝 KV，并配套 bitmap 稀疏格式与自定义 attention kernel 在压缩态上计算，把“压缩收益”与“内核加速”绑定以抵消运行时开销。

- (NAACL'2025) [**A Systematic Study of Cross-Layer KV Sharing for Efficient LLM Inference**](https://aclanthology.org/2025.naacl-short.34.pdf) (cross-layer KV sharing, unified framework, configuration sweep): 系统统一不同跨层 KV sharing 方案并做配置扫描，给出在不同提示长度/压缩比下的吞吐与效果规律，适合工程选型与边界判断。


## KV Cache Quantization & Outlier/Sink Handling (Often Orthogonal but Very Useful)

- (arXiv'2024) [**KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization**](https://arxiv.org/abs/2401.18079) (KV quantization, sub-4-bit, extreme long context, robustness): 面向超长上下文下 KV 成为显存瓶颈的问题，探索低比特 KV 量化在亚 4-bit 场景的精度与稳定性挑战，为后续“极低比特 + outlier/sink 处理”路线提供基础参照。

- (ICML'2025) [**KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization**](https://arxiv.org/abs/2502.04420) (mixed-precision KV, layer-wise sensitivity, configuration search): 用层级敏感性分析与混合精度配置来稳定 KV 量化效果，并提供可搜索的配置空间，在长上下文推理中以较小精度损失换取吞吐提升，适合作为量化部署时的系统化选型参考。

- (arXiv'2025) [**Accurate KV Cache Quantization with Outlier Tokens Tracing**](https://arxiv.org/abs/2505.10938) (KV quantization, outlier tokens, 2-bit stability, throughput): 识别并追踪少量破坏量化假设的 outlier tokens，将其从量化路径中“特殊对待”，显著改善极低比特 KV 量化的精度稳定性并提升吞吐，适合与任何量化内核结合。

- (arXiv'2025) [**KVSink: Understanding and Enhancing the Preservation of Attention Sinks in KV Cache Quantization for LLMs**](https://arxiv.org/abs/2508.04257) (attention sinks, sink prediction, quantization interaction): 解释 attention sink 与极端激活 outlier 的关联，并指出 sink 不仅出现在前若干 token；提出低开销 sink 预测与更彻底的保留策略，提升 KV 量化下的困惑度与鲁棒性。

- (arXiv'2025) [**KVLinC: KV Cache Quantization with Hadamard Rotation and Linear Correction**](https://arxiv.org/abs/2510.05373) (2-bit KV, Hadamard rotation, linear correction, custom kernel): 在极低比特（如 2-bit）KV 量化下，用旋转降低量化误差并用轻量线性校正补偿注意力畸变，同时实现自定义 attention kernel 提速，适合把 KV footprint 压到极限的部署路线。

- (arXiv'2025) [**RotateKV: Accurate and Robust 2-Bit KV Cache Quantization for LLMs via Rotation**](https://arxiv.org/abs/2501.16383) (2-bit KV, rotation, robustness, long-context): 通过旋转等变换改善极低比特 KV 量化的数值稳定性与鲁棒性，面向“2-bit 级别压缩但不想掉点”的部署需求，可与 outlier/sink 处理互补。

- (arXiv'2024) [**QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs**](https://arxiv.org/abs/2404.00456) (rotation, outlier removal, 4-bit end-to-end, KV included): 从更广的“旋转去 outlier”视角做端到端低比特推理（含 KV），为 KV 量化的 outlier 机理与工程路径提供通用参考。

- (arXiv'2024) [**Model Tells You Where to Merge: Adaptive KV Cache Merging for LLMs on Long-Context Tasks**](https://arxiv.org/abs/2407.08454) (KV merging, token similarity, adaptive compression, long-context): 观察同序列中 key state 的相似性并做自适应 merging，在不显著牺牲质量下压缩 KV；与驱逐/量化正交，适合做组合策略（merge + quantize / merge + eviction）。


## Serving Systems / Prefill Optimization (Background but Often Used)

- (SOSP'2023) [**Efficient Memory Management for Large Language Model Serving with PagedAttention**](https://arxiv.org/abs/2309.06180) (paged KV layout, fragmentation mitigation, high throughput): 引入分页式 KV 管理以减少显存碎片并提升并发吞吐，是多种 KV 复用/压缩/外存方案的底座能力。

- (NeurIPS'2022) [**FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**](https://arxiv.org/abs/2205.14135) (IO-aware exact attention, tiling, memory-efficient kernel): 通过分块与 I/O 感知降低 HBM↔SRAM 数据搬运，在保持精确注意力下显著提速，是长上下文/RAG 推理的关键内核基础。

- (ICLR'2025) [**Block-Attention for Efficient Prefilling**](https://arxiv.org/abs/2409.15355) (prefill acceleration, dynamic block sparsity, TTFT reduction): 在 prefill 阶段利用块状稀疏跳过无关注意力块计算以降低 TTFT，适合 RAG 这种长输入 prefill 占比极高的负载。

- (NeurIPS'2024) [**SGLang: Efficient Execution of Structured Language Model Programs**](https://proceedings.neurips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf) (structured decoding, RadixAttention, automatic prefix KV reuse): 用 RadixAttention（基数树）管理 KV 并自动跨请求复用前缀，结合结构化执行优化提升吞吐并降低端到端延迟，适合 agent/程序化推理形态。

- (OSDI'2024) [**DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving**](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf) (prefill/decode disaggregation, KV pressure isolation, goodput optimization): 将 prefill 与 decode 解耦到不同资源/流水线以减少相互干扰并提升 goodput，适合 RAG 长输入 prefill 与短步 decode 资源诉求差异明显的场景。


## Surveys

- (TMLR'2025) [**A Survey on Large Language Model Acceleration based on KV Cache**](https://openreview.net/forum?id=z3JZzu9EA3) (taxonomy, KV management, serving optimization): 系统梳理 KV cache 相关加速方法并给出分类与对比，适合快速建立领域地图并定位“RAG+KV”优化在方法谱系中的位置。

- (arXiv'2025) [**Key, Value, Compress: A Systematic Exploration of KV Cache Compression Strategies**](https://arxiv.org/abs/2503.11816) (KV compression taxonomy, evaluation, latency-impact analysis): 对 KV 压缩方法做系统分类与实验评测，补齐“方法很多但选型缺乏统一视角”的痛点，适合做工程选型与 ablation 设计的参考框架。

- (arXiv'2026) [**Agentic Reasoning for Large Language Models**](https://arxiv.org/pdf/2601.12538) (Agentic Reasoning, foundational/self-evolving/collective dimensions, in-context/post-training optimization, LLM agents): 面向 LLM 在开放动态环境中缺乏交互、适应与协作能力的问题，提出 Agentic Reasoning 范式，从基础能力（规划、工具使用、搜索）、自进化（反馈、记忆、适应）、集体协作（多智能体角色分配与协调）三大维度构建体系，结合上下文内推理与训练后优化桥接思考与行动，适用于科学发现、机器人、医疗、网页探索等领域的智能体系统设计与优化。
