## Chunk/Document KV Precomputation & Position-Independent Reuse (Reuse + Repair)

- ⭐⭐⭐(EuroSys'2025) [**CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion**](https://arxiv.org/abs/2405.16444) (RAG, chunk KV reuse, cached knowledge fusion, selective recomputation): 面向 RAG 多 chunk 且 chunk 顺序/位置频繁变化导致 prefix caching 命中率低的问题，复用离线预计算的 chunk-KV，并对少量 token 选择性重算以修复缺失的 cross-chunk attention；通过流水化与 I/O 重叠隐藏 KV 加载开销，提升吞吐与 TTFT。

- ⭐⭐⭐(ICML'2025) [**EPIC: Efficient Position-Independent Context Caching for Serving Large Language Models**](https://proceedings.mlr.press/v267/hu25j.html) (position-independent caching, AttnLink/LegoLink, selective recomputation): 面向 PIC（同块不同位置复用），利用注意力稀疏性只对极少关键 token 做“链接式重算”补齐跨块依赖，并显式处理重复 attention sink；在降低 KV footprint 的同时保持质量，适合 RAG/agent 的高复用上下文服务。

- (ICLR'2025) [**APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding**](https://openreview.net/forum?id=yUC8pU508S) (parallel encoding, KV precompute, distribution alignment, training-free): 提出“离线并行预计算多个 context 的 KV、在线组合使用”的范式，并指出 naive 并行编码会因注意力分布失配掉点；给出 shared prefix、adaptive temperature、scaling factor 等推理期对齐策略，作为后续“复用+重算修复”方法的重要基线。

- ⭐⭐⭐(SIGMOD'2025) [**Cache-Craft: Managing Chunk-Caches for Efficient Retrieval-Augmented Generation**](https://arxiv.org/abs/2502.15734) (chunk-cache, reuse detection, partial recomputation, eviction): 面向生产 RAG 中 chunk 高频复用但非前缀对齐的问题，先识别可复用 chunk-KV，再对少量被新 query/上下文“污染”的 token 做部分重算修复质量，并配套缓存组织与淘汰策略，让收益在真实 workload 下稳定落地。

- (arXiv'2025) [**CacheClip: Accelerating RAG with Effective KV Cache Reuse**](https://arxiv.org/abs/2510.10129) (auxiliary-model guidance, selective recomputation, inter-chunk attention): 针对直接复用 KV 丢失 inter-chunk attention、重复 attention sink 影响质量的问题，用小模型近似主模型注意力分布以更精准选 token 做重算，并结合 shared prefixes/分组更新提升局部一致性，强调固定重算预算下更稳地恢复跨 chunk 推理。

- (EMNLP'2025) [**TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text**](https://aclanthology.org/2025.emnlp-main.334/) (precomputed chunk KV, mask/position redesign, TTFT reduction): 主打离线预计算 chunk-KV、在线检索后直接加载以减少 prefill；通过 attention mask 与位置处理缓解拼接错配（可结合轻量调优），更偏“拼接范式”，选择性重算占比通常更小。

- (NeurIPS'2025) [**KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse**](https://neurips.cc/virtual/2025/poster/116061) (document KV precompute, position adjustment, special tokens): 将共享文档独立预计算 KV，在线按检索结果拼接复用，并通过 special tokens、注意力约束与位置调整缓解跨段依赖缺失；适合文档复用率高的 RAG 服务，主要收益是减少重复 prefill。

- (arXiv'2025) [**Parallel Key-Value Cache Fusion for Position Invariant RAG**](https://arxiv.org/abs/2501.07523) (position-invariant RAG, parallel KV fusion, multi-segment robustness): 面向检索段落可交换/顺序不稳定的问题，提出位置不变的 KV 融合思路，提升对段落排列的鲁棒性；用并行融合降低多段输入重复开销并缓解位置偏置导致的性能波动。

- (arXiv'2025) [**MPIC: Position-Independent Multimodal Context Caching System for Efficient MLLM Serving**](https://arxiv.org/abs/2502.01960) (multimodal PIC, reuse+recompute, MLLM serving): 将 PIC 扩展到多模态（图文交错）KV，支持本地/远端介质存储与并行加载，并在系统内集成 reuse+recompute 控精度损失，是“位置无关复用+选择性重算”在 MLLM 上的工程化代表。

- (arXiv'2025) [**MEPIC: Memory Efficient Position Independent Caching for LLM Serving**](https://arxiv.org/abs/2512.16822) (memory-efficient PIC, paged KV layout, block recomputation, RoPE fusion): 针对 PIC 显存节省有限，通过 paged KV layout 提升共享度，把重算从 token-level 提升到 block-level，并融合 RoPE/内核级优化降低位置处理开销，目标是在保持 PIC 可用性的同时进一步扩大显存收益。


## KV Storage, Offloading, and Multi-Instance Sharing (I/O-Aware)

- (Findings ACL'2025) [**KVPR: Efficient LLM Inference with I/O-Aware KV Cache Partial Recomputation**](https://aclanthology.org/2025.findings-acl.997/) (KV offloading, partial recomputation, compute–communication overlap): 面向 KV offload 到 CPU 后 PCIe 传输成瓶颈的问题，用“部分重算换带宽”：先传部分激活让 GPU 立即开工重算，同时并行传输剩余 KV，实现通信与计算重叠，并通过调度/切分策略权衡端到端延迟与精度。

- (arXiv'2025) [**Shared Disk KV Cache Management for Efficient Multi-Instance Inference in RAG-Powered LLMs**](https://arxiv.org/abs/2504.11765) (shared disk KV cache, multi-instance sharing, queue-aware caching): 将文档相关 KV 预生成并放入共享磁盘，多推理实例复用同一份 KV；结合查询局部性与排队延迟做主动生成/调度，偏集群部署与工程落地。

- (FAST'2025) [**Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot**](https://www.usenix.org/conference/fast25/presentation/qin) (KVCache-centric serving, multi-tier storage, precomputation): 以 KV cache 为中心重组推理服务，用分层存储与预计算“以存储换计算”，降低重复 prefill；对聊天与 RAG 文档块复用都具有迁移价值。

- (FAST'2025) [**IMPRESS: An Importance-Informed Multi-Tier Prefix KV Storage System for Large Language Model Inference**](https://www.usenix.org/system/files/fast25-chen-weijian-impress.pdf) (multi-tier prefix KV, hot/cold separation, prefetching): 通过重要性评估把前缀 KV 分冷热放到多层介质并结合预取减少 I/O；适合 prefix 命中率高的在线服务，也可作为 RAG “模板+检索段”模式的存储侧增强。

- (arXiv'2025) [**ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference**](https://arxiv.org/abs/2410.21465) (tiered KV, shadow cache, throughput-oriented): 用主缓存+影子缓存分层：主缓存保留高频 token，影子缓存保留关键低频 token，在吞吐与质量之间折中；适合长上下文/长文 RAG 的高并发吞吐优先场景。

- (ICML'2025) [**SpeCache: Speculative Key-Value Caching for Efficient Generation of LLMs**](https://proceedings.mlr.press/v267/jie25a.html) (CPU offload, speculative prefetch, top-k KV fetch): 将完整 KV 卸载到 CPU，GPU 仅保留摘要并按步动态取回 top-k 关键 KV；用投机预测与预取并行降低 CPU↔GPU 传输额外时延，适合显存紧张但主机内存充足的长上下文/RAG。


## RAG Decoding & Speculation

- (arXiv'2024) [**Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting**](https://arxiv.org/abs/2402.10410) (speculative decoding, draft-and-verify, RAG acceleration): 用小模型并行生成带检索增强的草稿，大模型验证/纠错，减少大模型在长输入上的解码负担；适合 chunk 很长、prefill+decode 都昂贵的 RAG。

- (arXiv'2025) [**REFRAG: Rethinking RAG Based Decoding**](https://arxiv.org/abs/2509.01092) (compressed decoding, KV footprint reduction, TTFT): 重新设计 RAG 解码，把长上下文压缩为更紧凑表示进行解码，降低 KV 压力与首词延迟；必要时局部恢复细节，在效率与质量间做结构化权衡。


## Knowledge / Structure Injection via KV

- (ICLR'2026 Submission) [**AtlasKV: Augmenting LLMs with Billion-Scale Knowledge Graphs in 20GB VRAM**](https://openreview.net/forum?id=6i1jVAYbHs) (KG to KV, hierarchical KV pruning, VRAM-efficient augmentation): 将知识图谱三元组转为可直接被注意力使用的 KV 风格数据，并用分层剪枝/管理把十亿级 KG 以较小显存开销接入推理；适合无需外部检索器但要强结构知识的问答/推理。

- (NeurIPS'2025) [**Graph-KV: Breaking Sequence via Injecting Structural Biases into Large Language Models**](https://openreview.net/forum?id=J4w4RtwLyB) (graph-structured attention mask, segment KV, multi-hop RAG): 将检索 chunk 视为图节点，利用 KV cache 作为段落级表示，并用图结构约束注意力实现类似 message passing 的交互；适合多跳推理与结构化文档下的 RAG，减少位置偏置与上下文浪费。


## Long-Context KV Reduction (Compression / Eviction / Quantization) — RAG-Compatible

- (NeurIPS'2023) [**H$_2$O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models**](https://arxiv.org/abs/2306.14048) (heavy-hitter eviction, dynamic cache, sparse attention): 基于累计注意力分数识别高频关键 token，只保留最重要的少量 KV，降低显存占用并提升吞吐，适合长上下文/RAG 的 decode 压力缓解。

- ⭐⭐⭐(ICLR'2024) [**Efficient Streaming Language Models with Attention Sinks (StreamingLLM)**](https://openreview.net/forum?id=NG7sS51zVF) (streaming inference, attention sinks, long-context stabilization): 发现 attention sink 并提出流式推理：滑窗下保留关键 sink token 的 KV，使超长流式输入更稳定；适合多轮对话与持续检索更新的 RAG。

- ⭐⭐⭐(NeurIPS'2024) [**SnapKV: LLM Knows What You Are Looking for before Generation**](https://proceedings.neurips.cc/paper_files/paper/2024/file/28ab418242603e0f7323e54185d19bde-Paper-Conference.pdf) (pattern selection, token clustering, training-free compression): 通过注意力模式自动识别关键信息簇，仅保留关键片段 KV；在长文/RAG 中同时提升速度与显存效率，且无需微调。

- ⭐⭐⭐(COLM'2025) [**PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling**](https://arxiv.org/abs/2406.02069) (layer-wise compression, pyramidal allocation): 利用层级信息密度差异，浅层保留更多细节 KV、深层逐级减少预算，在高压缩比下维持更稳的长文理解与检索能力。

- ⭐⭐⭐(NeurIPS'2025) [**Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference**](https://arxiv.org/abs/2407.11550) (adaptive budget, head-wise allocation): 动态为不同注意力头分配不同 KV 预算，针对“有的头全局、有的头局部”的差异做精细化显存管理，在相同预算下提高精度。

- (NeurIPS'2025) [**AttentionPredictor: Temporal Patterns Matter for KV Cache Compression**](https://arxiv.org/abs/2502.04077) (future relevance prediction, lightweight predictor): 用轻量预测器预判 token 未来注意力权重，缓解仅依赖历史统计导致的“误删未来关键 token”，提升压缩后生成连贯性与长程依赖。

- (ICLR'2026 Submission) [**RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression**](https://arxiv.org/abs/2502.14051) (two-stage compression, kernel optimization, end-to-end speedup): 结合两阶段压缩与系统内核优化，不仅压缩 KV 大小，还直接优化访存与计算流水线，实现端到端延迟降低与吞吐提升，适合工程化落地。

- ⭐⭐⭐(NeurIPS'2025) [**KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction**](https://openreview.net/forum?id=JFygzwx8SJ) (query-agnostic scoring, reconstruction-based importance, multi-query reuse): 用“能否从压缩 KV 重建原始上下文”来评估重要性，实现与查询无关的驱逐；适合同文档块被不同问题反复复用的 RAG，避免 query-aware 策略抖动。

- (ICLR'2026 Submission) [**SparseCache: Extreme Sparse Coding for KV Cache Compression**](https://openreview.net/forum?id=43zTdoRqY4) (dictionary learning, sparse coding, OMP reconstruction): 离线学习全局字典，在线用稀疏系数表示并重构 KV，实现极端压缩；适合 KV 成本极高的长上下文/RAG。

- (ICLR'2026 Submission) [**RACC: Retrieval-Augmented KV Cache Compression in Long-Context Generation**](https://openreview.net/forum?id=y2xi9ouYcg) (retrieval-aware importance, token ranking): 引入检索相关性评估 token 重要性，更偏向保留与证据强相关的 KV；适用于“输入很长但证据很稀疏”的 RAG 长文生成。

- (arXiv'2025) [**OjaKV: Context-Aware Online Low-Rank KV Cache Compression with Oja’s Rule**](https://arxiv.org/abs/2501.07137) (online low-rank, drift control, FlashAttention compatible): 用 Oja 规则在线更新低秩子空间，做上下文感知的在线低秩压缩并控制漂移，同时保持对高效注意力实现的兼容，适合流式/在线 RAG。

- (ICML'2025) [**LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models**](https://proceedings.mlr.press/v267/shi25b.html) (ladder-shaped caching, distance-aware compression): 采用梯形缓存结构与距离相关压缩策略，降低持续生成时的峰值显存风险；适合长上下文生成与长文 RAG。

- (ACL'2025) [**RefreshKV: Updating Small KV Cache During Long-form Generation**](https://aclanthology.org/2025.acl-long.1211.pdf) (periodic refresh, attention-pattern-driven update): 交替执行全上下文注意力与小缓存注意力，并按模式周期性刷新小 KV，在接近驱逐法速度收益下减少遗忘导致的质量劣化。

- (ICLR'2025) [**RazorAttention: Efficient KV Cache Compression Through Retrieval Heads**](https://openreview.net/forum?id=tkiZQlL04w) (retrieval heads, head-wise caching, compensation token): 发现少量 head 负责全局检索式注意力，据此对 head 采用差异化缓存，并用补偿 token 恢复信息；适合无需训练且希望兼容高效 attention kernel 的压缩部署。

- (NeurIPS'2024) [**ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification**](https://openreview.net/forum?id=5t4ZAkPiJs) (KV quantization, salient token identification): 量化 KV 时结合显著 token 识别以提高高压缩比下精度稳定性，并设计与快速注意力实现兼容的近似计算，适合“量化优先”的推理栈。

- (NeurIPS'2025) [**ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context Inference**](https://openreview.net/forum?id=20JDhbJqn3) (semantic chunking, compression, fragmentation mitigation): 将压缩单元从 token 提升到语义 chunk，减少逐 token 压缩带来的语义碎片，适合长文与 RAG 的语义一致性需求。

- (NeurIPS'2025) [**MUSTAFAR: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference**](https://openreview.net/forum?id=C69741fMFX) (unstructured sparsity, bitmap format, sparse attention kernel): 直接剪枝 KV 并用 bitmap 稀疏格式与自定义 attention kernel 在压缩态计算，把“压缩收益”和“内核加速”绑定以抵消运行时开销。

- (NAACL'2025) [**A Systematic Study of Cross-Layer KV Sharing for Efficient LLM Inference**](https://aclanthology.org/2025.naacl-short.34.pdf) (cross-layer KV sharing, unified framework, configuration sweep): 系统统一不同跨层 KV sharing 方案并做配置扫描，给出在不同提示长度/压缩比下的吞吐与效果规律，适合工程选型与边界判断。


## LLM Serving Systems / Prefill Optimization / Scheduling

- (SOSP'2023) [**Efficient Memory Management for Large Language Model Serving with PagedAttention**](https://arxiv.org/abs/2309.06180) (paged KV, fragmentation mitigation, high throughput): 将 KV cache 以分页方式存储，缓解显存碎片并提升并发与吞吐，是多种 KV 管理/复用方法的底座级能力。

- (NeurIPS'2022) [**FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**](https://arxiv.org/abs/2205.14135) (IO-aware exact attention, tiling, memory-efficient): 通过分块与 I/O 感知减少 HBM↔SRAM 搬运，在保持精确注意力下显著提速，是长上下文/RAG 推理内核的重要基础。

- (ICLR'2025) [**Block-Attention for Efficient Prefilling**](https://arxiv.org/abs/2409.15355) (prefill sparsity, block pruning, TTFT reduction): 在 prefill 阶段动态识别并跳过无关注意力块计算，降低长输入 TTFT 与功耗，适合 prefill 占比极高的 RAG。

- (NeurIPS'2024) [**SGLang: Efficient Execution of Structured Language Model Programs**](https://proceedings.neurips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf) (structured decoding, RadixAttention, automatic KV reuse): 面向 agent/结构化程序执行，用 RadixAttention 管理 KV 并自动跨请求复用前缀，同时优化结构化解码执行，提升吞吐并降低端到端延迟。

- (Tech Report'2025) [**LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference**](https://lmcache.ai/tech_report.pdf) (KV cache layer, reuse infrastructure, serving integration): 更偏工程系统层的 KV cache 基础设施，强调在企业级推理中把 KV 当作可共享可管理的数据层来做复用与治理，适合作为生产系统的缓存“平台层”。

- (OSDI'2024) [**DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving**](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf) (prefill/decode disaggregation, memory pressure isolation): 将 prefill 与 decode 解耦到不同资源/流水线，减少相互干扰并提高 goodput；对 RAG 这种 prefill 超重负载尤其有效，利于独立优化 KV/显存与算力。

- (OSDI'2024) [**Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve**](https://www.usenix.org/conference/osdi24/presentation/agrawal) (scheduler, interleaving, throughput–latency control): 用更精细的调度与批处理/交织策略缓解吞吐与延迟矛盾，适合 RAG 高峰期动态调度并间接影响 KV 峰值与排队抖动。

- (OSDI'2024) [**ServerlessLLM: Low-Latency Serverless Inference for Large Language Models**](https://www.usenix.org/conference/osdi24/presentation/fu) (serverless inference, cold-start mitigation, elastic serving): 面向弹性伸缩与多租户降低冷启动与调度开销；对潮汐流量 RAG 更依赖外部化 KV/跨实例共享策略。

- (USENIX ATC'2025) [**KVCache Cache in the Wild: Characterizing and Optimizing KVCache Cache at a Large Cloud Provider**](https://www.usenix.org/conference/atc25/presentation/wang-jiahao) (workload characterization, reuse patterns, eviction design): 基于真实生产 trace 刻画 KV 复用模式并据此优化策略，适合做生产系统容量规划与驱逐/复用策略落地参考。

- (arXiv'2025) [**METIS: Fast Quality-Aware RAG Systems with Configuration Adaptation**](https://arxiv.org/abs/2412.10543) (quality-aware scheduling, chunk count tuning, end-to-end optimization): 通过端到端调度与配置自适应（如检索 chunk 数、合成策略）在质量不降前提下降低延迟；虽非直接 KV 算法，但可显著减少无效 chunk 带来的 prefill/KV 压力。


## Semantic / Prompt Cache (Application-Layer Reuse)

- (NLPOSS'2023) [**GPTCache: An Open-Source Semantic Cache for LLM Applications Enabling Faster Answers and Cost Savings**](https://aclanthology.org/2023.nlposs-1.24.pdf) (semantic matching, embedding cache, cost reduction): 通过 embedding 语义相似度检索历史问答对，替代重复调用，降低延迟与 token 成本；更偏应用层 prompt→answer 缓存。

- (VLDB'2025 Demo) [**ContextCache: Context-Aware Semantic Cache for Multi-Turn Queries in Large Language Models**](https://arxiv.org/abs/2506.22791) (multi-turn cache keying, context management): 引入上下文感知机制构建缓存键并语义匹配，提升多轮会话连续交互中的命中率与回答一致性。

- (arXiv'2025) [**vCache: Verified Semantic Prompt Caching**](https://arxiv.org/abs/2502.03771v4) (semantic equivalence verification, correctness): 在语义缓存中加入轻量验证，降低 false positive 风险，提升缓存返回的正确性与安全性，适合对“错答代价高”的 RAG/agent。

- (EuroMLSys'2024) [**RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation**](https://arxiv.org/abs/2404.12457) (RAG pipeline caching, tiered storage, replacement policy): 从系统视角做 RAG 多级缓存（检索结果/文档块/中间表示/KV 等），解决文档顺序敏感导致难复用的问题；通过缓存组织与一致性/替换策略提升命中率并优化端到端延迟与成本。

- (NeurIPS'2025) [**SmartCache: Context-aware Semantic Cache for Efficient Multi-turn LLM Inference**](https://openreview.net/pdf/5bc13f5689dfb66b132abd36782eb71e1da88f36.pdf) (context-aware cache, dynamic eviction, long-context efficiency): 面向复杂多轮推理，利用上下文感知/动态策略识别高价值语义片段并复用，降低长上下文推理开销并提升效率。

- (arXiv'2025) [**Asteria: Semantic-Aware Cross-Region Caching for Agentic LLM Tool Access**](https://arxiv.org/abs/2509.17360) (distributed caching, tool-result reuse, cross-region latency): 面向跨地域 agent 工具调用，用语义感知分布式缓存复用昂贵的工具执行结果与中间态，降低跨地域通信延迟与重复调用成本。


## Hardware / Disaggregated Acceleration for RAG

- (VLDB'2025) [**Chameleon: A Heterogeneous and Disaggregated Accelerator System for Retrieval-Augmented Language Models**](https://www.vldb.org/pvldb/vol18/p42-jiang.pdf) (heterogeneous accelerators, disaggregation, retrieval+LLM co-design): 将检索与 LLM 推理映射到更匹配硬件并解耦扩展两类资源，适合大规模 RAG，把检索吞吐/能效与推理侧 KV/显存瓶颈分开优化。

- (ISCA'2025) [**REIS: A High-Performance and Energy-Efficient Retrieval System with In-Storage Processing**](https://arxiv.org/abs/2506.16444) (in-storage processing, ANNS acceleration, data movement reduction): 在存储侧做近数据处理降低向量检索数据搬运成本，降低检索端延迟并释放 host 资源，让更多预算留给 LLM 推理（含 KV 管理）。


## Surveys

- (TMLR'2025) [**A Survey on Large Language Model Acceleration based on KV Cache**](https://openreview.net/forum?id=z3JZzu9EA3) (taxonomy, KV management, serving optimization): 系统梳理 KV cache 相关加速方法（压缩/驱逐/共享/系统实现等），适合快速建立领域地图并定位“RAG+KV”优化在方法谱系中的位置。
