## RAG KV Precomputation & Cross-Query Reuse

- (EuroSys'2025) [**CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion**](https://arxiv.org/abs/2405.16444) (RAG, cached KV fusion, selective recomputation, cross-chunk attention repair): 面向 RAG 输入由多个检索 chunk 组成、且 chunk 顺序/位置频繁变化导致 prefix caching 命中率低的问题，允许直接复用离线预计算的 chunk-KV，并对少量 token 进行选择性重算来恢复缺失的 cross-chunk attention，从而在低重算比例下逼近 full prefill 的质量；同时通过流水化与重叠 I/O 隐藏 KV 加载开销，提升端到端吞吐与 TTFT。

- (ICML'2025) [**EPIC: Efficient Position-Independent Context Caching for Serving Large Language Models**](https://proceedings.mlr.press/v267/hu25j.html) (position-independent caching, AttnLink/LegoLink, selective recomputation, KV reuse): 面向同一文档块在不同请求中出现但位置不同的 PIC 场景，通过 AttnLink/LegoLink 利用注意力稀疏性，只对极少量关键 token 做“链接式重算”以补齐跨块依赖，并显式处理重复 attention sink；在显著降低 KV footprint 的同时保持生成/推理质量，适合 RAG/agent 中高复用上下文的服务形态。

- (ICLR'2025) [**APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding**](https://openreview.net/forum?id=yUC8pU508S) (parallel encoding, KV precompute, distribution alignment, training-free): 面向 RAG/ICL 的“离线并行预计算多个 context 的 KV、在线组合使用”范式，指出 naive parallel encoding 会因注意力分布失配而明显掉点；提出 shared prefix、adaptive temperature、scaling factor 等推理期对齐策略，使并行编码更接近 sequential encoding，从而减少重复 prefill 计算并保持质量，是后续“复用+重算修复”类方法的重要基线/对照。

- (SIGMOD'2025) [**Cache-Craft: Managing Chunk-Caches for Efficient Retrieval-Augmented Generation**](https://arxiv.org/abs/2502.15734) (chunk-cache, reuse detection, partial recomputation, eviction policy): 面向生产 RAG 中 chunk 高频复用但非前缀对齐导致“可复用但直接拼接会掉点”的场景，提出 chunk-cache：先识别可复用 chunk-KV，再对少量被新 query/新上下文“污染”的 token 做部分重算修复质量，避免全量重算；并配套缓存组织、淘汰策略与 overhead masking，使复用收益在真实 workload 下稳定落地。

- (Findings ACL'2025) [**KVPR: Efficient LLM Inference with I/O-Aware KV Cache Partial Recomputation**](https://aclanthology.org/2025.findings-acl.997/) (KV offloading, partial recomputation, compute–communication overlap, scheduling): 面向 KV offload 到 CPU 后 PCIe 传输成为瓶颈、GPU 等待造成空转的问题，通过“部分重算换带宽”：先传一部分激活让 GPU 立即开始重算部分 KV，同时并行传输剩余 KV，实现通信与计算重叠；结合调度/切分策略在端到端延迟与精度之间做系统化权衡。

- (arXiv'2025) [**CacheClip: Accelerating RAG with Effective KV Cache Reuse**](https://arxiv.org/abs/2510.10129) (auxiliary-model-guided selection, selective recomputation, inter-chunk attention): 面向 RAG 跨 chunk 推理中“直接复用 KV 会丢失 inter-chunk attention，且重复 attention sink 影响质量”的问题，使用辅助小模型近似主模型注意力分布来更精准地选 token 做选择性重算，并结合 shared prefixes 与分组更新策略提升局部一致性；强调在固定 recompute budget 下更稳地恢复 cross-chunk reasoning 质量。

- (EMNLP'2025) [**TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text**](https://aclanthology.org/2025.emnlp-main.334/) (precomputed KV cache, chunk-level reuse, mask/position redesign, TTFT reduction): 主打离线预计算并存储 chunk-KV，在线检索后直接加载 KV 以显著减少 prefill；为缓解 chunk 拼接带来的注意力/位置错配，提出相应的 attention mask 与位置处理（并可结合轻量调优）来尽量保持质量，更偏“拼接范式”，选择性重算通常占比更小。

- (NeurIPS'2025) [**KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse**](https://neurips.cc/virtual/2025/poster/116061) (document KV precompute, position adjustment, special tokens, KV reuse): 面向多请求共享同一检索文档/背景材料导致重复编码的问题，将文档独立预计算 KV，在线按检索结果拼接复用，并通过 special tokens、注意力约束与位置调整缓解跨段依赖缺失；适合文档复用率高的 RAG 服务，以减少重复 prefill 为主要收益来源。

- (arXiv'2025) [**MPIC: Position-Independent Multimodal Context Caching System for Efficient MLLM Serving**](https://arxiv.org/abs/2502.01960) (multimodal caching, position-independent reuse, reuse+recompute, MLLM serving): 面向多模态（文本-图像交错）与 multimodal RAG 中 prefix caching 更难命中的问题，把 PIC 思路扩展到多模态 KV：支持 KV 在本地/远端介质存储与并行加载，并在系统内集成 reuse+recompute 机制控制精度损失，属于把“位置无关复用 + 选择性重算”工程化落地到 MLLM 的代表方向。

- (arXiv'2025) [**MEPIC: Memory Efficient Position Independent Caching for LLM Serving**](https://arxiv.org/abs/2512.16822) (memory-efficient PIC, paged KV layout, block-level recomputation, RoPE fusion): 面向 PIC 在 HBM 节省有限的痛点，通过 paged KV layout 提升跨请求共享度，并把重算从 token-level 提升到 block-level，使请求相关部分更集中；同时融合 RoPE/内核级优化降低位置处理开销，目标是在保持 PIC 可用性的同时进一步扩大显存收益，适合长提示与高复用 RAG/agent 服务。

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

## Semantic-Aware RAG Granularity & Segmentation (RAG semantic analysis, chunking/blocking, structure-aligned compression)

- (arXiv'2025) [**SABlock: Semantic-Aware KV Cache Eviction with Adaptive Compression Block Size**](https://arxiv.org/abs/2510.22556) (semantic segmentation, adaptive block sizing, structure-aligned eviction, long-context): 面向长上下文/RAG 中“按 token 或固定 block 驱逐会切碎语义结构、导致关键证据断裂式丢失”的问题，先做语义分段使压缩边界贴合语言结构，再在段内用 segment-guided scoring 估计重要性，最后对每段用 budget-driven search 自适应选择 block size，在“语义完整性 vs 压缩效率”之间动态折中，重点减少语义边界被硬切导致的不可逆损失。
- (arXiv'2025) [**ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference**](https://arxiv.org/abs/2502.00299) (semantic chunking, chunk-level selection, context fragmentation mitigation, layer-wise index reuse): 面向 token-level 重要性评估容易造成“只保留零散关键词、丢掉句子级语义约束”的问题，把压缩决策单位提升为语义 chunk：以 chunk 为单位保留/丢弃来避免语义碎片；并利用多层保留索引相似性做 layer-wise index reuse 降低维护开销，适合长文 QA/RAG 中证据跨句跨段分布且需要语义连贯的任务。
- (arXiv'2025) [**SentenceKV: Efficient LLM Inference via Sentence-Level Semantic KV Caching**](https://arxiv.org/abs/2504.00970) (sentence-level caching, semantic coherence, semantic indexing, CPU offload): 面向“语义保持方法往往 TTFT 高、token-level 压缩又破坏句内语义”的矛盾，把 KV 管理提升到句子级：prefill 阶段将 token 聚合成句子级语义向量并保留在 GPU，同时把细粒度 KV offload；decode 阶段按 query 与句子向量的语义相似性选择性取回句子相关 KV，减少无关 KV 加载并保持句子语义完整性，适合证据通常以句子为自洽单元的 RAG。
- (arXiv'2024) [**ClusterKV: Manipulating LLM KV Cache in Semantic Space for Recallable Compression**](https://arxiv.org/abs/2412.03213) (semantic clustering, recallable compression, cluster-level recall, long-context QA): 面向“驱逐后不可召回导致后期生成突然需要旧证据而崩”的问题，把召回粒度从位置页提升到语义簇：在语义空间对 KV/Token 聚类并以 cluster 为单位做选择、索引与缓存；当后续步骤需要旧信息时按语义簇召回，从而在小 cache budget 下更稳地覆盖“证据晚出现/晚引用”的长链路 RAG 生成与推理场景。

---

## Dynamic Cache Maintenance (eviction/refresh/query-agnostic reuse/streaming stabilization)

- (arXiv'2024) [**RefreshKV: Updating Small KV Cache During Long-form Generation**](https://arxiv.org/abs/2411.05787) (periodic refresh, full-vs-subset attention alternation, attention-pattern-driven update, long-form): 面向“纯驱逐一旦丢了就回不来、长文生成后期质量塌陷”的问题，交替执行全上下文注意力与小缓存注意力；在若干步后用一次全注意力观测到的 attention pattern 重建/更新小 KV，实现“保持小 cache 但不永久遗忘”，适用于长文 RAG（报告/综述/长回答）这种后期仍可能回引用早期证据的场景。
- (arXiv'2025) [**KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction**](https://arxiv.org/abs/2505.23416) (query-agnostic eviction, context reconstruction scoring, multi-query reuse, robust caching): 面向生产 RAG 中“同一上下文会被不同 query 反复复用”的形态，提出 query-agnostic 的重要性度量：用模型从压缩 KV 中重建原始上下文的能力给 KV 打分并驱逐低分项，使压缩后的缓存对未来未知 query 更稳，避免 query-aware 策略在 multi-query 下抖动/掉点。
- (arXiv'2023) [**Efficient Streaming Language Models with Attention Sinks (StreamingLLM)**](https://arxiv.org/abs/2309.17453) (streaming stabilization, attention sinks, sliding-window compatibility, recomputation avoidance): 面向流式对话/持续 RAG（上下文不断增长）中滑窗在超过 cache size 后失稳的问题，利用 attention sink 现象：即便采用滑窗也固定保留少量 sink tokens 的 KV 来稳定注意力分布，从而减少重算并避免性能断崖，适合“不断追加检索证据”的 streaming RAG。
- (arXiv'2023) [**H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models**](https://arxiv.org/abs/2306.14048) (heavy-hitter eviction, recent+heavy hitter retention, dynamic submodular optimization, decode efficiency): 面向 decode 阶段 KV 带宽/显存瓶颈，基于“少量 heavy hitter tokens 贡献大部分注意力价值”的观察，动态维持 recent tokens 与 heavy hitters 的平衡；并把驱逐建模为动态优化问题给出理论性质，是动态驱逐路线中常见的基础对照基线。
- (arXiv'2024) [**SnapKV: LLM Knows What You are Looking for Before Generation**](https://arxiv.org/abs/2404.14469) (head-wise selection, clustered KV positions, observation window, plug-and-play): 面向无需训练、可直接插入推理栈的驱逐需求，利用 prompt 尾部 observation window 估计各 head 的注意力模式，做 head-wise 的 clustered important positions 选择以压缩 KV；强调以较低在线开销获得稳定的速度/显存收益，适合模板较固定但检索证据变化大的在线 RAG。
- (arXiv'2025) [**CAKE: Cascading and Adaptive KV Cache Eviction with Layer Preferences**](https://arxiv.org/abs/2503.12491) (layer preferences, adaptive budgeting, cascading eviction, temporal importance shift): 面向“不同层对长程依赖需求不同、统一预算会浪费或伤关键层”的问题，估计 layer preferences（空间/时间维注意力动态）并做层间自适应预算分配，再用级联式管理满足全局内存预算；同时引入考虑重要性随时间漂移的指标，降低驱逐对后续步骤的连锁伤害，适合长上下文 RAG 的稳健推理链。
- (arXiv'2025) [**LLMs Know What to Drop: Self-Attention Guided KV Cache Eviction for Efficient Long-Context Inference (SAGE-KV)**](https://arxiv.org/abs/2503.08879) (self-attention-guided eviction, one-shot top-k after prefill, head+token selection, low overhead): 面向超长上下文推理，利用 prefill 后可观测到的注意力稀疏性，一次性在 token 与 head 两个维度做 top-k 选择压缩 KV，并在整个 decode 期间复用该压缩结果，避免每步动态选择的开销；适合对吞吐/时延敏感的 RAG 在线服务。
- (arXiv'2024) [**Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference**](https://arxiv.org/abs/2407.11550) (head-wise adaptive budget allocation, plug-and-play, theoretical bound, eviction quality): 面向“不同 head 信息密度差异大但多数方法均匀分配预算导致利用率低”的问题，给出 attention 输出误差的理论上界并据此优化 head-wise 预算分配；作为可插拔模块可叠加到多种驱逐策略上，在相同总预算下提升驱逐后的质量稳定性，适合 RAG 中 head 分工明显（模板/证据/指令）的模型。


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
