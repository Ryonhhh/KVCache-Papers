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

## KV Cache Compression / Eviction for Long Context (RAG-Compatible)

- (NeurIPS'2023) [**H$_2$O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models**](https://arxiv.org/abs/2306.14048)(heavy-hitter eviction, sparse attention, dynamic cache): 针对 LLM 推理中 KV Cache 显存占用过大的问题；发现注意力矩阵中存在 "Heavy Hitter"（高频关键 Token）现象，提出基于累积注意力分数的动态驱逐策略，只保留最重要的少量 KV；在保持生成质量的同时，大幅降低了显存占用并提升了推理吞吐量。

- ⭐⭐(ICLR'2024) [**Efficient Streaming Language Models with Attention Sinks (StreamingLLM)**](https://openreview.net/forum?id=NG7sS51zVF) (streaming inference, attention sinks, long context stabilization): 发现“attention sink”现象并提出流式推理框架：即便采用滑窗也保留关键 sink token 的 KV，使模型在超长流式输入下更稳定；适合多轮对话与持续检索更新的 RAG（流式上下文不断增长），降低重算与漂移风险。

- ⭐⭐⭐(NeurIPS'2024) [**SnapKV: LLM Knows What You Are Looking for before Generation**](https://proceedings.neurips.cc/paper_files/paper/2024/file/28ab418242603e0f7323e54185d19bde-Paper-Conference.pdf)(automatic pattern selection, compression-friendly, long-context retrieval): 针对长上下文 RAG 中大量无关上下文导致检索精度下降及显存浪费的问题；利用模型对 Prompt 的注意力模式自动识别关键信息簇（Clustering），像“快照”一样只保留关键片段对应的 KV；显著提升了长文处理的生成速度与显存效率，且不需要微调即可应用于各种长窗口模型。

- ⭐⭐(COLM'2025) [**PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling**](https://arxiv.org/abs/2406.02069)(layer-wise compression, pyramidal allocation, information density): 针对传统 KV 压缩忽略了不同层级信息密度差异（深层需要更少上下文）的问题；提出金字塔式的压缩策略，浅层保留更多细节 KV，深层逐级减少 KV 预算；在极高压缩比下维持了优于均匀压缩的长文理解能力与Passkey 检索精度。

- ⭐⭐(NeurIPS'2025) [**Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference**](https://arxiv.org/abs/2407.11550)(adaptive budget, head-wise allocation, dynamic capacity): 针对不同注意力头（Attention Head）对上下文依赖程度不同（有的头关注全局，有的关注局部）的问题；通过分析 Query 向量的范数等特征，动态为每个头分配不同的 KV 缓存预算；实现了更精细化的显存管理，在相同显存预算下获得了更高的模型精度。

- (NeruIPS'2025) [**AttentionPredictor: Temporal Patterns Matter for KV Cache Compression**](https://arxiv.org/abs/2502.04077)(temporal awareness, future relevance prediction, lightweight predictor): 针对现有基于历史注意力分数驱逐 KV 可能误删未来重要 Token 的滞后性问题；引入一个轻量级预测器来预判 Token 在未来的注意力权重（Temporal Patterns），而非仅依赖历史统计；减少了误删关键信息的风险，提升了压缩后的生成连贯性与长程依赖能力。

- (ICLR'2026 Submission) [**RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression**](https://arxiv.org/abs/2502.14051)(two-stage compression, system optimization, end-to-end speedup): 针对现有 KV 压缩算法仅关注显存减少而忽视实际系统加速的问题；结合 SnapKV 的两阶段策略（先粗筛后精选）与底层系统优化（如自定义 Kernel），不仅压缩 KV 大小，更直接优化了访存与计算流水线；实现了真正的端到端推理延迟降低与吞吐量提升。

- (NAACL'2025) [**A Systematic Study of Cross-Layer KV Sharing for Efficient LLM Inference**](https://aclanthology.org/2025.naacl-short.34.pdf) (cross-layer KV sharing, unified framework, configuration sweep): 系统梳理并统一不同跨层 KV sharing 方案（哪些层共享、Q 与哪层 KV 配对等），给出在不同提示长度/压缩比下的吞吐与效果规律；适合做工程选型与配置扫描，为“跨层共享是否值得”提供经验边界。

- ⭐⭐⭐(NeurIPS'2025) [**KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction**](https://openreview.net/forum?id=JFygzwx8SJ) (query-agnostic eviction, context reconstruction scoring, multi-query reuse): 针对多查询复用的场景，KVzip 用“能否从压缩 KV 重建原始上下文”来评估 KV 重要性，从而实现**与查询无关**的驱逐；适合 RAG 中同一文档块会被不同问题复用的情况，避免 query-aware 策略在多查询下不稳定。

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

## LLM Serving Systems (KV/Prefix/Cache-Centric) / System Optimization

- (SOSP'2023) [**Efficient Memory Management for Large Language Model Serving with PagedAttention**](https://arxiv.org/abs/2309.06180)(virtual memory, memory fragmentation, high throughput): 针对 LLM 服务中 KV cache 显存碎片化（内/外碎片）严重导致并发度受限的问题；引入操作系统虚拟内存中“分页”的思想，允许 KV cache 在物理显存中非连续存储；消除了显存碎片，极大提升了显存利用率与系统的推理吞吐量（Throughput）。

- (NeurIPS'2022) [**FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**](https://arxiv.org/abs/2205.14135)(IO-aware, tiling, linear memory): 针对 Transformer 注意力层由于频繁读写 HBM（高带宽内存）导致的 IO 瓶颈问题；通过 Tiling（分块）和重计算策略，减少 GPU HBM 与片上 SRAM 之间的数据搬运次数；在保持注意力计算精确性的同时，将显存复杂度降为线性，显著提升了训练/推理速度。

- (ICLR'2025) [**Block-Attention for Efficient Prefilling**](https://arxiv.org/abs/2409.15355) (long-context prefill, dynamic sparsity, latency reduction): 针对长上下文输入在 Prefill 阶段计算量呈二次方增长（$O(N^2)$）导致首字延迟过高的问题；利用注意力矩阵的块状稀疏特征，在 Prefill 阶段动态识别并跳过无关的注意力块计算；在保证精度的前提下显著降低了长文输入的首字延迟（TTFT）与计算功耗。

- (NeurIPS'2024) [**SGLang: Efficient Execution of Structured Language Model Programs**](https://proceedings.neurips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf) (RadixAttention, automatic KV reuse, structured decoding): 针对复杂 LLM 编程模式（如 Agent、Chain-of-Thought）中 KV cache 复用困难及结构化输出效率低的问题；提出 RadixAttention（基于基数树的 KV 管理）自动跨请求复用前缀 KV，并结合解释器优化；大幅提升了多轮交互与复杂任务的服务吞吐量并降低了端到端延迟。

- (Arxiv'2025) [**LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference**](https://lmcache.ai/tech_report.pdf)(RadixAttention, automatic KV reuse, structured decoding): 针对复杂 LLM 编程模式（如 Agent、Chain-of-Thought）中 KV cache 复用困难及结构化输出效率低的问题；提出 RadixAttention（基于基数树的 KV 管理）自动跨请求复用前缀 KV，并结合解释器优化；大幅提升了多轮交互与复杂任务的服务吞吐量并降低了端到端延迟。

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

- (TMLR'2025) [**A Survey on Large Language Model Acceleration based on KV Cache**](https://openreview.net/forum?id=z3JZzu9EA3) (taxonomy, KV management, serving optimization): 对 KV cache 相关的加速方法做系统分类与对比（压缩/驱逐/共享/系统实现等），适合用来快速建立领域地图，并据此定位“RAG+KV”优化在方法谱系中的位置。

## Semantic Cache / RAG Cache

- (NLPOSS'2023) [**GPTCache: An Open-Source Semantic Cache for LLM Applications Enabling Faster Answers and Cost Savings**](https://aclanthology.org/2023.nlposs-1.24.pdf) (semantic matching, modular design, cost reduction): 针对 LLM 应用中重复或相似查询导致的高昂 API 调用成本与响应延迟问题；通过向量嵌入（Embedding）计算语义相似度来检索历史问答对，替代传统的精确匹配缓存；显著降低了端到端延迟并节省了 Token 费用。

- (VLDB'2025 demo paper) [**ContextCache: Context-Aware Semantic Cache for Multi-Turn Queries in Large Language Models**](https://arxiv.org/abs/2506.22791) (multi-turn dialogue, context management, hit rate optimization): 针对传统语义缓存忽略多轮对话上下文、导致在连续交互中命中率低的问题；引入上下文感知机制，结合历史对话信息构建缓存键值并进行语义匹配；优化了多轮会话场景下的缓存命中率与回答一致性。

- (Arxiv'2025) [**vCache: Verified Semantic Prompt Caching**](https://arxiv.org/abs/2502.03771v4) (correctness verification, exact semantic equivalence, safety): 针对近似语义匹配可能导致缓存返回错误或不相关答案（False Positives）的准确性风险；引入轻量级的验证机制来确保检索到的缓存 Prompt 与当前输入在语义逻辑上的严格等价性；在保持低延迟优势的同时，大幅提升了缓存响应的准确性（Correctness）与安全性。

- (EuroMLSys'2024) [**RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation**](https://arxiv.org/abs/2404.12457) (knowledge retrieval caching, tiered storage, replacement policy): 针对 RAG 系统中检索与文档处理（Prefill）阶段的高计算开销与重复访问特性；设计了一种感知 RAG 知识结构的缓存系统，在 GPU/CPU 主存间分层缓存热门文档的中间状态（如 KV）；显著缩短了 RAG 的首字延迟（TTFT）并提升了系统吞吐量。

- (NeurIPS'2025) [**SmartCache: Context-aware Semantic Cache for Efficient Multi-turn LLM Inference**](https://openreview.net/pdf/5bc13f5689dfb66b132abd36782eb71e1da88f36.pdf) (sub-graph matching, dynamic eviction, long-context efficiency): 针对复杂多轮推理中静态缓存策略难以识别高价值上下文片段的问题；提出基于子图匹配或动态策略的上下文感知缓存，智能识别并复用关键的语义片段；降低了长上下文推理的计算开销（FLOPs）并提高了推理效率。

- (Arxiv'2025) [**Asteria: Semantic-Aware Cross-Region Caching for Agentic LLM Tool Access**](https://arxiv.org/abs/2509.17360) (distributed caching, agent tool use, cross-region latency): 针对 Agent 系统在跨地域调用工具（Tool Access）时的高延迟与数据传输瓶颈；构建语义感知的跨区域分布式缓存系统，专门复用昂贵的工具执行结果与推理中间态；有效降低了跨地域通信延迟并减少了工具重复调用成本。