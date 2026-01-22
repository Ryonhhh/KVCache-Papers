(ICLR'2025) SparseCache: Extreme Sparse Coding for KV Cache Compression：基于字典学习和稀疏编码的 KV 缓存压缩框架，离线学习全局共享字典，在线 OMP 稀疏编码与重构，实现极高压缩比；
https://openreview.net/forum?id=43zTdoRqY4 (KV 缓存压缩、RAG 预计算 KV 优化、存储效率提升)
(ICLR'2025) AtlasKV: Augmenting LLMs with Billion-Scale Knowledge Graphs in 20GB VRAM：提出 KG2KV 与 HiKVP，将知识图谱转化为 KV 嵌入并分层管理，无需外部检索器集成十亿级知识图谱；
暂未公开 (KV 与知识图谱融合、RAG 知识嵌入、显存优化)
(ICLR'2025) Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting：小模型并行生成检索增强草稿，大模型验证，缓解长输入推理速度慢与理解质量下降问题；
暂未公开 (LLM 推理加速、RAG 流程优化、投机解码适配)
(ICLR'2025) RACC: Retrieval-Augmented KV Cache Compression in Long-Context Generation：检索感知的 KV 缓存压缩，利用检索相关性排序 token 重要性，仅保留 15% 缓存维持无损准确性；
暂未公开 (RAG 长文本优化、检索感知压缩、多方法正交应用)
(ICLR'2025) OjaKV: Context-Aware Online Low-Rank KV Cache Compression with Oja’s Rule：基于 Oja 规则的上下文感知在线低秩 KV 压缩，动态调整压缩率，避免上下文漂移，兼容 FlashAttention；
暂未公开 (在线 KV 压缩、低秩分解、实时推理适配)
(ICML'2025) LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models：梯形 KV 缓存模式，跨层存储 KV 对，结合距离动态压缩，实现长上下文无 OOM 持续生成；
暂未公开 (可参考 ICML 2025 官网：https://icml.cc/Conferences/2025) (KV 缓存结构创新、长文本 RAG 推理优化)
(ICML'2025) REFRAG: Rethinking RAG Based Decoding：重新设计 RAG 解码流程，将 KV 缓存需求降至传统方法的 3%，首词延迟大幅降低，保持低困惑度；
https://arxiv.org/abs/2509.01092v1 (解码范式革新、KV 缓存极致压缩、显存占用降低)
(NeurIPS'2025) KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse：跨查询 KV 缓存复用框架，通过特殊标记识别共享上下文，动态构建 KV 映射表，实现检索文档块缓存复用；
暂未公开 (RAG 上下文复用、KV 跨查询共享、TTFT 降低 3.2-5.1×)
(NeurIPS'2025) Graph-KV: Breaking Sequence via Injecting Structural Biases into Large Language Models：将图结构信息注入 KV 表示，增强复杂关系推理能力，准确率提升 12-15%；
https://openreview.net/pdf/ad782d6eade7cc1f82de7c69fcc3736c07be2ac3.pdf (图结构 RAG 优化、KV 缓存结构增强、关系推理适配)
(FAST'2025) Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot：以 KV 缓存为中心的 LLM 推理架构，分层存储与预计算策略，获最佳论文奖；
https://www.usenix.org/conference/fast25/presentation/qin (RAG 服务架构优化、KV 分层管理、吞吐量提升 3.5 倍)
(FAST'2025) IMPRESS: An Importance-Informed Multi-Tier Prefix KV Storage System for Large Language Model Inference：重要性感知的多层级前缀 KV 存储系统，冷热数据分离与预取，减少 I/O 时间 1.5-3.8 倍；
暂未公开 (RAG 服务存储优化、KV 分层存储、I/O 效率提升)
(EuroSys'2025) CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion：缓存知识融合技术，实现多预计算 KV 缓存无缝拼接，100% 缓存命中率，获最佳论文；
https://blog.lmcache.ai/en/2025/03/31/cacheblend-best-paper-acm-eurosys25-enabling-100-kv-cache-hit-rate-in-rag/、https://github.com/YaoJiayi/CacheBlend (RAG 缓存融合、多文本块 KV 复用、TTFT 降低 2.2-3.3×)
(EuroSys'2025) ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference：影子 KV 缓存机制，主缓存存高频 token，影子缓存存关键低频 token，吞吐量提升 3 倍 +；
暂未公开 (分层 KV 缓存、长文本 RAG 适配、吞吐量与质量平衡)
(USENIX ATC'2025) A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints：首个结合计算与 KV 缓存约束的排队论框架，推导稳定性条件，指导 GPU 资源规划；
暂未公开 (可参考 USENIX ATC 2025 官网：https://www.usenix.org/conference/atc25) (KV 缓存资源调度、LLM 推理系统稳定性、RAG 部署优化)
(USENIX ATC'2025) Leveraging Approximate Caching for Faster Retrieval-Augmented Generation：近似缓存策略，允许小幅检索误差换取更高 KV 缓存命中率，端到端延迟降低 30%+；
暂未公开 (近似 RAG 缓存、误差 - 效率权衡、大规模部署适配)
(EMNLP'2025) TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text：混合离线 - 在线 RAG 架构，预计算文档块 KV 缓存，独立注意力与位置重编码实现无缝拼接；
暂未公开 (可参考 EMNLP 2025 官网：https://2025.emnlp.org/) (预计算 RAG 范式、文档块级 KV 复用、TTFT 提升 8.6 倍)
(EMNLP'2025) RefreshKV: Updating Small KV Cache During Long-form Generation：交替使用全上下文注意力与子集注意力，基于注意力模式动态更新小型 KV 缓存，提升长文本生成性能；
暂未公开 (长文本 RAG 生成、动态 KV 缓存更新、生成质量保障)
(ACL'2025) CACHECLIP: Accelerating RAG with Effective KV Cache Reuse：辅助模型预测主模型注意力分布，识别关键 token 实现 KV 缓存复用，提升推理效率；
暂未公开 (可参考 ACL Anthology：https://aclanthology.org/) (注意力感知缓存、跨模型复用、RAG 推理加速)
(ACL'2025) A Systematic Study of Cross-Layer KV Sharing for Efficient LLM Inference：统一跨层 KV 共享框架，系统评估不同配置，2 倍缓存压缩下保持高吞吐量与性能；
暂未公开 (跨层 KV 复用、RAG 推理效率、多配置对比)
(VLDB'2025) Chameleon: A Heterogeneous and Disaggregated Accelerator System for Retrieval-Augmented Language Models：异构解耦的 RAG 加速架构，KV 缓存与检索计算分离到专用硬件，端到端延迟降低 40%+；
暂未公开 (可参考 VLDB 2025 官网：https://vldb.org/2025/) (RAG 硬件加速、KV 缓存专用化、异构计算协同)
(SIGMOD'2025) RAG-DCache: A Distributed KV Cache Management System for RAG-Powered LLMs：分布式 KV 缓存管理系统，利用查询局部性与等待时间优化缓存分配，GPU 内存占用减少 75%；
https://blog.csdn.net/u013524655/article/details/147318381 (分布式 RAG 缓存、多实例协同、生产环境适配)
(ISCA'2025) REIS: A High-Performance and Energy-Efficient Retrieval System with In-Storage Processing：近数据处理的检索系统，KV 缓存预计算嵌入存储层，减少数据移动开销，延迟降低 50%+；
暂未公开 (可参考 ISCA 2025 官网：https://isca2025.cs.ucr.edu/) (近数据 KV 计算、存储 - 计算融合、RAG 检索效率优化)
(MLSys'2025) Shared Disk KV Cache Management for Efficient Multi-Instance Inference in RAG-Powered LLMs：共享磁盘 KV 缓存管理方案，解决多实例推理缓存一致性与资源竞争问题；
暂未公开 (分布式 RAG 缓存、多实例协同、GPU 内存占用降低 75%)
(MLSys'2025) Cache-Craft: Managing Chunk-Caches for Efficient Retrieval-Augmented Generation：块缓存管理系统，支持前缀与非前缀 token 块缓存复用，位置偏移管理与部分重计算；
https://github.com/ustc-sunny/Awsome-RAG-LLM-Inference-System (RAG 块缓存管理、位置偏移处理、生产环境适配)
(MLSys'2025) SpeCache: Speculative Key-Value Caching for Efficient Generation of LLMs：训练无关的投机 KV 缓存方法，完整 KV 卸载到 CPU，推理仅获取 top-k 关键条目，GPU 内存减少 70-80%；
暂未公开 (RAG 内存优化、KV 投机获取、CPU/GPU 协同推理)
(ICLR'2026 已提交) ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs：零样本 KV 缓存压缩框架，多维 token 重要性动态分配内存，残差合并保留关键上下文；
暂未公开 (KV 动态压缩、零样本适配 RAG、长上下文优化)
(ICLR'2026 已提交) Hierarchical Adaptive Eviction (HAE)：多模态 LLM 的 KV 缓存驱逐框架，预填充双注意力剪枝，解码动态驱逐，内存降低 41%；
暂未公开 (多模态 KV 缓存管理、RAG 多模态适配、精度损失极小)
(ICLR'2026 已提交) HOLD ONTO THAT THOUGHT: ASSESSING KV CACHE COMPRESSION ON REASONING：系统评估 KV 压缩对推理任务影响，提出推理感知压缩策略，4-8× 压缩比保持推理能力；
暂未公开 (RAG 推理质量保障、KV 压缩与推理平衡、多任务评估)
(ICLR'2026 已提交) ORACLEKV: ORACLE GUIDANCE FOR QUESTION-INDEPENDENT KV CACHE EVICTION：Oracle 指导的无查询依赖 KV 驱逐策略，离线学习 token 重要性，在线动态驱逐非关键 token；
暂未公开 (KV 智能驱逐、RAG 长上下文管理、跨任务适配)
(ICLR'2026 已接收) Parallel Key-Value Cache Fusion for Position Invariant RAG：并行 KV 缓存融合技术，支持位置不变的 RAG 推理，处理多输入段落保持鲁棒性；
