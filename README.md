| 顶会类别 | 会议 | 年份 | 论文标题 | 核心创新 | 核心关联点 | 论文链接（OpenReview/会议官网/代码仓库） |
| --- | --- | --- | --- | --- | --- | --- |

| AI基础研究 | ICLR | 2025 | SparseCache: Extreme Sparse Coding for KV Cache Compression | 基于字典学习和稀疏编码的KV缓存压缩框架，离线学习全局共享字典，在线OMP稀疏编码与重构，实现极高压缩比 | KV缓存压缩、RAG预计算KV优化、存储效率提升 | [https://openreview.net/forum?id=43zTdoRqY4](https://openreview.net/forum?id=43zTdoRqY4)（OpenReview） |

| AI基础研究 | ICLR | 2025 | AtlasKV: Augmenting LLMs with Billion-Scale Knowledge Graphs in 20GB VRAM | 提出KG2KV与HiKVP，将知识图谱转化为KV嵌入并分层管理，无需外部检索器集成十亿级知识图谱 | KV与知识图谱融合、RAG知识嵌入、显存优化 | 暂未公开 |

| AI基础研究 | ICLR | 2025 | Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting | 小模型并行生成检索增强草稿，大模型验证，缓解长输入推理速度慢与理解质量下降问题 | LLM推理加速、RAG流程优化、投机解码适配 | 暂未公开 |

| AI基础研究 | ICLR | 2025 | RACC: Retrieval-Augmented KV Cache Compression in Long-Context Generation | 检索感知的KV缓存压缩，利用检索相关性排序token重要性，仅保留15%缓存维持无损准确性 | RAG长文本优化、检索感知压缩、多方法正交应用 | 暂未公开 |

| AI基础研究 | ICLR | 2025 | OjaKV: Context-Aware Online Low-Rank KV Cache Compression with Oja’s Rule | 基于Oja规则的上下文感知在线低秩KV压缩，动态调整压缩率，避免上下文漂移，兼容FlashAttention | 在线KV压缩、低秩分解、实时推理适配 | 暂未公开 |

| AI基础研究 | ICML | 2025 | LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models | 梯形KV缓存模式，跨层存储KV对，结合距离动态压缩，实现长上下文无OOM持续生成 | KV缓存结构创新、长文本RAG推理优化 | 暂未公开（可参考ICML 2025官网：[https://icml.cc/Conferences/2025](https://icml.cc/Conferences/2025)） |

| AI基础研究 | ICML | 2025 | REFRAG: Rethinking RAG Based Decoding | 重新设计RAG解码流程，将KV缓存需求降至传统方法的3%，首词延迟大幅降低，保持低困惑度 | 解码范式革新、KV缓存极致压缩、显存占用降低 | [https://arxiv.org/abs/2509.01092v1](https://arxiv.org/abs/2509.01092v1)（arXiv预印本） |

| AI基础研究 | NeurIPS | 2025 | KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse | 跨查询KV缓存复用框架，通过特殊标记识别共享上下文，动态构建KV映射表，实现检索文档块缓存复用 | RAG上下文复用、KV跨查询共享、TTFT降低3.2-5.1× | 暂未公开 |

| AI基础研究 | NeurIPS | 2025 | Graph-KV: Breaking Sequence via Injecting Structural Biases into Large Language Models | 将图结构信息注入KV表示，增强复杂关系推理能力，准确率提升12-15% | 图结构RAG优化、KV缓存结构增强、关系推理适配 | [https://openreview.net/pdf/ad782d6eade7cc1f82de7c69fcc3736c07be2ac3.pdf](https://openreview.net/pdf/ad782d6eade7cc1f82de7c69fcc3736c07be2ac3.pdf)（OpenReview） |

| 系统优化 | FAST | 2025 | Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot | 以KV缓存为中心的LLM推理架构，分层存储与预计算策略，获最佳论文奖 | RAG服务架构优化、KV分层管理、吞吐量提升3.5倍 | [https://www.usenix.org/conference/fast25/presentation/qin](https://www.usenix.org/conference/fast25/presentation/qin)（USENIX FAST 2025官网） |

| 系统优化 | FAST | 2025 | IMPRESS: An Importance-Informed Multi-Tier Prefix KV Storage System for Large Language Model Inference | 重要性感知的多层级前缀KV存储系统，冷热数据分离与预取，减少I/O时间1.5-3.8倍 | RAG服务存储优化、KV分层存储、I/O效率提升 | 暂未公开 |

| 系统优化 | EuroSys | 2025 | CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion | 缓存知识融合技术，实现多预计算KV缓存无缝拼接，100%缓存命中率，获最佳论文 | RAG缓存融合、多文本块KV复用、TTFT降低2.2-3.3× | [https://blog.lmcache.ai/en/2025/03/31/cacheblend-best-paper-acm-eurosys25-enabling-100-kv-cache-hit-rate-in-rag/](https://blog.lmcache.ai/en/2025/03/31/cacheblend-best-paper-acm-eurosys25-enabling-100-kv-cache-hit-rate-in-rag/)（作者团队博客，含代码链接）；[https://github.com/YaoJiayi/CacheBlend](https://github.com/YaoJiayi/CacheBlend)（GitHub代码） |

| 系统优化 | EuroSys | 2025 | ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference | 影子KV缓存机制，主缓存存高频token，影子缓存存关键低频token，吞吐量提升3倍+ | 分层KV缓存、长文本RAG适配、吞吐量与质量平衡 | 暂未公开 |

| 系统优化 | USENIX ATC | 2025 | A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints | 首个结合计算与KV缓存约束的排队论框架，推导稳定性条件，指导GPU资源规划 | KV缓存资源调度、LLM推理系统稳定性、RAG部署优化 | 暂未公开（可参考USENIX ATC 2025官网：[https://www.usenix.org/conference/atc25](https://www.usenix.org/conference/atc25)） |

| 系统优化 | USENIX ATC | 2025 | Leveraging Approximate Caching for Faster Retrieval-Augmented Generation | 近似缓存策略，允许小幅检索误差换取更高KV缓存命中率，端到端延迟降低30%+ | 近似RAG缓存、误差-效率权衡、大规模部署适配 | 暂未公开 |

| 系统优化 | OSDI | 2025 | （暂未收录直接关联论文，预计2026年有相关突破） | - | - | 待发布（可关注OSDI 2026官网：[https://www.usenix.org/conference/osdi26](https://www.usenix.org/conference/osdi26)） |

| 系统优化 | SOSP | 2025 | （暂未收录直接关联论文，聚焦KV缓存底层机制研究） | - | - | 待发布（可关注SOSP 2025官网：[https://sosp2025.cs.princeton.edu/](https://sosp2025.cs.princeton.edu/)） |

| NLP应用 | EMNLP | 2025 | TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text | 混合离线-在线RAG架构，预计算文档块KV缓存，独立注意力与位置重编码实现无缝拼接 | 预计算RAG范式、文档块级KV复用、TTFT提升8.6倍 | 暂未公开（可参考EMNLP 2025官网：[https://2025.emnlp.org/](https://2025.emnlp.org/)） |

| NLP应用 | EMNLP | 2025 | RefreshKV: Updating Small KV Cache During Long-form Generation | 交替使用全上下文注意力与子集注意力，基于注意力模式动态更新小型KV缓存，提升长文本生成性能 | 长文本RAG生成、动态KV缓存更新、生成质量保障 | 暂未公开 |

| NLP应用 | ACL | 2025 | CACHECLIP: Accelerating RAG with Effective KV Cache Reuse | 辅助模型预测主模型注意力分布，识别关键token实现KV缓存复用，提升推理效率 | 注意力感知缓存、跨模型复用、RAG推理加速 | 暂未公开（可参考ACL Anthology：[https://aclanthology.org/](https://aclanthology.org/)） |

| NLP应用 | ACL | 2025 | A Systematic Study of Cross-Layer KV Sharing for Efficient LLM Inference | 统一跨层KV共享框架，系统评估不同配置，2倍缓存压缩下保持高吞吐量与性能 | 跨层KV复用、RAG推理效率、多配置对比 | 暂未公开 |

| NLP应用 | NAACL | 2025 | （补充关联论文，聚焦跨层KV共享系统研究） | - | - | 暂未公开 |

| 数据库 | VLDB | 2025 | Chameleon: A Heterogeneous and Disaggregated Accelerator System for Retrieval-Augmented Language Models | 异构解耦的RAG加速架构，KV缓存与检索计算分离到专用硬件，端到端延迟降低40%+ | RAG硬件加速、KV缓存专用化、异构计算协同 | 暂未公开（可参考VLDB 2025官网：[https://vldb.org/2025/](https://vldb.org/2025/)） |

| 数据库 | VLDB | 2025 | （聚焦RAG检索与KV缓存结合的存储优化） | - | - | 暂未公开 |

| 数据库 | SIGMOD | 2025 | RAG-DCache: A Distributed KV Cache Management System for RAG-Powered LLMs | 分布式KV缓存管理系统，利用查询局部性与等待时间优化缓存分配，GPU内存占用减少75% | 分布式RAG缓存、多实例协同、生产环境适配 | [https://blog.csdn.net/u013524655/article/details/147318381](https://blog.csdn.net/u013524655/article/details/147318381)（论文解读，含核心信息） |

| 体系结构 | ISCA | 2025 | REIS: A High-Performance and Energy-Efficient Retrieval System with In-Storage Processing | 近数据处理的检索系统，KV缓存预计算嵌入存储层，减少数据移动开销，延迟降低50%+ | 近数据KV计算、存储-计算融合、RAG检索效率优化 | 暂未公开（可参考ISCA 2025官网：[https://isca2025.cs.ucr.edu/](https://isca2025.cs.ucr.edu/)） |

| 体系结构 | ISCA | 2025 | （聚焦KV缓存硬件加速与近数据处理优化） | - | - | 暂未公开 |

| 交叉领域 | MLSys | 2025 | Shared Disk KV Cache Management for Efficient Multi-Instance Inference in RAG-Powered LLMs | 共享磁盘KV缓存管理方案，解决多实例推理缓存一致性与资源竞争问题 | 分布式RAG缓存、多实例协同、GPU内存占用降低75% | 暂未公开 |

| 交叉领域 | MLSys | 2025 | Cache-Craft: Managing Chunk-Caches for Efficient Retrieval-Augmented Generation | 块缓存管理系统，支持前缀与非前缀token块缓存复用，位置偏移管理与部分重计算 | RAG块缓存管理、位置偏移处理、生产环境适配 | [https://github.com/ustc-sunny/Awsome-RAG-LLM-Inference-System](https://github.com/ustc-sunny/Awsome-RAG-LLM-Inference-System)（相关项目收录，含论文索引） |

| 交叉领域 | MLSys | 2025 | SpeCache: Speculative Key-Value Caching for Efficient Generation of LLMs | 训练无关的投机KV缓存方法，完整KV卸载到CPU，推理仅获取top-k关键条目，GPU内存减少70-80% | RAG内存优化、KV投机获取、CPU/GPU协同推理 | 暂未公开 |

| 交叉领域 | EuroMLSys | 2025 | （聚焦系统-算法协同优化，预计算RAG架构研究） | - | - | 暂未公开 |

| 2026年（已提交/已接收） | ICLR | 2026（已提交） | ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs | 零样本KV缓存压缩框架，多维token重要性动态分配内存，残差合并保留关键上下文 | KV动态压缩、零样本适配RAG、长上下文优化 | 暂未公开（已提交至ICLR 2026，可关注OpenReview更新） |

| 2026年（已提交/已接收） | ICLR | 2026（已提交） | Hierarchical Adaptive Eviction (HAE) | 多模态LLM的KV缓存驱逐框架，预填充双注意力剪枝，解码动态驱逐，内存降低41% | 多模态KV缓存管理、RAG多模态适配、精度损失极小 | 暂未公开（已提交至ICLR 2026，可关注OpenReview更新） |

| 2026年（已提交/已接收） | ICLR | 2026（已提交） | HOLD ONTO THAT THOUGHT: ASSESSING KV CACHE COMPRESSION ON REASONING | 系统评估KV压缩对推理任务影响，提出推理感知压缩策略，4-8×压缩比保持推理能力 | RAG推理质量保障、KV压缩与推理平衡、多任务评估 | 暂未公开（已提交至ICLR 2026，可关注OpenReview更新） |

| 2026年（已提交/已接收） | ICLR | 2026（已提交） | ORACLEKV: ORACLE GUIDANCE FOR QUESTION-INDEPENDENT KV CACHE EVICTION | Oracle指导的无查询依赖KV驱逐策略，离线学习token重要性，在线动态驱逐非关键token | KV智能驱逐、RAG长上下文管理、跨任务适配 | 暂未公开（已提交至ICLR 2026，可关注OpenReview更新） |

| 2026年（已提交/已接收） | ICLR | 2026（已接收） | Parallel Key-Value Cache Fusion for Position Invariant RAG | 并行KV缓存融合技术，支持位置不变的RAG推理，处理多输入段落保持鲁棒性 | 位置不变缓存、多段落融合、并行处理 | 暂未公开（已接收至ICLR 2026，将在会议官网发布） |
