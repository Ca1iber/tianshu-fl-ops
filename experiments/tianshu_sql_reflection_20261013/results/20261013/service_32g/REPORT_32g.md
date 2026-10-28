# 天数32g：只反射目标配置表的服务实测

4K/1K、C64/N64，新服务三轮，计算 profile 地址64/Native L0/Decode Stage1/75桶+gather保持不变。只替换SQLPersistantModel.build_sql_model_by_db；不改调优配置值、不改pool key、不混入新attention32或KV写入候选。

复用原计数实验的完整控制数据。installed compute sources逐字hash一致，提示seed/usage一致，两组7个稀有shape greedy响应全部相同。真实SQL hook已产生目标表读取记录，helper SHA4cfc6a34e29e0cc220b83d79898c1da4d986e35d4eb0796c37947b1631064c02；真实配置库902表只读快照及复合主键/配置行等价核验先行通过，本机和容器CPU合成检查均通过。

首轮耗时：258.122086→106.722455秒，减少58.654%。候选三轮3070.394142/3257.855410/3257.352824 total tok/s；耗时106.722455/100.581505/100.597024秒。候选服务启动60.080337秒。配对计分轮2/3控制3253.599268、候选3257.604117 total tok/s，变化0.123090%，不视为明确稳态吞吐收益。原始benchmark排首轮，所以不能把冷启动比例直接算成官方吞吐提升；主要价值是减少首次请求停顿和实验等待。这里不是重新跑的新控制组，数据库缓存与系统跨时段变化是比较限制。

原始归档2862645 bytes、11分片及整包SHA256 11dd697933d5fcb08bc3a1e69186d9cdde05442dc0407bc8eb23fe2ced170a3a全部本地verified。全部响应、source快照、SQL actual route与GPU cleanup均保留在service-results/20261013_tianshu/sql_target_reflection_service。服务helper/hook/script已逐字核对运行快照。新增Math与正式评测正在独立attention32配置下进行；本SQL改动尚未合并进正式计算profile或替代现有成绩。
