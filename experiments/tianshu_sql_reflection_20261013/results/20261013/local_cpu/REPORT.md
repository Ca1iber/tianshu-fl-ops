# 天数配置读取：仅反射目标表的本地 CPU 验证

诊断证据见 vLLM 仓库 cold_start_20261013 的 diagnostic_32g 报告：首轮低吞吐时 119 个样本中 116 个处于 SQLAlchemy，96 个位于 build_sql_model_by_db。FlagGems 每个新缓存表加载会重新反射整个 SQLite 库；此实验只缩小反射范围，不改配置数值，不修复 pool key，不改算子或调优策略。

本机 Python 3.13.13，采用容器已安装 SQLAlchemy 的纯 Python 源码副本。在本地执行，未与天数正在计时的 GPU 争用 CPU。数据源为只读 SQLite backup 的一致性快照：TunedConfig_iluvatar_triton_3_6.db，902 个表，没有外键；SHA256 d0397de7974ba6b3d83ef8c9749087e8a9b781ec231a0681b8bd532b5fffda94，15773696 bytes。数据库读取前后哈希一致。

合成 128 表检查：三张存在表的列、复合主键、选定行配置相同；缺失表同样返回 None。真实快照抽查 index、mm、linearized index、mul 四类表，全部列、主键和最多 32 行的完整值相同；缺失表返回行为相同。真实已有表的全库反射每次 1.50–1.99 秒，目标表反射约 4.06–4.58 毫秒；缺失表为 1.868→0.736 秒。这些是本地 CPU 微测，不代表天数服务吞吐或首轮耗时的实测收益。

原始数据、数据库快照、SQLAlchemy 来源副本和本地结果保存在 service-results/20261013_tianshu/sql_reflection_local_cpu；逐文件 SHA 见 backup_manifest.json。两个 probe 和 reflect_one.py 均通过本地真实数据读取核验。process-only 服务 hook 和 run_service.py 尚未在真实 GPU 服务中验证，不作为已采纳优化提交。后续需要同一计算配置、全新服务首轮耗时对照、actual SQL 路由和匹配的配置读取审计。
