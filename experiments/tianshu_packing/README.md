# BI-V150 权重布局重排实验

独立实验源，不进入 FlagGems 默认分派。沐曦只读来源 3b162a7ba78b4269c62f72d05023949c1e72dbd5，精确参考 SHA 与修改范围在 reference_provenance.json。

- run_packing_probe.py：直接迁移与相同 tile/layout 控制，实际 MiniCPM 第 0 层和完整词表。
- run_layer_cycle.py：42 层真实权重轮换，复核单块权重缓存影响。
- run_service_probe.py / run_service_series.py：原版、同配置普通 down、同配置重排 down 的串行短 4K 服务对照。
- hooks/sitecustomize.py / service_hook.py：仅测试子进程显式设置 TIANSHU_PACK_PROBE_MODE 时安装临时 launch hook。不改安装包、源权重、Attention、采样或官方 benchmark。仅 M2048/N2048/K6144 BF16 down 生效，其它形状保留原版。

服务 workload 为五轮、16 请求、并发16、4096 输入/64 输出；增加 2048/4096 greedy 请求验证真正命中路径。不是官方成绩。每组结束停止自有服务，23:58 watchdog 来源为用户确认的当日预约。运行前要更新/核验 lease，不能把本次授权视为未来日期续约。

完整原始材料与报告位于工作区 service-results/20261011_2015_weight_packing_tianshu。GPU 实验串行，正确性与耗时分开。不要把权重布局与 GROUP_M/tile/融合联合收益混为一谈。

本次结论：42层轮换中down重排比相同配置普通布局减少3.44%算子耗时。短4K服务原版3279.13、同配置普通布局3284.52、重排3293.45 total tok/s；独立重排的差值+0.272%，联合比原版+0.437%。新增1008MiB权重，实测KV521936→489072（-6.30%）。所有261请求完成，三处真实down输出位级一致，7组greedy文本/usage一致。未运行官方完整benchmark或MATH，保持实验，不默认启用。

实验钩子会由测试环境PYTHONPATH传入子进程；三组startup后台usage/cpuinfo线程均出现JSONDecodeError，原始日志保留，服务与请求正常。未修改vLLM核心来屏蔽此问题。Python callbacks仅作路径命中证据，不是图重放launch统计。
