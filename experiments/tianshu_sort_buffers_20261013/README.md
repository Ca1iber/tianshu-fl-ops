# 天数32g 私有排序缓冲int32

已完成120项GPU值/稳定索引/图重放精确检查。内部index32仍返回public int64；n<2**30确保数据范围安全，地址仍Triton指针。M64词表完整sort约24.95→21.40ms（速度+16.6%），前缀和32单独无清晰收益。真实服务正在Tianshu vLLM实验目录单独验证，不把micro当正式成绩。

报告及SHA：results/20261013/micro_32g/REPORT_32g.md。仅生成进程函数，不改installed source；本自有ops仓库已核实origin，沐曦不变。复现probe_buffers.py --output新的目录，GPU需闲置和预约有效。generated source包含private.dtype32断言、末尾public64转换。
