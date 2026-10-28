# 天数 BI-V150 32g：排序私有索引与前缀和int32

用户要求继续寻找int64→int32，审计发现radix_sort的索引ping-pong缓冲和torch.cumsum(global_hist)默认64位中间张量。此处原本已有n<2**30约束，索引0..n-1和频数前缀0..n均安全适用signed int32。

原型只复制原Python radix_sort，不改installed source，局部histogram仍原int64。index32：内部arange及两索引缓冲用int32，返回时精确转换int64；scan32：cumsum显式int32、非负前缀view为原uint32；both32组合两者。运行断言private dtype32，返回int64契约保持原样。

30种输入场景×4实现=120项GPU数值及CUDAgraph重放检查全部通过，值与CPU完全一致、稳定索引逐项相同、输出dtype为int64。涵盖随机/重复/负无穷、正反排序、8191/8193尾部和词表130560的M1/M32/M64。七轮旋转计时、每轮10重放取中位数，包含最终int64转换成本。

M64×130560随机float32升序：

|实现|完整sort/ms|相对原版速度提升|
|---|---:|---:|
|original|24.954239|0.000000%|
|index32|21.397672|16.621282%|
|scan32|24.954361|-0.000489%|
|both32|21.360211|16.825806%|

结论：private索引32有约16.6%的完整排序微测速度收益；前缀和32单独几乎没有收益，组合增量很小，优先只index32做实际服务对照。这个百分比不是vLLM吞吐提升；候选独立Math和完整正式成绩尚未运行，不能替换正式最好。

全量归档2012098bytes、8parts，SHA256 3f3fbd3848a69c90bec5ad5bfa14029ab40adbe36005b485f3efc9af35e36a32，本地完整核验。helper/probe与真实源快照逐字相同；未采集硬件profiler。不改模型、BF16权重、采样参数或官方benchmark。独立自有Tianshu ops origin已核对Ca1iber/tianshu-fl-ops.git，沐曦仓库只读。

后续热路径核查：天数vLLM batch>=8走原生Triton top-p，仅batch<8走FlagGems sort；MetaX补丁强制PyTorch排序。因此M64 sorter微测不能作为天数大批次热路径收益证明，详见vLLM私有索引服务报告。
