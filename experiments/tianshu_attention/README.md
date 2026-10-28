# BI-V150 变长 FlashAttention grid 候选

基点：FlagGems v5.3.5，4df52d9168ac55c1c1061cefb4638c570292d896。分支 exp/tianshu-attention-20261011。

FLAGGEMS_ILUVATAR_COMPACT_VARLEN 默认为0，显式置1后仅在Iluvatar、存在cu_seqlens_q、无dropout时启用。将最大查询长度×批大小的二维分配替换为总查询token数的紧凑grid。每个query tile在GPU上二分查找所属序列，不复制GPU序列长度到CPU。

设S_i为累计query tokens，BLOCK_M为原heuristic选择的tile大小。tile前缀floor(S_i/BLOCK_M)+i严格递增，其相邻差值足以覆盖第i条请求的ceil(query_len_i/BLOCK_M)个tile。总分配floor(total_query_tokens/BLOCK_M)+batch个tile，每序列至多多一个；空tile返回。原heuristic、QK/softmax/PV顺序、dtype和KV布局保留。纯单query GQA交换路径无cu_seqlens_q，沿用原grid。

CPU随机覆盖检查通过5000组不同序列与BLOCK_M，包括空序列。GPU七组检查确认compact与关闭开关的原FlashAttention逐元素完全一致，并对固定FP32参考通过原BF16 attention容差atol=.02、rtol=.01。源哈希与结果在工作区service-results/20261011_compact_attention_validation_tianshu和20261011_split_attention_validation_tianshu。

仅compact在代表性的64路混合批次约有6%–7%收益，仍慢于原vLLM通用路径。因此插件还需要单独的Prefill/Decode分离，将纯Decode保留在现成GQA快路径；本改动不能单独认定为比赛吞吐提升。联合实现与4K试验脚本在天数插件experiments/tianshu_attention。

源安装使用pip --no-build-isolation --no-deps，不升级Torch、Triton或vLLM。默认关闭可回到原分支。未修改任何沐曦仓库。
