# 背景介绍

长序列场景：PCP/DCP、变长序列（动态CP、CPP）

1. 动态CP代码上仓：vllm，vllm\-ascend；

2. 增大req/s，看吞吐是否还有上升空间；

3. 结果画图、分析为啥ttft的收益比吞吐更高（尤其是2:8场景）、32的batch size已经测试了长短比例0:10，2:8,5:5,8:2，10:0；

4. 测试结果的整理，包括：测试脚本、测试场景说明、精度测试、测试结果、结果的分析；

5. 新浪、**工行**（0\-2K 67.5%，2\-4K 10%，4\-6k 17%，6\-8k 0.5%，8\-10K 1.5%，10\-12K 1.5%， 12\-14K 0.5%， 14\-16K 0.5%，16\-18K 0.5%，23K 0.5%）、交行（16k以下60%，32k 20%，64k 15%，128k 5%）性能分析，这种场景是长序列占多，收益摸测一把，开和关动态CP的性能对比，需要考虑上CPP；

# 方案介绍

方案介绍视频：[clouddrive.huawei.com](https://clouddrive.huawei.com/p/e2e87ca9ee043d36a53674089d075e92)

前期结果：[Wiki](https://wiki.huawei.com/domains/75348/wiki/229426/WIKI2026020510065978?title=_4b8f173e)

## kv cache 在connector架构层面的处理流程

![](https://clouddocs.huawei.com/koopage/v1/app/api/documents/doc/preview/30a7fb16-d7f5-4c02-95e4-4441db2c5c58?document_id=4b6e1b80-7dab-4679-ae88-d50ea0559a22 "")

传统处理流程：

1. scheduler调度输出1个scheduler\_output到multiprocExecutor；

2. multiprocExecutor中的collective\_rpc负责RPC任务流的分发和收集，它会将计算任务发送给多个worker线程（例如TP，如果开了domain还会发送给dp/cp）；

3. 然后worker线程调用modelrunner模块进行计算；

4. modelrunner可能还会调用kvconnector进行kv cache的传输操作，并返回kvConnectorOutput到modelrunner\_output；

5. worker再将各自线程的计算结果返回给MultiprocExecutor，此处需要注意，可能每个worker线程执行的有快有慢，例如kv cache传输不一定同步完成，大家返回的kv cache output结果可能不一致；

6. 由于worker返回到MultiprocExecutor的结果可能不一致，MultiprocExecutor会调用aggregate函数去收集所有worker的结果：1）如果kv\_output\_aggregator为非空（即开了connector），会调用kv\_output\_aggregator.aggregate函数，收集所有worker的kv output信息，如果通过计数发现大家（所有TP）都完成了kv cache传输（这个传输完成的信号最初来自mooncake），就会将request添加到finished\_sending/finished\_recving，然后将TP张卡的结果只选取output\_rank（通常是随机选择TP0的结果）输出到output，并将finished\_sending/finished\_recving结果更新到output中，这就完成了多个TP结果的整理集成；2）如果kv\_output\_aggregator为空（即没有开connector），则aggregate函数会被设置为透传函数（lambda x:x），所有输入的output\_list直接输出，但在get\_response函数中会只选取0号卡的结果，也是实现了将TP结果整理成一个output输出；

![](https://clouddocs.huawei.com/koopage/v1/app/api/documents/doc/preview/44e5eff2-ddcb-4f8e-88e8-0ca93b6df086?document_id=4b6e1b80-7dab-4679-ae88-d50ea0559a22 "")

![](https://clouddocs.huawei.com/koopage/v1/app/api/documents/doc/preview/3bd4feda-87a1-4c07-b4ba-c7ce84e0d8c0?document_id=4b6e1b80-7dab-4679-ae88-d50ea0559a22 "")

![](https://clouddocs.huawei.com/koopage/v1/app/api/documents/doc/preview/7b9c4528-6e4b-4932-ab10-19c9bf7846cd?document_id=4b6e1b80-7dab-4679-ae88-d50ea0559a22 "image.png")

7. MultiprocExecutor将集成的结果再传输给scheduler，scheduler调用update\_from\_output函数处理scheduler\_output，model\_runner*\_*output,这其中会调用\_update*\_*from\_kv\_xfer\_finished函数，基于finished\_sending/finished\_recving来更新request的调度状态，以及决策是否free block。

![](https://clouddocs.huawei.com/koopage/v1/app/api/documents/doc/preview/07de8db5-73a9-4394-bc7e-95df146aa5bb?document_id=4b6e1b80-7dab-4679-ae88-d50ea0559a22 "")

![](https://clouddocs.huawei.com/koopage/v1/app/api/documents/doc/preview/68dbabfd-dcb8-4b58-841e-9ed01e040ad4?document_id=4b6e1b80-7dab-4679-ae88-d50ea0559a22 "")

以上就是传统kv cache在connector架构层面的处理流程，总结一下：

scheduler输出一个cp\_size个scheduler\_output到multiprocexecutor，multiprocexecutor调度多worker线程基于scheduler调度结果通过modelrunner进行计算或者传输kv cache，传输kv cache是modelrunner调度到mooncake执行的，然后将kv connector结果返回到modelrunneroutput中，tp\*cp个worker的modelrunneroutput汇集到multiprocexecutor中，multiprocexecutor通过collective\_rpc调用aggregator函数将多个TP结果压缩到一个，并返回cp个结果给scheduler，scheduler通过update\_from\_output调用\_update*\_*from\_kv\_xfer\_finished函数，更新request的state，并且决策是否释放block内存；

具体到domain方案中，差异点是：scheduler会数出cp个shceduler\_output结果，aggregator会数出cp个结果，update\_from\_output会接收cp个scheduler+modelrunneroutput，并在cp域内做结果的循环更新。

vllm v1引擎两阶段执行：

1. 第一阶段（execute\_model）：执行前向传播，计算logits，返回None；

2. 第二阶段（sample\_tokens）：基于logits进行采样，返回ModelRunnerOutput；

ModelRunner的输出有3种形式：None,False,ModelRunnerOutput

ModelRunnerOutput表示是正常request计算；

None表示本轮没有计算也没有kv cache传输；

False是domain才有的，表示本卡没有分到request数据，需要跳过；

# 代码仓

[https://github.com/Kurumi5210/vllm/tree/dev\-dycp\-release\-0.16](https://github.com/Kurumi5210/vllm/tree/dev-dycp-release-0.16)

[https://github.com/Kurumi5210/vllm\-ascend/tree/dev\-dycp\-release\-0.16](https://github.com/Kurumi5210/vllm-ascend/tree/dev-dycp-release-0.16)

# RFC

[https://github.com/vllm\-project/vllm/issues/29295](https://github.com/vllm-project/vllm/issues/29295)

# 精度验证

验证数据集：gsm8k，PD分离配置

测试模型：deepseek R1

常用配置：1P1D，P：tp8dp2，D：dp16

## 1. prefill（domain）+decode（no\-domain）

精度正常

![](https://clouddocs.huawei.com/koopage/v1/app/api/documents/doc/preview/7fc340aa-6834-4b22-89cd-9cabf7020808?document_id=4b6e1b80-7dab-4679-ae88-d50ea0559a22 "")

0416测试结果：跑了100条gsm8k数据，评分是97，精度正常

## 2. prefill（no\-domain）+decode（domain）

## 3. prefill（domain）+decode（domain）

0418测试结果：跑了200条gsm8k数据，评分是

# prefill收益测试

## 测试方案

测试模型：Deepseek\-v2 lite  
模型配置：TP8DP2，TP8PCP2，TP16，TP8DyCP2  
测试场景：4K，4K+64K（1%），4K+128K（1%）  
测试prefill，TTFT满足2s的SLO要求（4K,TP4DP4），然后测试吞吐；  
打流服务可以先inf配置concurrency，requestnum=3000，然后从1逐步调大request rate，可选\[1,2,4,8,16\];  
脚本中maxnumrequest=32，maxmodellen=128k,maxnumbatchtokens=128k

## 机器：

6台机器，可以组3个双机器PD分离

## 收益结果

## 1. prefill（no\-domain）+decode（domain）

## 2. prefill（domain）+decode（domain）

# 阶段进展

动态CP开发周报：[clouddrive.huawei.com](https://clouddrive.huawei.com/p/e611ffad75ab1bbe2ad7d298a5613119)

330当前进展：

1. prefill+PD分离+decode 跑通；

2. 图模式的算子已有初版

3. prefill（domain）+deocde（no\-domain）精度正常

4. prefill（domain）+deocde（no\-domain）收益验证，正在进行

后期规划：

1. 图模式验证  短请求采用ok，长请求待测试  陈潇  志祥

2. prefill（no\-domain）+decode（domain）精度验证，基于430配置  志祥

3. prefill（domain）+deocde（no\-domain）收益验证  魏桂华

4. prefill（no\-domain）+deocde（domain）收益验证   王小超

5. prefill（domain）+decode（domain）精度和收益验证

6. 性能叠加：chunkPrefill （魏桂华正在搞），DCP（志祥试一下），MTP，异步调度（当前已支持），细粒度混合并行，prefix cache（后期再搞）

7. 代码重构

8. 430上灰度测试，先用deepseek3.1测试，配置：prefill tp8dp2+domain（直接删除max\-model\-len试一下能不能跑下），或者tp8dp4；decode EP32+domain16 2个domain域

9. 后期要支持kimi2.5模型

本周：

图模式+chunk

基于430配置精度验证正常

采出性能基线

# 问题定位

## 问题1：kv cache拉取循环等待

【20260414问题】跑AISBench会有几个request一直在等待kv cache，如下图：

![](https://clouddocs.huawei.com/koopage/v1/app/api/documents/doc/preview/75091e21-20f9-4c61-9334-04145e4d2b2b?document_id=4b6e1b80-7dab-4679-ae88-d50ea0559a22 "")

【问题定位】

从step\_domain函数可以看出，如果execute\_model的执行结果包含None，就会执行sample\_token函数，并且以sample\_tokens函数的model\_outputs更新update\_from\_output函数execute\_model的执行结果就会被覆盖。

![](https://clouddocs.huawei.com/koopage/v1/app/api/documents/doc/preview/0e112a8c-e263-4398-adc6-63ed5f055abf?document_id=4b6e1b80-7dab-4679-ae88-d50ea0559a22 "")

原来step正常的写法是：

![](https://clouddocs.huawei.com/koopage/v1/app/api/documents/doc/preview/a8fe636a-361d-4699-b165-9f554dd4069d?document_id=4b6e1b80-7dab-4679-ae88-d50ea0559a22 "")

考虑一种场景：某次调度时，第一个DP rank上有未参与切分的数据需要计算，第二个DP rank上没有数据参与计算，但有kv cache需要传输。

此时scheduler会输出两个schedule\_output。两个scheduler\_output分别进入到model\_runner\_v1/execute\_model，第一个会输出None，表示接下来需要调度sample\_tokens函数处理logits，输出一个token；第二个会输出ModelRunnerOutput，表示存在kv cache传输。model\_runner的输出结果经过aggregate函数集合后，发现存在kv cache传输finish，会将finished\_set置位并且随着model\_output传输出去（如下图）；

![](https://clouddocs.huawei.com/koopage/v1/app/api/documents/doc/preview/9e9f9093-0ba2-4514-a97c-40da05851de7?document_id=4b6e1b80-7dab-4679-ae88-d50ea0559a22 "image.png")

两个结果混合起来，走到step\_domain函数的1119行，发现结果中存在None，sample\_tokens会重新生成model\_output，并将之前结果覆盖，而之前结果里的finished\_set有用信息也就被覆盖了，无法传输到update\_from\_output函数中，进而导致某个request的kv cache一直无法传输完成。

【代码修改】

异步调度场景修改结果如下，sample\_tokens的计算结果要有条件的覆盖execute\_model的结果

![](https://clouddocs.huawei.com/koopage/v1/app/api/documents/doc/preview/8bdbe33e-f45c-4283-9425-1765f1028149?document_id=4b6e1b80-7dab-4679-ae88-d50ea0559a22 "")

非异步调度场景修改结果如下，将execute\_model的结果与sample\_tokens的结果融合，而不是之前的sample\_tokens直接覆盖execute\_model结果。

![](https://clouddocs.huawei.com/koopage/v1/app/api/documents/doc/preview/c436c7bb-a56c-4f87-9a06-7c6426b6475e?document_id=4b6e1b80-7dab-4679-ae88-d50ea0559a22 "")

## 问题2：request释放kv cache时触发assert

【20260414问题】跑AISBench多轮后，\_update\_from\_kv\_xfer\_finished函数中会报assert req\_id in self.requests，如下图位置

![](https://clouddocs.huawei.com/koopage/v1/app/api/documents/doc/preview/13129f9e-a7e2-44f6-aac7-f769329ee062?document_id=4b6e1b80-7dab-4679-ae88-d50ea0559a22 "")

【问题定位】

scheduler中输出为cp\_size个output，在mooncake的build\_connector\_meta中，输入为单个dp的scheduler\_output，希望基于这个整理单个dp对应的meta信息，但基于的3个self变量都是包含了所有dp结果，导致整理出来的每个dp的meta信息包含了所有dp信息。引发的问题举例如下：

例如某个request i只执行在dp1上，其信息也会被包含在dp0上，之后在kv cache传输中，由于每个dp0的meta信息包含了所有dp的信息，包括request i，但其实request i没有在本dp rank上执行，但本dp rank也会对其拉kv cache做检测，最后因为超时导致强制free，但这个request i已经被dp1 free过了，就产生了多次free导致的assert报错。

![](https://clouddocs.huawei.com/koopage/v1/app/api/documents/doc/preview/8df51213-31bf-4ded-9332-a3e79e085c46?document_id=4b6e1b80-7dab-4679-ae88-d50ea0559a22 "")

【代码修改】

在scheduler中整理每个cp rank会处理哪些request：scheduler\_output.cp\_rank\_to\_req\_id，在mooncake中针对每个cp rank构建metadata时，过滤出只属于本cp rank的request。

![](https://clouddocs.huawei.com/koopage/v1/app/api/documents/doc/preview/bc1c3dc6-1738-4ffd-a878-3f9a14f309a3?document_id=4b6e1b80-7dab-4679-ae88-d50ea0559a22 "")

## 问题3：kv cache拉取循环等待

【20260414问题】跑AISBench多轮后，会出现某个request一直处于waitting for kv cache的状态

![](https://clouddocs.huawei.com/koopage/v1/app/api/documents/doc/preview/75091e21-20f9-4c61-9334-04145e4d2b2b?document_id=4b6e1b80-7dab-4679-ae88-d50ea0559a22 "")

【问题定位】

分析发现是dp0的8张卡全部完成传输，dp1的8张卡走进了下图的logger.error，进一步分析是dp1的后8张卡self.reqs\_to\_process中没有包含request\_id。即这个request已经被处理完了，但初始的self.reqs\_to\_process竟然还没有包含这个request。

![](https://clouddocs.huawei.com/koopage/v1/app/api/documents/doc/preview/2e683459-41f8-4879-bb88-5de4ef285548?document_id=4b6e1b80-7dab-4679-ae88-d50ea0559a22 "")

分析代码发现，start\_load\_kv函数中，request的kv cache拉取位于了add\_req\_to\_process前边，worker主线程在执行start\_load\_kv函数，接收线程执行kv cache拉取过程，理想情况下，拉取kv cache需要一定时间，这个时间足够worker线程完成add\_req\_to\_process操作。但在domain场景下，有些D节点的卡由于分不到kv cache，其不用拉取kv cache，因此拉取的过程直接空跑很快走到上图update\_done\_task\_count位置，但此时add\_req\_to\_process还没有完成，就导致了报错。需要调整下图代码函数位置。

![](https://clouddocs.huawei.com/koopage/v1/app/api/documents/doc/preview/5312beb9-6eb8-4e14-9613-d44a9b4bf4cb?document_id=4b6e1b80-7dab-4679-ae88-d50ea0559a22 "")

【代码修改】

代码修改后，如下图所示

![](https://clouddocs.huawei.com/koopage/v1/app/api/documents/doc/preview/3a17a154-3623-49b2-b222-5807d932d33d?document_id=4b6e1b80-7dab-4679-ae88-d50ea0559a22 "")

## 问题4：P和D都开domain精度问题

【20260418问题】P节点和D节点都开domain，跑100条gsm8k数据，评分为91分，比正常97分偏低，存在精度问题。

【问题定位】

最开始在100条数据集中测试发现某些推理结果是乱码，评分也比正常的要低，判断有精度问题，采取了如下分析步骤：

1. 尝试压测不同配置：只开P节点的domain，发现精度正常；只开D节点的domain，精度不正常；P和D都开domain，精度不正常。出问题的场景锁定在D侧；

2. 尝试不同输入数据类型：全部输入短数据集，精度正常；全部输入长数据集，精度正常，混合长短，精度异常。出问题的数据集锁定在长短混合场景；

3. 尝试压缩数据集大小浮现问题：最开始的数据集是100条报精度问题，大规模数据集场景很难通过打印或者dump数据定位问题，需要压缩复现规模；尝试了单条测试，问题没有复现；尝试把存在精度问题的几条数据整理出来单独测试，问题也没有复现；多次尝试后，发现数据样本可以缩小到20条，长短混合比例为7：3，如此就可以通过打印定位问题了；

4. 打印了长短混合下的PD分离的P节点和D节点的mapping建链结果，未发现异常，排除了PD分离；

5. 由于是长短混合场景才会报错，怀疑是模型后端对长短混合拆分时排序出了问题，尤其是slotmapping计算错误，在代码中增加了打印。结果如下图，slotmapping已经经过了长短序列排序，长序列排在前边，所有CP上均有值，但只有一个CP上有正常slot值，其他CP应该是\-1；短序列排在后边，只有一个CP上有值。下图发现存在CP切分两个rank上都有slot值的现象，怀疑是长短排序出了问题。

![](https://clouddocs.huawei.com/koopage/v1/app/api/documents/doc/preview/dcdbcc5d-82ca-4288-b92c-e1f80a54d255?document_id=4b6e1b80-7dab-4679-ae88-d50ea0559a22 "")

6. 进一步打印了input\_batch中req\_ids，发现确实顺序有错乱。进一步定位发现是排序之前不同rank上的CP排序不是按照相同顺序，而是有错位，让大模型检查了代码，给出了如下解释，可以发现确实input\_batch中request的顺序是存在不确定性的，虽然attention后端可以区分出长短，但长序列中顺序会乱，需要针对长序列再排一次顺序。

🎯



让我追踪 `self.input_batch._req_ids` 从初始赋值到进入 *`may`*`reorder_batch` 的完整排序变化过程。

`self.input_batch._req_ids` 排序变化全链路分析

以下按 *`update`*`states` 方法中的**严格执行顺序**梳理：

***

第1步：移除已完成请求（`gpu_model_runner.py:910-911`）

```python
for req_id in scheduler_output.finished_req_ids:
    self.input_batch.remove_request(req_id)
```

**影响**：`remove_request` 将对应位置的 *`req`*`ids[index]` 设为 `None`，产生**空洞**。列表长度不变。

例：`["A", "B", "C", "D"]` → B 完成 → `["A", None, "C", "D"]`

***

第2步：移除未调度请求（`gpu_model_runner.py:936-937`）

```python
for req_id in unscheduled_req_ids:
    self.input_batch.remove_request(req_id)
```

**影响**：同上，继续产生空洞。

例：`["A", None, "C", "D"]` → D 未调度 → `["A", None, "C", None]`

***

第3步：添加新/恢复请求（`gpu_model_runner.py:1126-1127`）

```python
for request in reqs_to_add:
    self.input_batch.add_request(request)
```

**影响**：`add_request` 调用 *`register`*`add_request`，通过 `pop_removed()` 获取**最小的空洞索引**优先填入，无空洞则追加到末尾。

例：`["A", None, "C", None]` → 添加 E → `["A", "E", "C", None]`（填入索引1）

***

第4步：condense 压缩（`gpu_model_runner.py:1131`）

```python
self.input_batch.condense()
```

**影响**：双指针算法——从尾部找活跃请求，从头部找空洞，将尾部请求搬到空洞位置，最后截断列表。**这会改变部分请求的相对顺序**。

例：`["A", "E", "C", None]` → condense → `["A", "E", "C"]`（截断尾部 None）

更复杂的例子：`["A", None, None, "D", "E"]` → condense → `["A", "E", "D"]`（E 从索引4搬到索引1，D 从索引3搬到索引2）

***

第5步：进入 *`may`*`reorder_batch`（`gpu_model_runner.py:1133`）

此时 *`req`*`ids` 已经是**紧凑的、无 None 的列表**，但顺序可能因 condense 而被打乱。

5a. `reorder_batch_to_split_decodes_and_prefills`

通过 `swap_states` 将 batch 重排为 **\[decode, extend, prefill\]** 顺序。

5b. `reorder_batch_to_split_cp_and_normal`

通过 `apply_permutation` 将 CP 请求移到前面：**\[cp请求\(按req\_id排序\), 非cp请求\]**。

***

总结流程图

```python
初始状态: ["A", "B", "C", "D", "E"]
    │
    ▼ remove_request (完成/未调度)
["A", None, "C", None, "E"]     ← 产生空洞
    │
    ▼ add_request (新/恢复请求)
["A", "F", "C", None, "E"]     ← 填入最小空洞
    │
    ▼ condense()
["A", "F", "C", "E"]           ← 尾部搬到空洞，截断
    │
    ▼ _may_reorder_batch()
      ├─ split_decodes_and_prefills → [decode, extend, prefill]
      └─ split_cp_and_normal       → [cp请求, 非cp请求]
["E", "A", "F", "C"]           ← 最终顺序
```

**关键点**：进入 *`may`*`reorder_batch` 时，`_req_ids` 的顺序已经不是 scheduler 输出的原始顺序了，而是经过 remove → add → condense 三步变换后的结果。condense 是最容易被忽略的排序变化来源。


【代码修改】

代码修改后，如下图所示，基于request\_id字符串先排一次序，保证所有长序列的顺序也一致，这样slotmaping的乱序问题也随之解决。

![](https://clouddocs.huawei.com/koopage/v1/app/api/documents/doc/preview/76c8c747-3eec-4809-8e1f-8727208f2754?document_id=4b6e1b80-7dab-4679-ae88-d50ea0559a22 "")

之后测试精度正常：

![](https://clouddocs.huawei.com/koopage/v1/app/api/documents/doc/preview/86861c74-3511-4ef3-9587-c5c06b51287e?document_id=4b6e1b80-7dab-4679-ae88-d50ea0559a22 "")

# 遗留问题

1. 基线DP性能不如开domain的短序列走DP性能，可能会是一个收益点，需要分析engincore的调度逻辑

2. 4K场景，tp比pcp性能差很多，需要分析原因（tongyuzhou，跟桂华要脚本，采一把profiling）

3. 4K场景，tp16比dp2tp8性能差，需要分析原因（主要是因为tp16的通信耗时大于dp2tp8）

4. request rate很小的时候，吞吐一样，但ttft差异很大，猜测是大家都没有打满，吞吐取决于来包量，但ttft可以展示系统小batch下真实处理能力，收益有20%；当request\_rate很大时，ttft没有多大意义，但吞吐可以体现系统真实处理能力，发现收益也是20%；

5. free\-block问题+分桶受到未释放block影响（一直会分桶到dp0）

6. 极端场景：每次只调度一条短序列，一定会被分配到dp0上，导致dp0上的显存压力大，想想怎么解决；

7. 当前execute\_model的异步调度是写死为True的

8. 阈值做成外部可配置  — 已完成
