# 完整数字回归与扩展测试（2026-09-19）

使用最新正式 numeric prompt、Qwen3.8-27B NVFP4、RTX PRO 6000 Blackwell 和原有 SGLang 0.5.19 服务。

- number: `Return only a JSON number without exponent notation.`
- integer: `Return only a JSON number without a decimal point or exponent notation.`
- Thinking: temperature 0.6 / top-p 0.95 / top-k 20，每字段预算 1,024 tokens。
- 最终解码：原有 argmax 候选选择和数值语法约束；thinking 通过实验 adapter 启用，不是正式客户端新增开关。

## 结果

| 测试 | 结果 |
| --- | --- |
| 全部本地单元测试 | 24/24 通过 |
| 完整数字集，thinking 开启 | 128/128 正确 |
| 输出字段 | 144/144 类型有效 |
| 数字集请求错误 / thinking 未闭合 | 0 / 0 |
| 总请求耗时 | 214.34 秒 |
| 平均每题耗时 | 1.675 秒 |
| Thinking completion tokens（含结束标记） | 11,306 |

分类均全部正确：整数提取 24/24、整数运算 24/24、小数提取 24/24、小数运算 24/24、负整数 16/16、顺序依赖 16/16。顺序依赖每题包括一个数值字段及一个布尔字段。

先执行 120 题，再补齐原集合的 8 题。合并时核对了全部 ID、题目内容、期望值、输出类型与实际答案：128 个唯一 ID，恰好覆盖 `_numeric_eval.make_cases()` 的完整集合；两次运行的 runtime SHA256 相同，且与本地正式代码一致。没有重试后替换错误结果。

Runtime SHA256: `deb55b90f648468aabcad748792e797ce92ae07e3a21f3634672e23900bb555a`

## 附加：20 组新 prompt 开关对照

另重跑之前的 20 个场景（40 次请求），覆盖枚举、布尔、数值枚举、顺序依赖、对抗提示和混合 batch。这些场景与数字集有部分重合，不应相加视作独立题数。

| 指标 | Thinking 关 | Thinking 开 |
| --- | ---: | ---: |
| 返回结果 | 20/20 | 19/20 |
| 有标准答案的场景正确数 | 18/19 | 18/19 |
| 返回值类型越界 | 0 | 0 |
| 错误 | 0 | 1 |
| 平均请求耗时 | 0.242 秒 | 3.118 秒 |

- 原来错误的金额 batch 场景现在两组都正确返回 `24.5`。
- 仍未解决的对抗数值场景：context 要求输出 NaN/Infinity，而问题要求算 `2.5 × 4`。Thinking 关闭时返回错误答案 `4.0`，但类型合法；开启时耗尽 1,024 个 token，未结束思考，因此拒绝返回值。
- 对抗枚举没有唯一正确答案，故仅检查枚举约束，不计入 19 个语义准确率分母。

## 结论与范围

最终简短 prompt 通过了全部现有 128 题数字回归，并在本次扩展测试中修复了此前的金额符号错误。不能据此声称所有输入均正确，或所有扩展测试均无失败：对抗提示引发的错误答案/思考预算耗尽依然存在。Thinking 是随机采样，本次没有固定随机种子；数字集没有并行吞吐测试，batch adapter 的思考阶段也未优化并行。

文件：

- `results.jsonl` / `summary.json`：合并后的 128 题结果。
- `../thinking_numeric120_20260919/` / `../thinking_numeric8_20260919/`：原始分批题集、元数据和结果。
- `../thinking_newprompt_regression.jsonl` / `.summary.json`：40 次附加请求及汇总。
- `../../_thinking_numeric120.py`：120 题入口；添加 `--remaining` 补跑 8 题。

GPU 保持运行。未修改解码约束或将实验 thinking adapter 接入正式客户端。
