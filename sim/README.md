# 验证实验代码

复现并检验 Luo2024 / Zhang2026 两篇论文的关键声明，并给出所推荐创新点的原型。

```
nf_model.py        近场宽带模型：精确球面波 vs Fresnel 近似、TTD/PS 波束斜视轨迹
estimators.py      Zhang2026 Algorithm 1（几何补偿平滑 + 局部 MUSIC）；四种 CRLB
proposed_sfj.py    推荐方案原型：相干空-频（时延域）联合定位

exp_a_inverse_crime.py   实验 A：Fig. 13 的 inverse crime
exp_b_trajectory.py      实验 B：一维轨迹无法覆盖二维区域 + 距离轨迹非单调
exp_c_threshold.py       实验 C：Fig. 10 缺失的门限效应
exp_d_complexity.py      实验 D：宣称精度与宣称时延的矛盾
```

运行：`pip install numpy scipy matplotlib && cd sim && python3 exp_b_trajectory.py`

全部实验均使用 Zhang2026 Table II 的默认配置
（fc=60 GHz, N=256, W=3 GHz, M=2048, d=λ/2, Ms=128, P=129, 感知区 ±60°/15–50 m）。
SNR 一律定义为**每阵元** |α|²/σ²，四种 CRLB 使用同一定义，故比值与定义无关。
