"""SYNASPOT 超参数配置.

论文 3.2 Setup:
- 音频编码器 L=7 层 DFSMN, hidden=256
- 50 epochs, Adam lr=1e-3, annealing factor 0.3, batch size 100
- (alpha_A, beta_A) = (0.5, 0.5);  (alpha_M, beta_M, gamma_M) = (0.4, 0.3, 0.3)
- 推理分数融合 (alpha_S, beta_S, gamma_S) = (0.5, 0.25, 0.25)
"""
from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    # ---- 特征 ----
    sample_rate: int = 16000
    frame_shift_ms: int = 10
    frame_length_ms: int = 25
    num_fbanks: int = 80
    # fbank 后拼接一/二阶差分 (80 -> 240 维). 声母为瞬态音(爆破/擦音/塞擦),
    # 判别信息集中在频谱变化率里, 静态 fbank 表达弱; 差分可显著提升帧级声母分类.
    # 注意: 标准 ±2 帧差分窗口引入 40ms 未来, 对 KWS 延迟可接受; 需严格因果可改用左移差分.
    use_delta: bool = False

    # ---- 统一特征帧率 (实验开关, 训练之初配置) ----
    # 所有音频主干的输出与帧级标签都对齐到该帧率. 帧标签恒以 100Hz 存储
    # (04_build_dataset.py), 训练/解码时按 100/feat_frame_rate 降采样对齐.
    # 25/50 用于帧率消融实验; 100 = 论文 fbank 基线(dfsmn 原始帧率).
    feat_frame_rate: int = 50

    # ---- 音素粒度 (实验开关, 训练之初配置) ----
    # "initial":       仅声母 (~24 类, 零声母归 <O>)
    # "syllable":      声母+韵母合并的去调音节 (~420 类)
    # "syllable_tone": 声母+韵母+声调 (~1160 类)
    # 实际词表由数据构建 (04_build_dataset.py --granularity) 决定并写入
    # dataset_info.json; 此字段供训练/解码显式读取, 替代脆弱的词表启发式.
    phoneme_granularity: str = "initial"

    # ---- 音频主干选择 ----
    # "dfsmn":     原版, 输入 80 维 fbank(100Hz), 从零训练(小模型在全量多说话人易塌缩).
    # "ssl":       输入离线缓存的预训练 SSL 特征(HuBERT, 768 维, 50Hz), 冻结 SSL 只训
    #              适配器, 特征已含强判别力与时序上下文, 直接规避从零塌缩.
    audio_backbone: str = "dfsmn"
    ssl_feat_dim: int = 768      # HuBERT/wav2vec2-base 隐层维度
    ssl_downsample: int = 2      # [遗留] 旧 ssl 标签降采样倍数; 新代码统一用 label_ds

    # ---- 离线→在线蒸馏 (教师/学生可用不同语音编码器) ----
    # 教师 = 离线强编码器(冻结, 不蒸馏, 仅作监督源); 学生 = 在线因果编码器(待蒸馏, 部署).
    # 流程: 先用 distill_teacher_backbone 主干训好一个完整模型并存为教师检查点;
    #       蒸馏训练时学生用 audio_backbone(因果)主干, 同时载入教师检查点与教师离线特征,
    #       以 E_A 嵌入级(逐帧余弦) + Logit 级 KD(音素软标签)联合把教师的音素知识迁给学生.
    # 约束: 教师与学生的 feat_frame_rate / hidden_dim / vocab_size(音素粒度)必须一致,
    #       否则帧级对齐与嵌入/词表维度不匹配, load_teacher 会显式报错.
    distill_enabled: bool = False
    distill_teacher_ckpt: str = ""         # 教师检查点路径(完整模型, 含 config)
    distill_teacher_backbone: str = "ssl"  # 教师主干=离线特征来源(须与教师训练时一致)
    distill_w_embed: float = 1.0           # E_A 嵌入级蒸馏权重(逐帧余弦, 全部有效帧)
    distill_w_logit: float = 1.0           # Logit 级 KD 权重(音素软标签, 内容帧)
    distill_temperature: float = 2.0       # KD 温度

    # ---- 音频编码器 (DFSMN) ----
    num_layers: int = 7          # L
    hidden_dim: int = 256
    proj_dim: int = 128          # DFSMN 层内瓶颈维度
    lorder: int = 20             # memory 回看帧数
    rorder: int = 0              # memory 前看帧数, 0=因果(支持流式)
    dropout: float = 0.1

    # ---- 分类头 ----
    vocab_size: int = 27         # 声母词表(含 <PAD>/<UNK>/<SIL>/<O>, ~24声母); 音节为420
    num_speakers: int = 340      # train 说话人数(梯度反转域自适应)

    # ---- AAM-softmax (音素分类, Function 1) ----
    aam_scale: float = 30.0      # s
    aam_margin: float = 0.2      # m

    # ---- 文本分支 ----
    text_embed_dim: int = 256
    text_lstm_hidden: int = 256
    text_lstm_layers: int = 1
    # 文本注册时每个拼音 token 展开的帧数(与训练时帧级展开保持一致)
    enroll_expand_frames: int = 10

    # ---- 混合嵌入 cross-attention ----
    num_heads: int = 4

    # ---- 对比学习 ----
    temperature: float = 0.1     # τ
    cl_max_frames: int = 4096    # 一个 batch 参与对比学习的最大帧数(显存控制)

    # ---- text-only 对齐阶段 ----
    # 共享音素分类器辅助损失权重: 0=纯逐帧余弦对齐(推荐起点);
    # >0 时叠加"用冻结音素头分类 e_t"的蒸馏式对齐, 信号更强但更复杂.
    align_cls_w: float = 0.0

    # ---- 损失权重 ----
    alpha_A: float = 0.5
    beta_A: float = 0.5
    alpha_M: float = 0.4
    beta_M: float = 0.3
    gamma_M: float = 0.3

    # ---- 推理分数融合 ----
    # 默认三模态等权 1:1:1 (融合时按权重和归一化). SSL 50Hz 实测等权与最优 1:1:0 持平,
    # 且优于论文 0.5:0.25:0.25 (TA WP_E EER 3.91% vs 4.09%), 故偏离论文默认.
    alpha_S: float = 1.0
    beta_S: float = 1.0
    gamma_S: float = 1.0

    # ---- 流式解码 ----
    w_smooth: int = 10           # 因果滑动平均窗口 (Function 6)
    w_scoring: int = 50          # 打分窗口 (Function 7)
    score_skip_last: bool = True # Function 7 中连乘取前 n-1 个单元

    # ---- 派生量 (由实验开关推导, 非独立超参) ----
    @property
    def label_ds(self) -> int:
        """100Hz 帧标签降采样到 feat_frame_rate 的倍数 (50Hz->2, 25Hz->4)."""
        return max(1, 100 // self.feat_frame_rate)

    @property
    def backbone_native_hz(self) -> int:
        """各音频主干的原始输出帧率 (重采样到 feat_frame_rate 之前)."""
        return {"dfsmn": 100, "ssl": 50}[self.audio_backbone]

    @property
    def feat_dim(self) -> int:
        """音频主干输入到 DFSMN 的特征维度."""
        if self.audio_backbone == "dfsmn":
            return self.num_fbanks * (3 if self.use_delta else 1)
        return self.ssl_feat_dim

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        valid = {f for f in ModelConfig.__dataclass_fields__}
        return ModelConfig(**{k: v for k, v in d.items() if k in valid})
