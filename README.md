# GTO Poker Calculator

一个基于 GTO（Game Theory Optimal）理论的德州扑克计算器，提供翻前范围、翻后策略建议、胜率计算和对手范围推断等功能。

![screenshot](frontend/assets/screenshot.png)

## 功能

- **翻前策略** — 根据位置和行动历史给出 开牌/跟注/3-bet/4-bet 建议
- **翻后策略** — 基于相对强度、底池赔率、MDF 的数学驱动建议（下注/跟注/弃牌/加注频率）
- **手牌分类** — 顶 set / 中 set / 底 set、nut straight / weak straight、顶两对 / 底两对、flush 质量等精细分类
- **摸牌检测** — 同花听牌（nuts / non-nut）、OESD、gutshot（含 broadway 单侧修正）、backdoor draws、combo draw
- **Villain 范围矩阵** — 13×13 可视化对手应对频率（加注/跟注/弃牌）
- **多人底池调整** — 2-8 人底池的诈唬/价值频率自动修正
- **胜率计算** — 支持多人底池的蒙特卡洛/枚举胜率
- **翻牌纹理感知** — 干燥/湿润/单色牌面识别，turn 换牌纹理变化检测
- **3-bet 底池识别** — 自动检测 3-bet 底池，使用更大尺度
- **SPR 感知** — 短筹码 commit-or-fold，深筹码慢打

## 技术栈

| 层 | 技术 |
|---|---|
| 后端 | Python · FastAPI · Uvicorn |
| 前端 | 原生 HTML / CSS / JavaScript |
| 核心逻辑 | 纯 Python（无外部 solver 依赖） |

## 快速开始

**环境要求：** Python 3.10+

```bash
# 克隆项目
git clone https://github.com/yli40700-star/gto-poker-calculator.git
cd gto-poker-calculator

# 安装依赖
pip install -r requirements.txt

# 启动服务
python run.py
```

浏览器打开 **http://localhost:8000**

> 如果端口被占用：`lsof -ti:8000 | xargs kill -9`

## 项目结构

```
gto-poker-calculator/
├── backend/
│   ├── api/
│   │   └── server.py        # FastAPI 路由
│   └── core/
│       ├── advisor.py       # GTO 核心决策引擎
│       ├── reasoning.py     # 动态推理 + Villain 矩阵
│       ├── ranges.py        # Villain 范围推断
│       ├── evaluator.py     # 手牌评估器
│       └── cards.py         # 牌的解析
├── frontend/
│   ├── index.html
│   ├── js/app.js            # 前端状态机
│   └── css/style.css
├── requirements.txt
└── run.py
```

## API 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/advisor` | 主要建议（翻前 + 翻后） |
| POST | `/api/equity` | 多人底池胜率计算 |
| GET | `/api/preflop/{position}` | 13×13 翻前范围矩阵 |

## 核心参数说明

### `POST /api/advisor`

```json
{
  "hero_hand": "AhKd",
  "hero_position": "BTN",
  "villain_position": "BB",
  "board": "As Qc 7h",
  "pot_size": 30,
  "stack_size": 200,
  "action_history": ["BTN:raise_3", "BB:call", "check"],
  "street": "flop",
  "num_villains": 1
}
```

### 返回字段

| 字段 | 说明 |
|---|---|
| `recommended_actions` | `{action: frequency}` 字典，频率之和为 1 |
| `hand_analysis` | 手牌分类、摸牌检测、相对强度 |
| `board_analysis` | 牌面湿润度、是否配对、turn 换牌类型 |
| `spr` | 筹码底池比 |
| `is_3bet_pot` | 是否 3-bet 底池 |
| `facing_bet` | 是否正面对抗下注 |
| `gto_reasoning` | 详细 GTO 推理 + Villain 矩阵 |

## 相对强度参考

| 手牌 | `relative_strength` |
|---|---|
| Straight flush | 1.00 |
| Quads | 0.99 |
| Top full house | 0.95 |
| Bottom full house | 0.87 |
| Nut flush | 0.95 |
| Top set | 0.94 |
| Middle set | 0.91 |
| Bottom set | 0.88 |
| Nut straight | 0.88 |
| Flush (Q+) | 0.90 |
| Straight | 0.80–0.84 |
| Weak straight / wheel | 0.76 |
| Two pair (top) | 0.78 |
| Two pair (top-bottom) | 0.70 |
| Overpair | 0.65–0.70 |
| TPTK | 0.68 |
| TP good kicker | 0.53–0.59 |
| TP weak kicker | 0.48 |
| Second pair | 0.40 |
| Bottom pair | 0.28 |
| Underpair | 0.25 |
| Ace high | 0.20 |
| Air | 0.10 |

## License

MIT
