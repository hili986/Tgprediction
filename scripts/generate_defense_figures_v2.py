"""生成 SITP 升级答辩配图 v2 (共 6 张，高抽象层次)"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

plt.rcParams.update({
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Arial'],
    'axes.unicode_minus': False,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
})

OUT = r"C:\Users\24020\Desktop\Tg预测项目\答辩图片"
os.makedirs(OUT, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# 配图 1: 技术路线流程图 (A→B→C→D)
# ═══════════════════════════════════════════════════════════
def fig01_roadmap():
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    phases = [
        (0.3,  '#E3F2FD', '#1565C0', 'A', '理论与语法建模',
         ['文献调研 (25+ 篇)', 'BigSMILES 语法梳理', '示例复现与手册'], 'DONE'),
        (3.5,  '#E8F5E9', '#2E7D32', 'B', '工具开发与实现',
         ['BigSMILES 解析器', '生成器 + 规范化', '语法校验'], 'DONE'),
        (6.7,  '#FFF3E0', '#E65100', 'C', '数据集与 ML 预测',
         ['Tg 数据集 7,486 条', '特征工程 + VPD', '模型对比 + 消融'], 'DONE'),
        (9.9,  '#F3E5F5', '#6A1B9A', 'D', '扩展探索',
         ['多尺度物理特征', '共聚物 Tg 预测', '核酸高分子 Tg'], 'WIP'),
    ]

    for x, bg, border, letter, title, items, status in phases:
        w, h = 3.0, 4.0
        rect = mpatches.FancyBboxPatch((x, 0.6), w, h, boxstyle="round,pad=0.15",
                                        facecolor=bg, edgecolor=border, lw=2)
        ax.add_patch(rect)

        # Phase letter circle
        circle = plt.Circle((x + w/2, 4.1), 0.35, color=border, zorder=3)
        ax.add_patch(circle)
        ax.text(x + w/2, 4.1, letter, ha='center', va='center',
                fontsize=14, fontweight='bold', color='white', zorder=4)

        # Title
        ax.text(x + w/2, 3.4, title, ha='center', va='center',
                fontsize=11, fontweight='bold', color=border)

        # Items
        for i, item in enumerate(items):
            ax.text(x + 0.3, 2.6 - i * 0.55, f'  {item}',
                    fontsize=9, color='#444', va='center')

        # Status badge
        sc = '#2E7D32' if status == 'DONE' else '#E65100'
        badge_text = 'DONE' if status == 'DONE' else 'WIP'
        ax.text(x + w/2, 0.9, badge_text, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.25', facecolor=sc, edgecolor='none'))

    # Arrows between phases
    for x in [3.3, 6.5, 9.7]:
        ax.annotate('', xy=(x + 0.2, 2.6), xytext=(x, 2.6),
                    arrowprops=dict(arrowstyle='->', color='#666', lw=2.5))

    # Title
    ax.text(7, 5.2, '技术路线总览', ha='center', fontsize=16, fontweight='bold')

    plt.savefig(os.path.join(OUT, '配图1_技术路线.png'))
    plt.close()
    print('[OK] 配图1 技术路线')


# ═══════════════════════════════════════════════════════════
# 配图 2: 数据集建设 (304 → 7,486)
# ═══════════════════════════════════════════════════════════
def fig02_dataset():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.axis('off')

    ax.text(5.5, 4.6, '数据集建设历程', ha='center', fontsize=16, fontweight='bold')

    # Phase 1: Bicerano
    r1 = mpatches.FancyBboxPatch((0.3, 1.5), 3.2, 2.5, boxstyle="round,pad=0.15",
                                  facecolor='#E3F2FD', edgecolor='#1565C0', lw=2)
    ax.add_patch(r1)
    ax.text(1.9, 3.6, 'Bicerano', ha='center', fontsize=13, fontweight='bold', color='#1565C0')
    ax.text(1.9, 3.0, '304 条', ha='center', fontsize=20, fontweight='bold', color='#1565C0')
    ax.text(1.9, 2.3, '线型均聚物', ha='center', fontsize=10, color='#555')
    ax.text(1.9, 1.85, 'BigSMILES 标注', ha='center', fontsize=9, color='#888')

    # Arrow
    ax.annotate('', xy=(4.2, 2.75), xytext=(3.6, 2.75),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2.5))
    ax.text(3.9, 3.2, '+6 个\n数据库', ha='center', fontsize=9, color='#E65100', fontweight='bold')

    # Phase 2: Unified
    r2 = mpatches.FancyBboxPatch((4.3, 1.5), 3.2, 2.5, boxstyle="round,pad=0.15",
                                  facecolor='#E8F5E9', edgecolor='#2E7D32', lw=2)
    ax.add_patch(r2)
    ax.text(5.9, 3.6, '统一数据集', ha='center', fontsize=13, fontweight='bold', color='#2E7D32')
    ax.text(5.9, 3.0, '7,486 条', ha='center', fontsize=20, fontweight='bold', color='#2E7D32')
    ax.text(5.9, 2.3, '去重 + SMILES 规范化', ha='center', fontsize=10, color='#555')
    ax.text(5.9, 1.85, '+ 异常值检测', ha='center', fontsize=9, color='#888')

    # Arrow
    ax.annotate('', xy=(8.2, 2.75), xytext=(7.6, 2.75),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2.5))
    ax.text(7.9, 3.2, '+共聚物\n数据', ha='center', fontsize=9, color='#E65100', fontweight='bold')

    # Phase 3: future
    r3 = mpatches.FancyBboxPatch((8.3, 1.5), 2.4, 2.5, boxstyle="round,pad=0.15",
                                  facecolor='#F3E5F5', edgecolor='#6A1B9A', lw=2, ls='--')
    ax.add_patch(r3)
    ax.text(9.5, 3.6, '扩展中', ha='center', fontsize=13, fontweight='bold', color='#6A1B9A')
    ax.text(9.5, 3.0, '?', ha='center', fontsize=20, fontweight='bold', color='#6A1B9A')
    ax.text(9.5, 2.3, '共聚物 + 核酸', ha='center', fontsize=10, color='#555')

    # Data sources at bottom
    sources = ['Bicerano', 'PolyMetriX', 'NeurIPS OPP', 'Pilania PHA', '...']
    ax.text(5.5, 0.9, '数据来源:  ' + '  |  '.join(sources),
            ha='center', fontsize=10, color='#777',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FAFAFA', edgecolor='#DDD'))

    plt.savefig(os.path.join(OUT, '配图2_数据集建设.png'))
    plt.close()
    print('[OK] 配图2 数据集建设')


# ═══════════════════════════════════════════════════════════
# 配图 3: 特征工程金字塔
# ═══════════════════════════════════════════════════════════
def fig03_feature_pyramid():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)
    ax.axis('off')

    ax.text(5, 6.2, '多层级特征工程', ha='center', fontsize=16, fontweight='bold')

    # Three layers (bottom to top)
    layers = [
        (1.0, 0.5, 8.0, 1.4, '#E3F2FD', '#1565C0',
         '基础物理量 (4d)', '分子柔性  |  溶解度  |  分子量  |  密度'),
        (1.8, 2.2, 6.4, 1.4, '#E8F5E9', '#2E7D32',
         '2D 分子描述符 (30d)', '拓扑指数  |  极性面积  |  氢键供体/受体  |  环数  |  ...'),
        (2.6, 3.9, 4.8, 1.4, '#FFF3E0', '#E65100',
         'VPD 虚拟聚合描述符 (12d)', '聚合效应  |  连接柔性  |  极性变化  |  环数变化'),
    ]

    for x, y, w, h, bg, border, title, detail in layers:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                        facecolor=bg, edgecolor=border, lw=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.35, title,
                ha='center', fontsize=12, fontweight='bold', color=border)
        ax.text(x + w/2, y + 0.35, detail,
                ha='center', fontsize=9, color='#555')

    # Star on VPD
    ax.text(7.6, 5.0, '  Core\nInnovation', ha='center', fontsize=10,
            fontweight='bold', color='#C62828',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#EF9A9A'))

    # Total
    ax.text(5, 0.1, '  Total: 46 dim  =  M2M-V feature set  ',
            ha='center', fontsize=11, fontweight='bold', color='#333',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFDE7', edgecolor='#F9A825'))

    plt.savefig(os.path.join(OUT, '配图3_特征工程金字塔.png'))
    plt.close()
    print('[OK] 配图3 特征工程金字塔')


# ═══════════════════════════════════════════════════════════
# 配图 4: VPD 直觉图
# ═══════════════════════════════════════════════════════════
def fig04_vpd():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    ax.text(6.5, 5.1, 'VPD: 捕捉"聚合效应"', ha='center', fontsize=16, fontweight='bold')

    # ── Left: monomer ──
    ax.add_patch(mpatches.FancyBboxPatch((0.3, 2.0), 3.0, 2.2, boxstyle="round,pad=0.15",
                                          facecolor='#E3F2FD', edgecolor='#1565C0', lw=2))
    ax.text(1.8, 3.8, '单体', ha='center', fontsize=13, fontweight='bold', color='#1565C0')
    # Draw single monomer circle
    c1 = plt.Circle((1.8, 2.9), 0.45, color='#1565C0', alpha=0.3, zorder=3)
    ax.add_patch(c1)
    ax.text(1.8, 2.9, 'A', ha='center', va='center', fontsize=16,
            fontweight='bold', color='#1565C0', zorder=4)

    # ── Arrow 1 ──
    ax.annotate('', xy=(4.0, 3.1), xytext=(3.4, 3.1),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2.5))
    ax.text(3.7, 3.6, 'x3', fontsize=11, ha='center', fontweight='bold', color='#E65100')

    # ── Middle: trimer ──
    ax.add_patch(mpatches.FancyBboxPatch((4.1, 2.0), 4.0, 2.2, boxstyle="round,pad=0.15",
                                          facecolor='#FFF3E0', edgecolor='#E65100', lw=2))
    ax.text(6.1, 3.8, '3-mer (虚拟聚合)', ha='center', fontsize=13,
            fontweight='bold', color='#E65100')
    # Draw 3 linked circles
    for i, cx in enumerate([5.0, 6.1, 7.2]):
        c = plt.Circle((cx, 2.9), 0.35, color='#E65100', alpha=0.25, zorder=3)
        ax.add_patch(c)
        ax.text(cx, 2.9, 'A', ha='center', va='center', fontsize=14,
                fontweight='bold', color='#E65100', zorder=4)
    # Links
    ax.plot([5.35, 5.75], [2.9, 2.9], color='#E65100', lw=3, zorder=3)
    ax.plot([6.45, 6.85], [2.9, 2.9], color='#E65100', lw=3, zorder=3)

    # ── Arrow 2 ──
    ax.annotate('', xy=(8.8, 3.1), xytext=(8.2, 3.1),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2.5))

    # ── Right: delta ──
    ax.add_patch(mpatches.FancyBboxPatch((8.9, 2.0), 3.8, 2.2, boxstyle="round,pad=0.15",
                                          facecolor='#E8F5E9', edgecolor='#2E7D32', lw=2))
    ax.text(10.8, 3.8, 'VPD = f(3-mer)/3 - f(A)', ha='center', fontsize=11,
            fontweight='bold', color='#2E7D32')
    ax.text(10.8, 3.1, '  "聚合带来了什么变化？"  ', ha='center', fontsize=11,
            color='#2E7D32',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#A5D6A7'))
    ax.text(10.8, 2.4, '环数 | 柔性 | 极性 | 旋转键', ha='center', fontsize=9, color='#666')

    # ── Bottom: result ──
    ax.add_patch(mpatches.FancyBboxPatch((2.5, 0.3), 8.0, 1.2, boxstyle="round,pad=0.15",
                                          facecolor='#FFEBEE', edgecolor='#C62828', lw=2))
    ax.text(6.5, 0.9,
            '  +4.6% R\u00b2 (0.820 \u2192 0.866)   |   '
            'Top 15 SHAP 特征中 VPD 占 6 席 (40%)  ',
            ha='center', va='center', fontsize=11, fontweight='bold', color='#C62828')

    plt.savefig(os.path.join(OUT, '配图4_VPD直觉图.png'))
    plt.close()
    print('[OK] 配图4 VPD直觉图')


# ═══════════════════════════════════════════════════════════
# 配图 5: 实验结果总览 (消融 + 模型对比, 简洁版)
# ═══════════════════════════════════════════════════════════
def fig05_results():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                    gridspec_kw={'width_ratios': [1, 1.1]})

    # ── Left: Phase 4 ablation (simplified) ──
    labels = ['L1H\n(baseline)', 'M2M-V\n(+VPD)', 'GNN\nembed']
    r2 = [0.820, 0.866, 0.839]
    colors = ['#90CAF9', '#EF5350', '#FFA726']

    bars1 = ax1.bar(labels, r2, color=colors, edgecolor='white', width=0.55)
    for bar, v in zip(bars1, r2):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f'{v:.3f}', ha='center', fontsize=12, fontweight='bold')

    ax1.annotate('+4.6%', xy=(1, 0.866), xytext=(1.6, 0.875),
                fontsize=14, fontweight='bold', color='#C62828',
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=2))

    ax1.set_ylabel('R\u00b2', fontsize=13)
    ax1.set_title('消融实验 (304 samples)', fontsize=13, fontweight='bold')
    ax1.set_ylim(0.79, 0.90)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.text(0.03, 0.03, '* GNN end-to-end: R\u00b2 < 0 (failed)',
             transform=ax1.transAxes, fontsize=8, color='#999')

    # ── Right: Phase 5 model comparison (simplified) ──
    models = ['GBR', 'ExtraTrees', 'LightGBM', 'CatBoost', 'Stacking', 'TabPFN v2']
    r2_5 = [0.856, 0.861, 0.870, 0.874, 0.875, 0.896]
    colors2 = ['#B0BEC5'] * 5 + ['#EF5350']

    bars2 = ax2.barh(models, r2_5, color=colors2, edgecolor='white', height=0.55)
    for bar, v in zip(bars2, r2_5):
        ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{v:.3f}', ha='left', va='center', fontsize=11, fontweight='bold')

    ax2.text(0.893, 5.4, 'zero\ntuning', ha='center', fontsize=9,
             fontweight='bold', color='#C62828')
    ax2.set_xlabel('R\u00b2', fontsize=13)
    ax2.set_title('模型对比 (7,486 samples)', fontsize=13, fontweight='bold')
    ax2.set_xlim(0.84, 0.91)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('实验结果总览', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, '配图5_实验结果总览.png'))
    plt.close()
    print('[OK] 配图5 实验结果总览')


# ═══════════════════════════════════════════════════════════
# 配图 6: 拓展路线图 (均聚物→共聚物→核酸高分子)
# ═══════════════════════════════════════════════════════════
def fig06_expansion():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    ax.text(6.5, 5.1, '预测范围拓展路线', ha='center', fontsize=16, fontweight='bold')

    # Three stages
    stages = [
        (0.3, '#E3F2FD', '#1565C0', 'DONE',
         '均聚物 Tg', '  A-A-A-A-A  ', '单一单体\n7,486 条数据\nR\u00b2 = 0.8955'),
        (4.5, '#FFF3E0', '#E65100', 'WIP',
         '共聚物 Tg', '  A-B-A-B-A  ', '多单体 + 组成比\n104 条实验数据\n+ Fox 虚拟数据'),
        (8.7, '#F3E5F5', '#6A1B9A', 'PLAN',
         '核酸高分子 Tg', '  A-T-G-C-A  ', 'DNA/RNA = 4 元共聚物\n药物递送 + 疫苗稳定性\n文献空白'),
    ]

    for x, bg, border, status, title, structure, desc in stages:
        w, h = 3.8, 3.8
        rect = mpatches.FancyBboxPatch((x, 0.8), w, h, boxstyle="round,pad=0.15",
                                        facecolor=bg, edgecolor=border, lw=2,
                                        ls='--' if status == 'PLAN' else '-')
        ax.add_patch(rect)

        ax.text(x + w/2, 4.2, title, ha='center', fontsize=13,
                fontweight='bold', color=border)
        ax.text(x + w/2, 3.3, structure, ha='center', fontsize=14,
                fontweight='bold', color=border, fontfamily='Consolas',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=border, alpha=0.5))
        ax.text(x + w/2, 2.1, desc, ha='center', fontsize=10, color='#555', linespacing=1.5)

        # Status
        sc = {'DONE': '#2E7D32', 'WIP': '#E65100', 'PLAN': '#6A1B9A'}[status]
        ax.text(x + w/2, 1.1, status, ha='center', fontsize=10, fontweight='bold',
                color='white', bbox=dict(boxstyle='round,pad=0.25', facecolor=sc, edgecolor='none'))

    # Arrows
    for x in [4.1, 8.3]:
        ax.annotate('', xy=(x + 0.35, 2.8), xytext=(x, 2.8),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=2.5))

    # Bottom note
    ax.text(6.5, 0.25,
            '  Key insight: DNA/RNA 本质是共聚物  \u2192  '
            '先建共聚物框架，再纳入核酸  ',
            ha='center', fontsize=10, color='#777',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FAFAFA', edgecolor='#DDD'))

    plt.savefig(os.path.join(OUT, '配图6_拓展路线图.png'))
    plt.close()
    print('[OK] 配图6 拓展路线图')


# ── main ──
if __name__ == '__main__':
    print('generating...\n')
    fig01_roadmap()
    fig02_dataset()
    fig03_feature_pyramid()
    fig04_vpd()
    fig05_results()
    fig06_expansion()
    print(f'\nDone! -> {OUT}')
