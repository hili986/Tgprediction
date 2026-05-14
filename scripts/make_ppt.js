const pptxgen = require("pptxgenjs");
const path = require("path");

const OUT_DIR = path.resolve(__dirname, "..");

// ── Deep, muted color palette (unified with cover/ending) ──
const C = {
  navy:     "1A365D",
  darkBlue: "2A4365",
  medBlue:  "3B5998",
  slate:    "4A5568",
  steel:    "718096",
  silver:   "A0AEC0",
  fog:      "CBD5E0",
  cloud:    "E2E8F0",
  snow:     "EDF2F7",
  white:    "FFFFFF",
  text:     "2D3748",
  muted:    "718096",
  // Accents (muted, not bright)
  teal:     "285E61",
  green:    "276749",
  amber:    "975A16",
  rust:     "9B2C2C",
  plum:     "553C9A",
};

const FONT_H = "Georgia";
const FONT_B = "Calibri";

let pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "SITP Team";
pres.title = "AI-Assisted Polymer Material Design";

// ── Helpers ──
function darkSlide(s) { s.background = { color: C.navy }; }
function contentSlide(s) {
  s.background = { color: C.snow };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.05, fill: { color: C.darkBlue } });
}
function pageNum(s, n) {
  s.addText(String(n), { x: 9.2, y: 5.2, w: 0.6, h: 0.3, fontSize: 9, color: C.silver, fontFace: FONT_B, align: "right" });
}
function heading(s, t) {
  s.addText(t, { x: 0.6, y: 0.18, w: 8.8, h: 0.5, fontSize: 20, fontFace: FONT_H, color: C.navy, bold: true, margin: 0 });
}
function makeCard(s, x, y, w, h) {
  s.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h, fill: { color: C.white },
    shadow: { type: "outer", blur: 4, offset: 1, angle: 135, color: "000000", opacity: 0.08 }
  });
}
// Muted left-border card
function accentCard(s, x, y, w, h, accentColor) {
  makeCard(s, x, y, w, h);
  s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.06, h, fill: { color: accentColor } });
}

// ═══════════════════════════════════════════════════════════
// 1. Cover
// ═══════════════════════════════════════════════════════════
{
  let s = pres.addSlide(); darkSlide(s);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.6, w: 1.2, h: 0.05, fill: { color: C.silver } });
  s.addText("人工智能辅助高分子材料设计的研究", {
    x: 0.6, y: 1.8, w: 8.8, h: 1.1, fontSize: 30, fontFace: FONT_H, color: C.white, bold: true, margin: 0
  });
  s.addText("AI-Assisted Polymer Material Design", {
    x: 0.6, y: 2.85, w: 8.8, h: 0.45, fontSize: 15, fontFace: FONT_B, color: C.silver, italic: true, margin: 0
  });
  s.addText([
    { text: "同济大学 SITP 大学生创新训练项目", options: { breakLine: true, fontSize: 13 } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "主持人：魏子赫    成员：赵鹏依  郭永欣", options: { breakLine: true, fontSize: 12 } },
    { text: "指导教师：胡勇 教授", options: { breakLine: true, fontSize: 12 } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "升级答辩  |  2026 年 4 月", options: { fontSize: 11 } },
  ], { x: 0.6, y: 3.5, w: 8.8, h: 1.8, fontFace: FONT_B, color: C.fog, margin: 0 });
}

// ═══════════════════════════════════════════════════════════
// 2. Background
// ═══════════════════════════════════════════════════════════
{
  let s = pres.addSlide(); contentSlide(s); heading(s, "一、研究背景"); pageNum(s, 2);

  // Left card
  accentCard(s, 0.5, 0.9, 4.3, 3.7, C.rust);
  s.addText("高分子材料设计的困境", { x: 0.8, y: 1.0, w: 3.8, h: 0.4, fontSize: 14, fontFace: FONT_H, color: C.rust, bold: true, margin: 0 });
  s.addText([
    { text: "传统设计依赖 \"经验 + 试错\"", options: { bullet: true, breakLine: true } },
    { text: "研发周期长、实验成本高", options: { bullet: true, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 4 } },
    { text: "AI 驱动材料设计是新范式，但：", options: { bullet: true, breakLine: true } },
    { text: "高分子缺乏统一结构表示语言", options: { bullet: true, indentLevel: 1, breakLine: true, bold: true, color: C.rust } },
    { text: "小分子有 SMILES，聚合物没有标准", options: { bullet: true, indentLevel: 1, breakLine: true } },
    { text: "无标准 = 无数据库 = 无法做 ML", options: { bullet: true, indentLevel: 1 } },
  ], { x: 0.85, y: 1.5, w: 3.7, h: 2.8, fontSize: 11, fontFace: FONT_B, color: C.text, margin: 0, paraSpaceAfter: 3 });

  // Right card
  accentCard(s, 5.2, 0.9, 4.3, 3.7, C.teal);
  s.addText("我们的切入点", { x: 5.5, y: 1.0, w: 3.8, h: 0.4, fontSize: 14, fontFace: FONT_H, color: C.teal, bold: true, margin: 0 });
  s.addText([
    { text: "BigSMILES (2019, ACS Central Sci.)", options: { breakLine: true, bold: true } },
    { text: "高分子的统一结构表示语言", options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "构建完整链路:", options: { breakLine: true, bold: true } },
    { text: "结构表示 \u2192 数据集 \u2192 性质预测", options: { breakLine: true, fontSize: 13, bold: true, color: C.teal } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "预测目标: 玻璃化转变温度 (Tg)", options: { breakLine: true } },
    { text: "\u2014 最核心的高分子热学性质", options: { italic: true, color: C.muted } },
  ], { x: 5.5, y: 1.5, w: 3.7, h: 2.8, fontSize: 11, fontFace: FONT_B, color: C.text, margin: 0, paraSpaceAfter: 2 });

  s.addText("以 BigSMILES 为基础，探索 AI 在高分子设计中的应用潜力", {
    x: 0.5, y: 4.85, w: 9.0, h: 0.35, fontSize: 11, fontFace: FONT_B, color: C.muted, italic: true, align: "center"
  });
}

// ═══════════════════════════════════════════════════════════
// 3. What is BigSMILES
// ═══════════════════════════════════════════════════════════
{
  let s = pres.addSlide(); contentSlide(s); heading(s, "BigSMILES \u2014 高分子的结构表示语言"); pageNum(s, 3);

  // ── Top: SMILES vs BigSMILES comparison ──
  // Left: SMILES (small molecule)
  accentCard(s, 0.5, 0.85, 4.3, 1.6, C.steel);
  s.addText("SMILES (小分子)", { x: 0.8, y: 0.95, w: 3.8, h: 0.3, fontSize: 12, fontFace: FONT_H, color: C.steel, bold: true, margin: 0 });
  s.addText([
    { text: "乙醇:  ", options: { color: C.muted } },
    { text: "CCO", options: { bold: true, fontFace: "Consolas", fontSize: 13, color: C.darkBlue } },
  ], { x: 0.8, y: 1.35, w: 3.8, h: 0.3, fontSize: 11, fontFace: FONT_B, color: C.text, margin: 0 });
  s.addText("确定性分子 \u2014 一个 SMILES 对应一个确定的分子结构", {
    x: 0.8, y: 1.75, w: 3.8, h: 0.4, fontSize: 9.5, fontFace: FONT_B, color: C.muted, margin: 0
  });

  // Right: BigSMILES (polymer)
  accentCard(s, 5.2, 0.85, 4.3, 1.6, C.teal);
  s.addText("BigSMILES (高分子)", { x: 5.5, y: 0.95, w: 3.8, h: 0.3, fontSize: 12, fontFace: FONT_H, color: C.teal, bold: true, margin: 0 });
  s.addText([
    { text: "聚乙烯:  ", options: { color: C.muted } },
    { text: "{[]CC[]}", options: { bold: true, fontFace: "Consolas", fontSize: 13, color: C.teal } },
  ], { x: 5.5, y: 1.35, w: 3.8, h: 0.3, fontSize: 11, fontFace: FONT_B, color: C.text, margin: 0 });
  s.addText("随机性分子集合 \u2014 花括号 {} 内描述可重复的结构单元", {
    x: 5.5, y: 1.75, w: 3.8, h: 0.4, fontSize: 9.5, fontFace: FONT_B, color: C.muted, margin: 0
  });

  // ── Middle: core concepts ──
  s.addText("BigSMILES 核心机制", { x: 0.6, y: 2.7, w: 4.0, h: 0.35, fontSize: 13, fontFace: FONT_H, color: C.navy, bold: true, margin: 0 });

  const concepts = [
    { c: C.darkBlue, title: "随机对象 { }", desc: "花括号表示\"结构集合\"\n一个 BigSMILES 串对应\n无数种可能的链结构" },
    { c: C.teal,     title: "键描述符 [$] [<] [>]", desc: "描述重复单元之间的\n连接方式 (AA/AB 型)\n区分头尾和键型" },
    { c: C.amber,    title: "嵌套与扩展", desc: "支持嵌段、支化、网络\n端基修饰、环状结构\n可递归嵌套" },
  ];

  concepts.forEach((c, i) => {
    let x = 0.5 + i * 3.1, y = 3.15, w = 2.85, h = 1.65;
    accentCard(s, x, y, w, h, c.c);
    s.addText(c.title, { x: x + 0.2, y: y + 0.1, w: w - 0.4, h: 0.35, fontSize: 11, fontFace: FONT_H, color: c.c, bold: true, margin: 0 });
    s.addText(c.desc, { x: x + 0.2, y: y + 0.5, w: w - 0.4, h: 1.0, fontSize: 10, fontFace: FONT_B, color: C.text, margin: 0 });
  });

  // ── Bottom example ──
  makeCard(s, 0.5, 5.0, 9.0, 0.5);
  s.addText([
    { text: "示例 \u2014 聚苯乙烯-嵌段-聚丁二烯:  ", options: { color: C.muted, fontSize: 10 } },
    { text: "{[$]CC(c1ccccc1)[$]}{[$]CC=CC[$]}", options: { bold: true, fontFace: "Consolas", fontSize: 11, color: C.darkBlue } },
  ], { x: 0.7, y: 5.05, w: 8.6, h: 0.4, fontFace: FONT_B, valign: "middle", margin: 0 });
}

// ═══════════════════════════════════════════════════════════
// 4. Roadmap (native shapes, no image)
// ═══════════════════════════════════════════════════════════
{
  let s = pres.addSlide(); contentSlide(s); heading(s, "二、技术路线总览"); pageNum(s, 4);

  const phases = [
    { x: 0.35,  c: C.darkBlue, letter: "A", title: "理论与语法建模", items: ["文献调研 (25+ 篇)", "BigSMILES 语法梳理", "示例复现"], done: true },
    { x: 2.7,   c: C.teal,     letter: "B", title: "工具开发与实现", items: ["BigSMILES 解析器", "生成器 + 规范化", "语法校验"],       done: true },
    { x: 5.05,  c: C.amber,    letter: "C", title: "数据集与 ML 预测", items: ["Tg 数据集 7,486 条", "特征工程 + VPD", "模型对比 + 消融"], done: true },
    { x: 7.4,   c: C.plum,     letter: "D", title: "扩展探索",       items: ["多尺度物理特征", "共聚物 Tg 预测", "核酸高分子 Tg"],   done: false },
  ];

  phases.forEach((p, idx) => {
    let w = 2.15, h = 3.6, y = 0.95;
    // Card
    s.addShape(pres.shapes.RECTANGLE, {
      x: p.x, y, w, h, fill: { color: C.white },
      shadow: { type: "outer", blur: 4, offset: 1, angle: 135, color: "000000", opacity: 0.08 }
    });
    s.addShape(pres.shapes.RECTANGLE, { x: p.x, y, w, h: 0.05, fill: { color: p.c } });

    // Letter circle
    s.addShape(pres.shapes.OVAL, { x: p.x + w/2 - 0.25, y: y + 0.2, w: 0.5, h: 0.5, fill: { color: p.c } });
    s.addText(p.letter, { x: p.x + w/2 - 0.25, y: y + 0.2, w: 0.5, h: 0.5, fontSize: 14, fontFace: FONT_H, color: C.white, bold: true, align: "center", valign: "middle", margin: 0 });

    // Title
    s.addText(p.title, { x: p.x + 0.15, y: y + 0.85, w: w - 0.3, h: 0.4, fontSize: 11, fontFace: FONT_H, color: p.c, bold: true, align: "center", margin: 0 });

    // Items
    p.items.forEach((item, i) => {
      s.addText(item, { x: p.x + 0.25, y: y + 1.4 + i * 0.5, w: w - 0.5, h: 0.4, fontSize: 10, fontFace: FONT_B, color: C.text, margin: 0 });
    });

    // Status badge
    let badgeColor = p.done ? C.teal : C.amber;
    let badgeText = p.done ? "DONE" : "WIP";
    s.addShape(pres.shapes.RECTANGLE, {
      x: p.x + w/2 - 0.45, y: y + h - 0.55, w: 0.9, h: 0.32,
      fill: { color: badgeColor }, rectRadius: 0.05
    });
    s.addText(badgeText, { x: p.x + w/2 - 0.45, y: y + h - 0.55, w: 0.9, h: 0.32, fontSize: 9, fontFace: FONT_B, color: C.white, bold: true, align: "center", valign: "middle", margin: 0 });

    // Arrow between
    if (idx < 3) {
      s.addShape(pres.shapes.LINE, { x: p.x + w + 0.02, y: y + h/2, w: 0.33, h: 0, line: { color: C.fog, width: 2 } });
      // Arrowhead triangle approximation
      s.addText("\u2192", { x: p.x + w + 0.2, y: y + h/2 - 0.15, w: 0.3, h: 0.3, fontSize: 10, color: C.fog, align: "center", valign: "middle", margin: 0 });
    }
  });
}

// ═══════════════════════════════════════════════════════════
// 4. Phase A - Literature (no "语法说明文档", focus on content)
// ═══════════════════════════════════════════════════════════
{
  let s = pres.addSlide(); contentSlide(s); heading(s, "三、研究进展 \u2014 阶段 A: 文献调研与理论分析"); pageNum(s, 5);

  const topics = [
    { c: C.darkBlue, title: "高分子表示体系", desc: "SMILES / InChI / CurlySMILES / HELM\n各体系在聚合物中的局限性分析" },
    { c: C.teal,     title: "BigSMILES 语法精读", desc: "随机对象、AA/AB 键合描述符\n嵌套、支化、网络、端基等核心语义" },
    { c: C.amber,    title: "Tg 预测方法 SOTA", desc: "覆盖 25+ 篇论文 (2022-2026)\n传统 QSPR、GNN、Transformer 等" },
    { c: C.plum,     title: "数据源与可获取性", desc: "PolyInfo / PolyMetriX / Bicerano\n数据质量评估与整合策略" },
  ];

  topics.forEach((t, i) => {
    let x = 0.5 + (i % 2) * 4.6;
    let y = 0.9 + Math.floor(i / 2) * 2.2;
    accentCard(s, x, y, 4.3, 1.85, t.c);
    s.addText(t.title, { x: x + 0.3, y: y + 0.15, w: 3.8, h: 0.35, fontSize: 13, fontFace: FONT_H, color: t.c, bold: true, margin: 0 });
    s.addText(t.desc, { x: x + 0.3, y: y + 0.55, w: 3.8, h: 1.1, fontSize: 10.5, fontFace: FONT_B, color: C.text, margin: 0 });
  });

  s.addText("产出: 18 份调研报告 (按 基础理论/方法/领域/决策 分类归档)", {
    x: 0.5, y: 5.05, w: 9.0, h: 0.3, fontSize: 11, fontFace: FONT_B, color: C.muted, italic: true, align: "center"
  });
}

// ═══════════════════════════════════════════════════════════
// 5. Phase B - BigSMILES Toolchain
// ═══════════════════════════════════════════════════════════
{
  let s = pres.addSlide(); contentSlide(s); heading(s, "三、研究进展 \u2014 阶段 B: BigSMILES 工具链开发"); pageNum(s, 6);

  const cards = [
    { c: C.darkBlue, title: "解析器",   desc: "BigSMILES 文本\n\u2192 抽象语法树 (AST)" },
    { c: C.teal,     title: "生成器",   desc: "内部结构表示\n\u2192 BigSMILES 文本" },
    { c: C.amber,    title: "规范化",   desc: "重复单元重排序\n\u2192 唯一表达" },
    { c: C.rust,     title: "语法校验", desc: "三阶段流水线\n自动检测非法串" },
  ];

  cards.forEach((c, i) => {
    let x = 0.5 + i * 2.35;
    accentCard(s, x, 0.95, 2.1, 1.7, c.c);
    s.addText(c.title, { x: x + 0.2, y: 1.05, w: 1.8, h: 0.35, fontSize: 13, fontFace: FONT_H, color: c.c, bold: true, margin: 0 });
    s.addText(c.desc, { x: x + 0.2, y: 1.45, w: 1.8, h: 1.0, fontSize: 10.5, fontFace: FONT_B, color: C.text, margin: 0 });
  });

  // Coverage
  makeCard(s, 0.5, 2.95, 9.0, 2.2);
  s.addText("覆盖结构类型", { x: 0.8, y: 3.05, w: 4.0, h: 0.3, fontSize: 13, fontFace: FONT_H, color: C.navy, bold: true, margin: 0 });

  const structs = ["线型均聚物", "随机共聚物", "嵌段共聚物", "交替共聚物", "端基修饰"];
  structs.forEach((st, i) => {
    let x = 0.8 + i * 1.7;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 3.5, w: 1.5, h: 0.45, fill: { color: C.snow } });
    s.addText(st, { x, y: 3.5, w: 1.5, h: 0.45, fontSize: 10, fontFace: FONT_B, color: C.text, align: "center", valign: "middle", margin: 0 });
  });

  s.addText([
    { text: "产出: ", options: { bold: true } },
    { text: "完整 src/bigsmiles/ Python 包 (解析器含三阶段流水线: 分词 \u2192 递归下降解析 \u2192 语义校验)", options: { breakLine: true } },
    { text: "          39 种聚合物 BigSMILES 编码示例库  |  153 个单元测试用例", options: {} },
  ], { x: 0.8, y: 4.2, w: 8.4, h: 0.7, fontSize: 10.5, fontFace: FONT_B, color: C.text, margin: 0 });
}

// ═══════════════════════════════════════════════════════════
// 6. Phase C-1 - Dataset (native shapes)
// ═══════════════════════════════════════════════════════════
{
  let s = pres.addSlide(); contentSlide(s); heading(s, "三、研究进展 \u2014 阶段 C: 数据集建设"); pageNum(s, 7);

  // Three boxes with wider gaps for connector text
  const boxes = [
    { x: 0.3, c: C.darkBlue, title: "Bicerano", num: "304", sub: "线型均聚物\nBigSMILES 标注", solid: true },
    { x: 3.7, c: C.teal,     title: "统一数据集", num: "7,486", sub: "去重 + SMILES 规范化\n+ 异常值检测", solid: true },
    { x: 7.1, c: C.plum,     title: "扩展中", num: "...", sub: "共聚物 + 核酸\n数据收集进行中", solid: false },
  ];

  boxes.forEach((b, i) => {
    let w = 2.5, h = 3.2, y = 1.0;
    s.addShape(pres.shapes.RECTANGLE, {
      x: b.x, y, w, h, fill: { color: C.white },
      line: b.solid ? undefined : { color: b.c, width: 1.5, dashType: "dash" },
      shadow: { type: "outer", blur: 4, offset: 1, angle: 135, color: "000000", opacity: 0.08 }
    });
    s.addShape(pres.shapes.RECTANGLE, { x: b.x, y, w, h: 0.05, fill: { color: b.c } });

    s.addText(b.title, { x: b.x, y: y + 0.2, w, h: 0.35, fontSize: 13, fontFace: FONT_H, color: b.c, bold: true, align: "center", margin: 0 });
    s.addText(b.num, { x: b.x, y: y + 0.65, w, h: 0.7, fontSize: 36, fontFace: FONT_H, color: b.c, bold: true, align: "center", margin: 0 });
    s.addText(b.sub, { x: b.x + 0.2, y: y + 1.5, w: w - 0.4, h: 1.0, fontSize: 10.5, fontFace: FONT_B, color: C.text, align: "center", margin: 0 });

    // Connectors with enough room
    if (i < 2) {
      let ax = b.x + w;
      s.addText("\u2192", { x: ax, y: y + 1.0, w: 1.0, h: 0.4, fontSize: 22, fontFace: FONT_B, color: C.amber, bold: true, align: "center", valign: "middle", margin: 0 });
      s.addText(i === 0 ? "+6 个数据库" : "+共聚物数据", {
        x: ax, y: y + 1.4, w: 1.0, h: 0.35, fontSize: 8, fontFace: FONT_B, color: C.amber, align: "center", margin: 0
      });
    }
  });

  s.addText("数据来源:  Bicerano  |  PolyMetriX  |  NeurIPS OPP  |  Pilania PHA  |  ...", {
    x: 0.5, y: 4.5, w: 9.0, h: 0.35, fontSize: 10, fontFace: FONT_B, color: C.muted, align: "center",
    fill: { color: C.white }
  });
}

// ═══════════════════════════════════════════════════════════
// 7. Phase C-2a - Feature Pyramid (native)
// ═══════════════════════════════════════════════════════════
{
  let s = pres.addSlide(); contentSlide(s); heading(s, "三、研究进展 \u2014 阶段 C: 多层级特征工程"); pageNum(s, 8);

  // Pyramid layers
  const layers = [
    { x: 1.2, y: 3.4, w: 7.6, h: 1.1, c: C.darkBlue, bg: "DBEAFE", title: "基础物理量 (4d)", desc: "分子柔性  |  溶解度  |  分子量  |  密度" },
    { x: 2.0, y: 2.1, w: 6.0, h: 1.1, c: C.teal,     bg: "C6F6D5", title: "2D 分子描述符 (30d)", desc: "拓扑指数  |  极性面积  |  氢键  |  环数  |  ..." },
    { x: 2.8, y: 0.8, w: 4.4, h: 1.1, c: C.amber,    bg: "FEFCBF", title: "VPD 虚拟聚合描述符 (12d)", desc: "聚合效应  |  连接柔性  |  极性变化" },
  ];

  layers.forEach(l => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: l.x, y: l.y, w: l.w, h: l.h, fill: { color: l.bg },
      shadow: { type: "outer", blur: 3, offset: 1, angle: 135, color: "000000", opacity: 0.06 }
    });
    s.addShape(pres.shapes.RECTANGLE, { x: l.x, y: l.y, w: 0.06, h: l.h, fill: { color: l.c } });
    s.addText(l.title, { x: l.x + 0.2, y: l.y + 0.05, w: l.w - 0.4, h: 0.4, fontSize: 12, fontFace: FONT_H, color: l.c, bold: true, margin: 0 });
    s.addText(l.desc, { x: l.x + 0.2, y: l.y + 0.5, w: l.w - 0.4, h: 0.4, fontSize: 10, fontFace: FONT_B, color: C.text, margin: 0 });
  });

  // Core innovation label
  s.addShape(pres.shapes.RECTANGLE, {
    x: 7.5, y: 0.85, w: 1.8, h: 0.45, fill: { color: C.rust }
  });
  s.addText("Core Innovation", { x: 7.5, y: 0.85, w: 1.8, h: 0.45, fontSize: 10, fontFace: FONT_B, color: C.white, bold: true, align: "center", valign: "middle", margin: 0 });

  s.addText("合计: 46 维 = M2M-V 特征集", {
    x: 2.5, y: 4.7, w: 5.0, h: 0.4, fontSize: 12, fontFace: FONT_B, color: C.navy, bold: true, align: "center",
    fill: { color: C.cloud }
  });
}

// ═══════════════════════════════════════════════════════════
// 8. Phase C-2b - VPD Intuition (native)
// ═══════════════════════════════════════════════════════════
{
  let s = pres.addSlide(); contentSlide(s); heading(s, "三、研究进展 \u2014 VPD 虚拟聚合描述符"); pageNum(s, 9);

  // Step 1: Monomer
  accentCard(s, 0.3, 1.0, 2.5, 2.2, C.darkBlue);
  s.addText("单体", { x: 0.5, y: 1.1, w: 2.1, h: 0.35, fontSize: 13, fontFace: FONT_H, color: C.darkBlue, bold: true, margin: 0 });
  s.addShape(pres.shapes.OVAL, { x: 1.15, y: 1.7, w: 0.9, h: 0.9, fill: { color: C.darkBlue, transparency: 75 } });
  s.addText("A", { x: 1.15, y: 1.7, w: 0.9, h: 0.9, fontSize: 20, fontFace: FONT_H, color: C.darkBlue, bold: true, align: "center", valign: "middle", margin: 0 });

  // Arrow x3
  s.addText("\u2192", { x: 2.9, y: 1.85, w: 0.4, h: 0.4, fontSize: 14, color: C.slate, align: "center", valign: "middle", margin: 0 });
  s.addText("x3", { x: 2.85, y: 1.5, w: 0.5, h: 0.3, fontSize: 11, fontFace: FONT_B, color: C.amber, bold: true, align: "center", margin: 0 });

  // Step 2: Trimer
  accentCard(s, 3.4, 1.0, 3.0, 2.2, C.amber);
  s.addText("3-mer (虚拟聚合)", { x: 3.6, y: 1.1, w: 2.6, h: 0.35, fontSize: 13, fontFace: FONT_H, color: C.amber, bold: true, margin: 0 });
  // Three linked circles
  [4.0, 4.8, 5.6].forEach(cx => {
    s.addShape(pres.shapes.OVAL, { x: cx, y: 1.7, w: 0.7, h: 0.7, fill: { color: C.amber, transparency: 75 } });
    s.addText("A", { x: cx, y: 1.7, w: 0.7, h: 0.7, fontSize: 16, fontFace: FONT_H, color: C.amber, bold: true, align: "center", valign: "middle", margin: 0 });
  });
  // Links
  s.addShape(pres.shapes.LINE, { x: 4.7, y: 2.05, w: 0.1, h: 0, line: { color: C.amber, width: 3 } });
  s.addShape(pres.shapes.LINE, { x: 5.5, y: 2.05, w: 0.1, h: 0, line: { color: C.amber, width: 3 } });

  // Arrow
  s.addText("\u2192", { x: 6.5, y: 1.85, w: 0.4, h: 0.4, fontSize: 14, color: C.slate, align: "center", valign: "middle", margin: 0 });

  // Step 3: Delta
  accentCard(s, 7.0, 1.0, 2.7, 2.2, C.teal);
  s.addText("VPD = \u0394 / RU", { x: 7.2, y: 1.1, w: 2.3, h: 0.35, fontSize: 13, fontFace: FONT_H, color: C.teal, bold: true, margin: 0 });
  s.addText("\"聚合带来了\n什么变化？\"", { x: 7.2, y: 1.55, w: 2.3, h: 0.7, fontSize: 12, fontFace: FONT_B, color: C.teal, align: "center", margin: 0 });
  s.addText("环数 | 柔性 | 极性 | 旋转键", { x: 7.2, y: 2.35, w: 2.3, h: 0.3, fontSize: 9, fontFace: FONT_B, color: C.muted, align: "center", margin: 0 });

  // Bottom: result
  makeCard(s, 1.5, 3.6, 7.0, 1.2);
  s.addShape(pres.shapes.RECTANGLE, { x: 1.5, y: 3.6, w: 7.0, h: 0.05, fill: { color: C.rust } });
  s.addText([
    { text: "+4.6% R\u00b2", options: { fontSize: 18, bold: true, color: C.rust } },
    { text: "  (0.820 \u2192 0.866)", options: { fontSize: 13, color: C.text } },
    { text: "    |    Top 15 SHAP 特征中 VPD 占 6 席 (40%)", options: { fontSize: 11, color: C.muted } },
  ], { x: 1.7, y: 3.75, w: 6.6, h: 0.9, fontFace: FONT_B, valign: "middle", margin: 0 });
}

// ═══════════════════════════════════════════════════════════
// 9. Phase C-3 - Results (hand-drawn bars)
// ═══════════════════════════════════════════════════════════
{
  let s = pres.addSlide(); contentSlide(s); heading(s, "三、研究进展 \u2014 阶段 C: 实验结果"); pageNum(s, 10);

  // ── Left: Ablation (manual vertical bars) ──
  s.addText("消融实验 (304 samples)", { x: 0.5, y: 0.85, w: 4.3, h: 0.35, fontSize: 12, fontFace: FONT_H, color: C.navy, bold: true, margin: 0 });

  const ablData = [
    { label: "L1H\n(baseline)", val: 0.820, c: C.steel },
    { label: "M2M-V\n(+VPD)",   val: 0.866, c: C.rust },
    { label: "GNN\nembed",      val: 0.839, c: C.amber },
  ];
  const ablBase = 0.78, ablMax = 0.90, ablChartY = 4.0, ablChartH = 2.5;
  // Y axis
  s.addShape(pres.shapes.LINE, { x: 0.8, y: ablChartY - ablChartH, w: 0, h: ablChartH, line: { color: C.fog, width: 1 } });

  ablData.forEach((d, i) => {
    let barW = 0.9, gap = 1.35;
    let bx = 1.1 + i * gap;
    let ratio = (d.val - ablBase) / (ablMax - ablBase);
    let barH = ratio * ablChartH;
    let by = ablChartY - barH;

    s.addShape(pres.shapes.RECTANGLE, { x: bx, y: by, w: barW, h: barH, fill: { color: d.c } });
    // Value label
    s.addText(d.val.toFixed(3), { x: bx, y: by - 0.3, w: barW, h: 0.28, fontSize: 11, fontFace: FONT_B, color: d.c, bold: true, align: "center", margin: 0 });
    // X label
    s.addText(d.label, { x: bx - 0.1, y: ablChartY + 0.05, w: barW + 0.2, h: 0.5, fontSize: 9, fontFace: FONT_B, color: C.muted, align: "center", margin: 0 });
  });

  // +4.6% annotation
  s.addText("+4.6%", { x: 2.7, y: 1.1, w: 0.8, h: 0.3, fontSize: 12, fontFace: FONT_B, color: C.rust, bold: true, align: "center", margin: 0 });

  // ── Right: Model comparison (manual horizontal bars) ──
  s.addText("模型对比 (7,486 samples)", { x: 5.2, y: 0.85, w: 4.5, h: 0.35, fontSize: 12, fontFace: FONT_H, color: C.navy, bold: true, margin: 0 });

  const modData = [
    { label: "TabPFN v2", val: 0.896, c: C.rust },
    { label: "Stacking",  val: 0.875, c: C.steel },
    { label: "CatBoost",  val: 0.874, c: C.steel },
    { label: "LightGBM",  val: 0.870, c: C.steel },
    { label: "ExtraTrees", val: 0.861, c: C.steel },
    { label: "GBR",       val: 0.856, c: C.steel },
  ];
  const modMinVal = 0.84, modMaxVal = 0.91, modChartX = 6.6, modBarMaxW = 3.1;

  modData.forEach((d, i) => {
    let barH = 0.38, gap = 0.52;
    let by = 1.35 + i * gap;
    let ratio = (d.val - modMinVal) / (modMaxVal - modMinVal);
    let barW = ratio * modBarMaxW;

    // Label
    s.addText(d.label, { x: 5.2, y: by, w: 1.35, h: barH, fontSize: 10, fontFace: FONT_B, color: C.text, align: "right", valign: "middle", margin: 0 });
    // Bar
    s.addShape(pres.shapes.RECTANGLE, { x: modChartX, y: by, w: barW, h: barH, fill: { color: d.c } });
    // Value
    s.addText(d.val.toFixed(3), { x: modChartX + barW + 0.05, y: by, w: 0.6, h: barH, fontSize: 10, fontFace: FONT_B, color: d.c, bold: true, valign: "middle", margin: 0 });
  });

  // "零调参" label for TabPFN
  s.addText("零调参", { x: 9.35, y: 1.2, w: 0.7, h: 0.35, fontSize: 8, fontFace: FONT_B, color: C.rust, bold: true, margin: 0 });

  // Bottom key findings
  makeCard(s, 0.5, 4.55, 9.0, 0.7);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.55, w: 9.0, h: 0.05, fill: { color: C.rust } });
  s.addText([
    { text: "VPD +4.6%", options: { bold: true, color: C.rust } },
    { text: "  |  TabPFN v2 零调参即 SOTA  |  GNN 端到端失败  |  不确定性量化 90% 覆盖率", options: { color: C.text } },
  ], { x: 0.7, y: 4.6, w: 8.6, h: 0.55, fontSize: 10.5, fontFace: FONT_B, align: "center", valign: "middle", margin: 0 });
}

// ═══════════════════════════════════════════════════════════
// 10. Phase D - Expansion (native shapes)
// ═══════════════════════════════════════════════════════════
{
  let s = pres.addSlide(); contentSlide(s); heading(s, "三、研究进展 \u2014 阶段 D: 预测范围拓展"); pageNum(s, 11);

  const stages = [
    { x: 0.3, c: C.darkBlue, title: "均聚物 Tg", seq: "A-A-A-A-A", desc: "单一单体\n7,486 条数据\nR\u00b2 = 0.896", status: "DONE", solid: true },
    { x: 3.6, c: C.amber,    title: "共聚物 Tg", seq: "A-B-A-B-A", desc: "多单体 + 组成比\n104 条实验数据\n+ Fox 虚拟数据", status: "WIP", solid: true },
    { x: 6.9, c: C.plum,     title: "核酸高分子 Tg", seq: "A-T-G-C-A", desc: "DNA/RNA = 4 元共聚物\n药物递送 + 疫苗稳定性\n文献空白", status: "PLAN", solid: false },
  ];

  stages.forEach((st, i) => {
    let w = 3.0, h = 3.4, y = 0.85;
    s.addShape(pres.shapes.RECTANGLE, {
      x: st.x, y, w, h, fill: { color: C.white },
      line: st.solid ? undefined : { color: st.c, width: 1.5, dashType: "dash" },
      shadow: { type: "outer", blur: 4, offset: 1, angle: 135, color: "000000", opacity: 0.08 }
    });
    s.addShape(pres.shapes.RECTANGLE, { x: st.x, y, w, h: 0.05, fill: { color: st.c } });

    s.addText(st.title, { x: st.x, y: y + 0.2, w, h: 0.35, fontSize: 13, fontFace: FONT_H, color: st.c, bold: true, align: "center", margin: 0 });

    // Sequence
    s.addShape(pres.shapes.RECTANGLE, { x: st.x + 0.4, y: y + 0.7, w: w - 0.8, h: 0.45, fill: { color: C.snow } });
    s.addText(st.seq, { x: st.x + 0.4, y: y + 0.7, w: w - 0.8, h: 0.45, fontSize: 14, fontFace: "Consolas", color: st.c, bold: true, align: "center", valign: "middle", margin: 0 });

    s.addText(st.desc, { x: st.x + 0.3, y: y + 1.3, w: w - 0.6, h: 1.2, fontSize: 10.5, fontFace: FONT_B, color: C.text, align: "center", margin: 0 });

    // Status
    let sc = { DONE: C.teal, WIP: C.amber, PLAN: C.plum }[st.status];
    s.addShape(pres.shapes.RECTANGLE, { x: st.x + w/2 - 0.45, y: y + h - 0.55, w: 0.9, h: 0.32, fill: { color: sc } });
    s.addText(st.status, { x: st.x + w/2 - 0.45, y: y + h - 0.55, w: 0.9, h: 0.32, fontSize: 9, fontFace: FONT_B, color: C.white, bold: true, align: "center", valign: "middle", margin: 0 });

    // Arrow
    if (i < 2) {
      s.addText("\u2192", { x: st.x + w + 0.1, y: y + h/2 - 0.15, w: 0.35, h: 0.35, fontSize: 20, color: C.steel, align: "center", valign: "middle", margin: 0 });
    }
  });

  // Bottom insight
  makeCard(s, 1.0, 4.5, 8.0, 0.7);
  s.addText("DNA/RNA 本质是共聚物 \u2192 先建共聚物框架，再纳入核酸  |  ATP/ADP 迁移验证误差 < 5K", {
    x: 1.2, y: 4.55, w: 7.6, h: 0.55, fontSize: 10.5, fontFace: FONT_B, color: C.text, align: "center", valign: "middle", margin: 0
  });
}

// ═══════════════════════════════════════════════════════════
// 11. Innovation Summary
// ═══════════════════════════════════════════════════════════
{
  let s = pres.addSlide(); contentSlide(s); heading(s, "四、创新点总结"); pageNum(s, 12);

  const items = [
    { c: C.darkBlue, num: "01", title: "BigSMILES 工具链", desc: "完整的解析/生成/规范化 Python 库，从\"语言概念\"到\"可用基础设施\"" },
    { c: C.teal,     num: "02", title: "VPD 虚拟聚合描述符", desc: "通过虚拟聚合捕捉\"链级效应\"，突破传统单体描述符的表达瓶颈 (+4.6%)" },
    { c: C.amber,    num: "03", title: "均聚物 \u2192 共聚物 \u2192 核酸", desc: "逐步拓展预测范围，核酸高分子 Tg 预测属文献空白领域" },
    { c: C.rust,     num: "04", title: "系统性实验验证", desc: "15 组消融 + 7 种模型对比 + 不确定性量化，科学严谨" },
  ];

  items.forEach((item, i) => {
    let y = 1.0 + i * 1.1;
    // Number circle
    s.addShape(pres.shapes.OVAL, { x: 0.7, y: y + 0.08, w: 0.5, h: 0.5, fill: { color: item.c } });
    s.addText(item.num, { x: 0.7, y: y + 0.08, w: 0.5, h: 0.5, fontSize: 12, fontFace: FONT_H, color: C.white, bold: true, align: "center", valign: "middle", margin: 0 });
    s.addText(item.title, { x: 1.45, y: y, w: 4.0, h: 0.35, fontSize: 13, fontFace: FONT_H, color: item.c, bold: true, margin: 0 });
    s.addText(item.desc, { x: 1.45, y: y + 0.38, w: 7.8, h: 0.35, fontSize: 10.5, fontFace: FONT_B, color: C.text, margin: 0 });
    if (i < 3) s.addShape(pres.shapes.LINE, { x: 1.45, y: y + 0.9, w: 7.8, h: 0, line: { color: C.cloud, width: 0.5 } });
  });
}

// ═══════════════════════════════════════════════════════════
// 12. Future Plans (table)
// ═══════════════════════════════════════════════════════════
{
  let s = pres.addSlide(); contentSlide(s); heading(s, "五、后续研究计划"); pageNum(s, 13);

  const hdr = { fill: { color: C.darkBlue }, color: C.white, bold: true, fontSize: 11, fontFace: FONT_B, align: "center", valign: "middle" };
  const cell = { fontSize: 10.5, fontFace: FONT_B, color: C.text, valign: "middle" };
  const cellC = { ...cell, align: "center" };

  s.addTable([
    [{ text: "阶段", options: hdr }, { text: "时间", options: hdr }, { text: "内容", options: hdr }],
    [{ text: "多尺度特征", options: { ...cell, bold: true } }, { text: "近期", options: cellC }, { text: "扩展链段/聚合物链尺度特征，提升预测精度", options: cell }],
    [{ text: "共聚物 Tg", options: { ...cell, bold: true } }, { text: "中期", options: cellC }, { text: "从均聚物扩展到共聚物，处理多单体组成与混合比例", options: cell }],
    [{ text: "核酸高分子 Tg", options: { ...cell, bold: true } }, { text: "中期", options: cellC }, { text: "DNA/RNA = 4 元共聚物，在共聚物框架下预测", options: cell }],
    [{ text: "GNN 训练", options: { ...cell, bold: true } }, { text: "中期", options: cellC }, { text: "A800 GPU + 大规模预训练数据训练图神经网络", options: cell }],
    [{ text: "Web 演示", options: { ...cell, bold: true } }, { text: "后期", options: cellC }, { text: "\"输入分子结构 \u2192 预测 Tg + 置信区间\" 交互演示", options: cell }],
    [{ text: "论文撰写", options: { ...cell, bold: true } }, { text: "后期", options: cellC }, { text: "整理成果，投稿学术期刊", options: cell }],
  ], {
    x: 0.5, y: 1.0, w: 9.0,
    colW: [1.8, 0.8, 6.4],
    rowH: [0.42, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48],
    border: { pt: 0.5, color: C.fog },
    autoPage: false,
  });
}

// ═══════════════════════════════════════════════════════════
// 13. Thanks
// ═══════════════════════════════════════════════════════════
{
  let s = pres.addSlide(); darkSlide(s);

  s.addShape(pres.shapes.RECTANGLE, { x: 3.5, y: 1.8, w: 3.0, h: 0.04, fill: { color: C.slate } });
  s.addText("感谢聆听", { x: 1, y: 2.0, w: 8, h: 0.9, fontSize: 38, fontFace: FONT_H, color: C.white, bold: true, align: "center", margin: 0 });

  s.addText([
    { text: "从 BigSMILES 结构表示到 AI 性质预测的完整链路", options: { breakLine: true } },
    { text: "原创虚拟聚合描述符，均聚物 Tg 预测优异性能", options: { breakLine: true } },
    { text: "正在向共聚物和核酸高分子方向拓展", options: {} },
  ], { x: 1.5, y: 3.1, w: 7, h: 1.1, fontSize: 12, fontFace: FONT_B, color: C.silver, align: "center", margin: 0, paraSpaceAfter: 5 });

  s.addText("指导教师: 胡勇 教授  |  同济大学材料科学与工程学院", {
    x: 1, y: 4.5, w: 8, h: 0.35, fontSize: 11, fontFace: FONT_B, color: C.silver, align: "center"
  });
}

// ── Save ──
const outPath = path.join(OUT_DIR, "SITP升级答辩.pptx");
pres.writeFile({ fileName: outPath }).then(() => {
  console.log("Done! ->", outPath);
});
