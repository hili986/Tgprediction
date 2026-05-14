const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "第14组";
pres.title = "第二次实践辅导课汇报";

const C = {
  primary: "065A82",
  secondary: "1C7293",
  light: "F5F8FA",
  cardBg: "FFFFFF",
  text: "21295C",
  muted: "64748B",
  accent: "E76F51",
  border: "DCE3E8",
  chipBg: "E0F2FE",
  chipText: "065A82",
};

const slide = pres.addSlide();
slide.background = { color: C.light };

// 顶部装饰条
slide.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 10, h: 0.14,
  fill: { color: C.primary },
  line: { color: C.primary, width: 0 },
});

// 题目（双行）
slide.addText(
  [
    { text: "基于物理因果链驱动的多尺度特征工程的", options: { breakLine: true } },
    { text: "高分子玻璃化转变温度预测与分析方法" },
  ],
  {
    x: 0.4, y: 0.26, w: 9.2, h: 1.0,
    fontSize: 22, fontFace: "Microsoft YaHei", bold: true, color: C.primary,
    align: "center", valign: "middle", margin: 0,
  }
);

// 关键词带
const keywords = [
  "玻璃化转变温度预测",
  "多尺度特征工程",
  "物理因果链",
  "TabPFN + GNN",
  "不确定性量化",
  "高分子材料信息学",
];
const chipY = 1.40;
const chipH = 0.42;
const chipGap = 0.12;
const chipsTotalW = 9.2;
const chipW = (chipsTotalW - chipGap * (keywords.length - 1)) / keywords.length;

keywords.forEach((kw, i) => {
  const x = 0.4 + i * (chipW + chipGap);
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y: chipY, w: chipW, h: chipH,
    fill: { color: C.chipBg },
    line: { color: C.chipBg, width: 0 },
    rectRadius: 0.06,
  });
  slide.addText(kw, {
    x, y: chipY, w: chipW, h: chipH,
    fontSize: 11, fontFace: "Microsoft YaHei", bold: true, color: C.chipText,
    align: "center", valign: "middle", margin: 0,
  });
});

// 三列卡片
const colY = 1.97;
const colH = 3.30;
const colGap = 0.18;
const colW = (9.2 - colGap * 2) / 3;
const colAccent = [C.primary, C.secondary, C.accent];

function makeShadow() {
  return { type: "outer", color: "1C7293", blur: 8, offset: 2, angle: 90, opacity: 0.10 };
}

function drawColumn(col, title, sections) {
  const x = 0.4 + col * (colW + colGap);

  // 卡片底
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y: colY, w: colW, h: colH,
    fill: { color: C.cardBg },
    line: { color: C.border, width: 0.75 },
    shadow: makeShadow(),
  });

  // 顶部强调条（与卡片对齐）
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y: colY, w: colW, h: 0.07,
    fill: { color: colAccent[col] },
    line: { color: colAccent[col], width: 0 },
  });

  // 列标题
  slide.addText(title, {
    x: x + 0.20, y: colY + 0.18, w: colW - 0.40, h: 0.42,
    fontSize: 16, fontFace: "Microsoft YaHei", bold: true, color: C.primary,
    align: "left", valign: "middle", margin: 0,
  });

  // 列内容（rich text）
  const richText = [];
  sections.forEach((sec, secIdx) => {
    richText.push({
      text: sec.header,
      options: {
        fontSize: 11, bold: true, color: colAccent[col],
        breakLine: true,
        paraSpaceBefore: secIdx === 0 ? 0 : 8,
        paraSpaceAfter: 2,
      },
    });
    sec.items.forEach((item, itemIdx) => {
      const isLast = secIdx === sections.length - 1 && itemIdx === sec.items.length - 1;
      richText.push({
        text: item,
        options: {
          fontSize: 10.5, color: C.text,
          bullet: true,
          breakLine: !isLast,
        },
      });
    });
  });

  slide.addText(richText, {
    x: x + 0.20, y: colY + 0.66, w: colW - 0.40, h: colH - 0.82,
    fontFace: "Microsoft YaHei", valign: "top", margin: 0,
  });
}

drawColumn(0, "学术论文检索", [
  { header: "国际数据库", items: [
    "Web of Science",
    "Scopus / ScienceDirect",
    "ACS / Wiley / RSC",
    "Springer Nature",
    "arXiv（预印本）",
  ]},
  { header: "中文数据库", items: [
    "中国知网 CNKI",
    "万方数据",
  ]},
  { header: "广度兜底", items: [
    "Google Scholar",
  ]},
]);

drawColumn(1, "专利检索", [
  { header: "国内专利", items: [
    "国家知识产权局 CNIPA",
    "万方专利",
  ]},
  { header: "国际专利", items: [
    "欧洲专利局 Espacenet",
    "美国专利商标局 USPTO",
    "WIPO PatentScope",
  ]},
  { header: "广度兜底", items: [
    "Google Patents",
  ]},
]);

drawColumn(2, "AI 工具与应用边界", [
  { header: "拟使用工具", items: [
    "ChatGPT / Claude",
  ]},
  { header: "仅用于辅助环节", items: [
    "文献检索关键词联想",
    "英文文献辅助阅读",
    "排版与公式编辑",
  ]},
  { header: "不参与的核心环节", items: [
    "实验方案设计",
    "物理机制选择",
    "结果分析与解释",
    "权利要求书撰写",
  ]},
]);

// 底部脚注
slide.addText(
  [
    { text: "第14组", options: { color: C.accent, bold: true } },
    { text: "  ·  2026-05-08  ·  ", options: { color: C.muted } },
    {
      text: "研究方法中的机器学习模型（TabPFN、图神经网络）属于研究核心方法，不计入 AI 辅助工具范畴",
      options: { italic: true, color: C.muted },
    },
  ],
  {
    x: 0.4, y: 5.34, w: 9.2, h: 0.24,
    fontSize: 9.5, fontFace: "Microsoft YaHei",
    align: "center", valign: "middle", margin: 0,
  }
);

const outPath = path.join(
  "C:", "Users", "24020", "Desktop", "Tg预测项目",
  "科学前沿与实践", "第二次实践辅导课-第14组.pptx"
);
pres.writeFile({ fileName: outPath }).then((file) => {
  console.log("Written:", file);
});
