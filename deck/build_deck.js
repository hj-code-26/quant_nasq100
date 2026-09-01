// GAF·VAE Quant 발표자료 생성기.  node deck/build_deck.js
const pptxgen = require("pptxgenjs");

const NAVY = "0F1B3C", NAVY2 = "1E2E5A", INK = "16213E";
const MINT = "02C39A", CORAL = "F96167", ICE = "CADCFC";
const CARD = "F1F4FA", MUTED = "6B7794", WHITE = "FFFFFF";
const KR = "맑은 고딕";
const W = 13.3, H = 7.5, M = 0.7;

const p = new pptxgen();
p.layout = "LAYOUT_WIDE";
p.author = "GAF-VAE Quant";
p.title = "GAF·VAE Quant";

const sh = () => ({ type: "outer", color: "1E2E5A", blur: 12, offset: 3, angle: 90, opacity: 0.14 });

// 어두운 배경 슬라이드
function dark() {
  const s = p.addSlide();
  s.background = { color: NAVY };
  return s;
}
// 밝은 본문 슬라이드 + 제목/부제
function light(title, kicker) {
  const s = p.addSlide();
  s.background = { color: WHITE };
  if (kicker) s.addText(kicker, {
    x: M, y: 0.42, w: 11.9, h: 0.3, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 12, bold: true, color: MINT, charSpacing: 1.5,
  });
  s.addText(title, {
    x: M, y: kicker ? 0.72 : 0.55, w: 11.9, h: 0.75, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 32, bold: true, color: INK, valign: "top",
  });
  return s;
}
// 번호 원 + 제목 + 본문 (모티프)
function numRow(s, n, x, y, w, head, body, color) {
  s.addShape(p.ShapeType.ellipse, {
    x, y, w: 0.42, h: 0.42, fill: { color: color || MINT },
  });
  s.addText(String(n), {
    x, y, w: 0.42, h: 0.42, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 13, bold: true, color: WHITE, align: "center", valign: "middle",
  });
  s.addText(head, {
    x: x + 0.6, y: y - 0.02, w: w - 0.6, h: 0.32, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 15, bold: true, color: INK,
  });
  s.addText(body, {
    x: x + 0.6, y: y + 0.32, w: w - 0.6, h: 0.8, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 12, color: MUTED, lineSpacing: 17,
  });
}
// 라운드 카드
function card(s, x, y, w, h, fill) {
  s.addShape(p.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.1, fill: { color: fill || CARD }, shadow: sh(),
  });
}
// 큰 숫자 스탯
function stat(s, x, y, w, value, label, color) {
  s.addText(value, {
    x, y, w, h: 0.85, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 40, bold: true, color: color || MINT, align: "center",
  });
  s.addText(label, {
    x, y: y + 0.85, w, h: 0.6, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 12, color: MUTED, align: "center", lineSpacing: 16,
  });
}

/* ───────────── 1. 표지 ───────────── */
{
  const s = dark();
  s.addShape(p.ShapeType.ellipse, { x: 9.6, y: -1.5, w: 5.6, h: 5.6, fill: { color: NAVY2 } });
  s.addShape(p.ShapeType.ellipse, { x: 11.2, y: 5.0, w: 2.6, h: 2.6, fill: { color: NAVY2 } });
  s.addText("논문 재현에서 실거래 시스템까지", {
    x: M, y: 2.15, w: 9.0, h: 0.35, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 14, bold: true, color: MINT, charSpacing: 1.5,
  });
  s.addText("GAF · VAE Quant", {
    x: M, y: 2.6, w: 9.5, h: 1.0, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 52, bold: true, color: WHITE,
  });
  s.addText("캔들차트를 이미지로 인코딩해 주가 방향을 예측하고,\n검증을 통과한 신호만 실제 증권사 API로 주문하는 종단 간 시스템", {
    x: M, y: 3.75, w: 8.6, h: 1.0, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 16, color: ICE, lineSpacing: 26,
  });
  s.addShape(p.ShapeType.line, { x: M, y: 5.15, w: 2.2, h: 0, line: { color: MINT, width: 2 } });
  s.addText("Python · PyTorch · FastAPI · scikit-learn · gs-quant · 토스증권 Open API · Claude API", {
    x: M, y: 5.4, w: 9.5, h: 0.3, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 11, color: MUTED,
  });
  s.addNotes("한 줄 요약: 논문 방법론을 구현한 데서 끝내지 않고, 검증·게이트·주문·자가감사까지 붙인 완결된 시스템을 만들었습니다. 오늘 발표는 그 시스템 엔지니어링이 주제입니다.");
}

/* ───────────── 2. 한눈에 보기 ───────────── */
{
  const s = light("한눈에 보기", "OVERVIEW");
  const items = [
    ["14", "백엔드 모듈\n3,300여 줄", MINT],
    ["250", "나스닥 종목\n일괄 스캔", INK],
    ["2", "매매 트랙\n주간 · 월간", MINT],
    ["3", "대시보드 탭\n분석·스캔·감사", INK],
  ];
  items.forEach(([v, l, c], i) => {
    const x = M + i * 3.05;
    card(s, x, 1.9, 2.75, 2.25);
    stat(s, x, 2.2, 2.75, v, l, c);
  });
  const rows = [
    [1, "데이터 → 예측", "yfinance 일봉 → GAF 이미지 인코딩 →\nVAE 잠재변수 15차원 → 앙상블 분류"],
    [2, "예측 → 검증", "Walk-forward 표본 외 평가에\n라벨 누수·표본 겹침·다중검정 보정까지"],
    [3, "검증 → 주문", "통계적 유의성을 통과한 종목만\n토스증권 Open API로 주문 (기본 dry-run)"],
  ];
  rows.forEach(([n, h2, b], i) => numRow(s, n, M + i * 4.05, 4.7, 3.75, h2, b));
  s.addText("여기에 코드 자체를 점검하는 AI 감사 에이전트가 한 겹 더 붙습니다.", {
    x: M, y: 6.45, w: 11.9, h: 0.35, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 13, italic: true, color: MUTED,
  });
  s.addNotes("규모부터 말씀드립니다. 모듈 14개, 3300줄. 핵심은 세 단계입니다 — 예측하고, 검증하고, 검증을 통과한 것만 주문한다.");
}

/* ───────────── 3. 출발점: 논문 ───────────── */
{
  const s = light("출발점 — 캔들차트를 '이미지'로 본다", "BACKGROUND");
  card(s, M, 1.75, 5.7, 4.75, CARD);
  s.addText("참고 논문", {
    x: M + 0.35, y: 2.0, w: 5.0, h: 0.3, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 11, bold: true, color: MINT, charSpacing: 1.2,
  });
  s.addText("촛대 차트의 극좌표 위치 인코딩 이미지에 대한\n변이형 오토인코더를 이용한 주가 예측", {
    x: M + 0.35, y: 2.35, w: 5.0, h: 0.95, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 17, bold: true, color: INK, lineSpacing: 25,
  });
  s.addText("한양대학교 조예서, 2022", {
    x: M + 0.35, y: 3.35, w: 5.0, h: 0.3, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 12, color: MUTED,
  });
  s.addText([
    { text: "핵심 아이디어", options: { bold: true, color: INK, fontSize: 14, breakLine: true } },
    { text: "시계열을 극좌표로 옮겨 2차원 행렬(GAF)로 만들면, 시점 간 상관관계가 이미지의 공간 구조로 보존된다. 그 이미지를 VAE로 압축한 잠재변수를 예측 변수로 쓴다.", options: { color: MUTED, fontSize: 13 } },
  ], {
    x: M + 0.35, y: 3.85, w: 5.0, h: 2.0, isTextBox: true, margin: 0,
    fontFace: KR, lineSpacing: 21, valign: "top",
  });

  s.addText("논문이 남긴 세 가지 숙제", {
    x: 7.0, y: 1.85, w: 5.6, h: 0.4, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 17, bold: true, color: INK,
  });
  const homework = [
    ["학습 비용", "128×128 RGB · Conv 5층 → 학습에 4시간 50분.\n실시간 재분석이 불가능한 규모."],
    ["평가 방식", "정확도·F1·AUC까지만. '이 성적이 우연인가'를\n따지는 유의성 검정이 없다."],
    ["수익과의 연결", "익일 방향 이진 분류에서 멈춤.\n거래비용·보유기간·주문으로 이어지지 않는다."],
  ];
  homework.forEach(([h2, b], i) => numRow(s, i + 1, 7.0, 2.5 + i * 1.42, 5.6, h2, b, CORAL));
  s.addText("본 프로젝트는 이 방법론을 그대로 쓰되, 세 숙제를 시스템 설계로 채웠다.", {
    x: M + 0.35, y: 5.75, w: 5.0, h: 0.6, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 12, bold: true, color: INK, lineSpacing: 17,
  });
  s.addNotes("논문의 방법론은 그대로 가져오되, 학습 비용·평가의 엄밀성·수익과의 연결 세 가지를 제가 채웠습니다. 이 세 숙제가 이후 슬라이드의 뼈대입니다.");
}

/* ───────────── 4. 아키텍처 ───────────── */
{
  const s = light("파이프라인 아키텍처", "ARCHITECTURE");
  const boxes = [
    ["일봉 OHLC", "yfinance\n5분 캐시", "data.py"],
    ["GAF 인코딩", "GADF+GASF\n2채널 40×40", "gaf.py"],
    ["Conv VAE", "잠재변수\n15차원", "vae.py"],
    ["앙상블 분류", "로지스틱·SVM\n랜덤포레스트", "models.py"],
    ["검증 게이트", "Walk-forward\n이항검정", "trader.py"],
    ["주문", "토스증권\nOpen API", "toss.py"],
  ];
  const bw = 1.83, gap = 0.19;
  boxes.forEach(([t, d, f], i) => {
    const x = M + i * (bw + gap);
    const isLast = i === boxes.length - 1;
    s.addShape(p.ShapeType.roundRect, {
      x, y: 2.0, w: bw, h: 1.85, rectRadius: 0.1,
      fill: { color: isLast ? MINT : (i === 4 ? INK : CARD) }, shadow: sh(),
    });
    s.addText(t, {
      x, y: 2.2, w: bw, h: 0.35, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 13, bold: true, align: "center",
      color: (isLast || i === 4) ? WHITE : INK,
    });
    s.addText(d, {
      x, y: 2.6, w: bw, h: 0.7, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 11, align: "center", lineSpacing: 15,
      color: (isLast || i === 4) ? ICE : MUTED,
    });
    s.addText(f, {
      x, y: 3.36, w: bw, h: 0.3, isTextBox: true, margin: 0,
      fontFace: "Consolas", fontSize: 9, align: "center",
      color: (isLast || i === 4) ? MINT : MUTED,
    });
    if (i < boxes.length - 1) s.addText("›", {
      x: x + bw, y: 2.6, w: gap, h: 0.4, isTextBox: true, margin: 0,
      fontFace: "Arial", fontSize: 18, bold: true, color: MUTED, align: "center",
    });
  });
  const side = [
    ["실시간 계층", "FastAPI + 단일 HTML 대시보드. 현재가 15초 폴링,\n1시간마다 자동 재분석, 학습 결과는 기준일 단위 디스크 캐시.", "server.py · pipeline.py"],
    ["분석 확장 계층", "gs-quant 기반 변동성·RSI·MACD·볼린저·MDD 산출,\n1/5/20일 예상 주가와 ±σ√h 밴드.", "quant.py · indicators.py"],
    ["자가 점검 계층", "GAF 수학 불변식·데이터 무결성·VAE 사후분포 붕괴를 진단하고\n실패 시 Claude API로 패치를 받아 적용·롤백.", "ai_auditor.py"],
  ];
  side.forEach(([h2, b, f], i) => {
    const y = 4.25 + i * 0.9;
    s.addShape(p.ShapeType.ellipse, { x: M, y: y + 0.06, w: 0.16, h: 0.16, fill: { color: MINT } });
    s.addText(h2, {
      x: M + 0.32, y, w: 2.4, h: 0.3, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 13, bold: true, color: INK,
    });
    s.addText(b, {
      x: M + 2.75, y, w: 6.6, h: 0.62, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 11.5, color: MUTED, lineSpacing: 16,
    });
    s.addText(f, {
      x: 9.6, y, w: 3.0, h: 0.3, isTextBox: true, margin: 0,
      fontFace: "Consolas", fontSize: 9.5, color: MUTED, align: "right",
    });
  });
  s.addNotes("위쪽이 예측에서 주문까지의 주 경로, 아래가 이를 감싸는 세 계층입니다. 파일명을 같이 적어둔 이유는 뒤에 시연에서 이 파일들을 그대로 열어보기 때문입니다.");
}

/* ───────────── 5. GAF 인코딩 ───────────── */
{
  const s = light("① GAF 인코딩 — 시계열을 극좌표 이미지로", "PIPELINE 1/3");
  card(s, M, 1.8, 5.9, 2.55);
  s.addText("변환 두 단계", {
    x: M + 0.35, y: 2.0, w: 5.2, h: 0.3, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 13, bold: true, color: MINT,
  });
  s.addText("x̃ᵢ = ((xᵢ − max X) + (xᵢ − min X)) / (max X − min X)\nφᵢ = arccos(x̃ᵢ)", {
    x: M + 0.35, y: 2.4, w: 5.2, h: 0.72, isTextBox: true, margin: 0,
    fontFace: "Cambria", fontSize: 14, color: INK, lineSpacing: 24,
  });
  s.addText("GADF = sin(φᵢ − φⱼ)   ·   GASF = cos(φᵢ + φⱼ)", {
    x: M + 0.35, y: 3.2, w: 5.2, h: 0.32, isTextBox: true, margin: 0,
    fontFace: "Cambria", fontSize: 14, bold: true, color: INK,
  });
  s.addText("값을 [−1,1]로 정규화한 뒤 각도로 바꾸면, 두 시점의 각도 차·합이\n행렬 원소가 된다. 시간 순서와 상관구조가 이미지에 그대로 남는다.", {
    x: M + 0.35, y: 3.6, w: 5.2, h: 0.6, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 11.5, color: MUTED, lineSpacing: 16,
  });

  // 2x2 타일 도식
  s.addText("입력 이미지 구성 — 2×2 타일", {
    x: 7.15, y: 1.85, w: 5.45, h: 0.32, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 13, bold: true, color: INK,
  });
  const tiles = [["종가", 0, 0], ["시가", 1, 0], ["고가", 0, 1], ["저가", 1, 1]];
  tiles.forEach(([t, cx, cy]) => {
    const x = 7.15 + cx * 1.15, y = 2.25 + cy * 1.15;
    s.addShape(p.ShapeType.rect, {
      x, y, w: 1.08, h: 1.08, fill: { color: cy ? NAVY2 : INK },
    });
    s.addText(t, {
      x, y, w: 1.08, h: 1.08, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 13, bold: true, color: WHITE, align: "center", valign: "middle",
    });
  });
  s.addText("20일 창의 OHLC 네 계열을 각각 20×20 GAF로 만들어 하나의 40×40 이미지로 붙인다. GADF·GASF 두 장이 2채널로 들어간다.", {
    x: 9.6, y: 2.3, w: 3.0, h: 1.4, isTextBox: true, margin: 0, valign: "top",
    fontFace: KR, fontSize: 11.5, color: MUTED, lineSpacing: 16,
  });

  card(s, M, 4.8, 11.9, 1.55, "E8F7F2");
  s.addText("논문 대비 개선  ·  GADF 1채널 → GADF + GASF 2채널", {
    x: M + 0.35, y: 5.03, w: 6.0, h: 0.32, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 14, bold: true, color: INK,
  });
  s.addText("각도의 차만 쓰면 대칭 정보가 버려진다. 합(GASF)을 두 번째 채널로 더했더니\nAAPL 기준 정확도 54% → 56%, AUC 0.53 → 0.64 로 올라갔다.", {
    x: M + 0.35, y: 5.4, w: 7.0, h: 0.65, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 12, color: MUTED, lineSpacing: 17,
  });
  s.addText("AUC 0.53 → 0.64", {
    x: 9.2, y: 5.15, w: 3.1, h: 0.6, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 22, bold: true, color: MINT, align: "right",
  });
  s.addNotes("GAF는 논문 3.2.2절 그대로입니다. 제가 바꾼 건 채널 구성 하나 — GASF를 더해 AUC가 0.53에서 0.64로 올랐습니다.");
}

/* ───────────── 6. VAE ───────────── */
{
  const s = light("② Conv VAE — 잠재변수 15차원", "PIPELINE 2/3");
  s.addText("40×40 GAF 이미지를 Conv 인코더로 압축해 평균·분산을 얻고, 재매개변수 트릭으로 15차원 잠재변수를 뽑는다. 손실은 재구축 오차(BCE) + KL 발산.", {
    x: M, y: 1.75, w: 11.9, h: 0.5, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 13.5, color: MUTED, lineSpacing: 19,
  });
  const cols = [
    ["항목", "논문", "본 구현"],
    ["입력 이미지", "128 × 128 RGB", "40 × 40 · 2채널"],
    ["Conv 층", "5층 (128~1024 필터)", "3층 (32~128 필터)"],
    ["잠재 차원", "15", "15 (동일)"],
    ["학습 시간", "4시간 50분", "수십 초"],
    ["재현성", "명시 없음", "torch·numpy 시드 고정"],
    ["재학습", "매 실행", "기준일 단위 디스크 캐시"],
  ];
  const cw = [2.6, 3.6, 3.9], tx = M + 0.4;
  card(s, M, 2.45, 10.4, 3.7);
  cols.forEach((row, r) => {
    const y = 2.68 + r * 0.5;
    if (r === 0) s.addShape(p.ShapeType.line, {
      x: tx, y: y + 0.42, w: 9.6, h: 0, line: { color: MUTED, width: 0.75 },
    });
    row.forEach((cell, c) => {
      s.addText(cell, {
        x: tx + cw.slice(0, c).reduce((a, b) => a + b, 0), y, w: cw[c], h: 0.4,
        isTextBox: true, margin: 0, fontFace: KR,
        fontSize: r === 0 ? 11.5 : 13, bold: r === 0 || c === 2,
        color: r === 0 ? MINT : (c === 2 ? INK : MUTED), valign: "middle",
      });
    });
  });
  card(s, 11.0, 2.45, 1.6, 3.7, INK);
  s.addText("580×", {
    x: 11.0, y: 3.55, w: 1.6, h: 0.6, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 27, bold: true, color: MINT, align: "center",
  });
  s.addText("학습 시간\n단축", {
    x: 11.0, y: 4.15, w: 1.6, h: 0.7, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 12, color: ICE, align: "center", lineSpacing: 17,
  });
  s.addText("규모를 줄인 건 타협이 아니라 요구사항이다 — 대시보드가 1시간마다 자동 재분석하고 250종목을 일괄 스캔하려면 종목당 학습이 초 단위여야 한다.", {
    x: M, y: 6.35, w: 11.9, h: 0.4, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 12.5, italic: true, color: MUTED,
  });
  s.addNotes("논문은 한 종목 학습에 5시간 가까이 걸립니다. 실시간 대시보드와 250종목 스캔이라는 요구사항을 맞추려면 규모를 줄일 수밖에 없었고, 대신 잠재 차원 15와 하이퍼파라미터는 논문 그대로 유지했습니다.");
}

/* ───────────── 7. 분류 + Walk-forward ───────────── */
{
  const s = light("③ 앙상블 분류와 Walk-forward 검증", "PIPELINE 3/3");
  s.addText("세 모형의 확률 평균", {
    x: M, y: 1.8, w: 5.5, h: 0.32, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 15, bold: true, color: INK,
  });
  const models = [
    ["로지스틱 회귀", "선형 경계 · 확률 해석 직관적"],
    ["SVM (RBF)", "비선형 경계 · 확률 보정 적용"],
    ["랜덤 포레스트", "상호작용 포착 · 300 트리"],
  ];
  models.forEach(([h2, b], i) => {
    const y = 2.3 + i * 0.82;
    card(s, M, y, 5.5, 0.68);
    s.addText(h2, {
      x: M + 0.3, y: y + 0.08, w: 2.3, h: 0.28, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 13, bold: true, color: INK,
    });
    s.addText(b, {
      x: M + 2.6, y: y + 0.1, w: 2.7, h: 0.5, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 11, color: MUTED, lineSpacing: 14,
    });
  });
  s.addText("→  앙상블 확률 = 세 확률의 산술 평균", {
    x: M, y: 4.85, w: 5.5, h: 0.32, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 13.5, bold: true, color: MINT,
  });
  s.addText("입력은 VAE 잠재변수 15차원뿐이다. 원본 가격을 직접 넣지 않아 스케일 편향이 들어오지 않는다.", {
    x: M, y: 5.3, w: 5.5, h: 0.65, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 11.5, color: MUTED, lineSpacing: 16,
  });

  // walk-forward 도식
  s.addText("Walk-forward — 항상 과거로 학습, 미래로 평가", {
    x: 6.7, y: 1.8, w: 5.9, h: 0.32, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 15, bold: true, color: INK,
  });
  for (let i = 0; i < 4; i++) {
    const y = 2.35 + i * 0.62;
    const trainW = 2.6 + i * 0.62;
    s.addShape(p.ShapeType.rect, { x: 6.7, y, w: trainW, h: 0.36, fill: { color: ICE } });
    s.addShape(p.ShapeType.rect, { x: 6.7 + trainW, y, w: 0.28, h: 0.36, fill: { color: CORAL } });
    s.addShape(p.ShapeType.rect, { x: 6.7 + trainW + 0.28, y, w: 0.62, h: 0.36, fill: { color: INK } });
    s.addText(`walk ${i + 1}`, {
      x: 6.7 + trainW + 0.98, y, w: 1.0, h: 0.36, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 10, color: MUTED, valign: "middle",
    });
  }
  const legend = [[ICE, "학습 구간"], [CORAL, "embargo (h−1일 폐기)"], [INK, "표본 외 평가 21일"]];
  legend.forEach(([c, t], i) => {
    const y = 4.95 + i * 0.32;
    s.addShape(p.ShapeType.rect, { x: 6.7, y: y + 0.05, w: 0.2, h: 0.16, fill: { color: c } });
    s.addText(t, {
      x: 7.0, y, w: 5.0, h: 0.28, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 11.5, color: MUTED, valign: "middle",
    });
  });
  s.addText("VAE는 고정하고 walk마다 분류기만 재학습한다. 평가에 쓰인 확률은 전부 학습에 쓰이지 않은 표본 외 예측이다.", {
    x: 6.7, y: 6.0, w: 5.9, h: 0.6, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 11.5, color: MUTED, lineSpacing: 16,
  });
  s.addNotes("embargo가 빨간 구간입니다. 다음 슬라이드에서 이게 왜 필요한지 설명합니다.");
}

/* ───────────── 8. 검증의 함정 3가지 ───────────── */
{
  const s = light("검증을 진지하게 — 성적을 부풀리는 세 가지 함정", "VALIDATION");
  s.addText("여기가 이 프로젝트에서 가장 많은 시간을 쓴 부분이다. 이 셋을 처리하지 않으면 없는 예측력이 있다고 나온다.", {
    x: M, y: 1.72, w: 11.9, h: 0.4, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 13.5, color: MUTED,
  });
  const traps = [
    ["라벨 누수", "h일 뒤 종가를 라벨로 쓰면, 학습 구간 끝 h−1개 표본의 정답이 이미 테스트 구간 가격을 참조한다.",
      "walk_forward(embargo = h−1)\n해당 구간을 통째로 버린다", "models.py"],
    ["표본 겹침", "이웃 표본이 미래 구간을 공유해 서로 독립이 아니다. 표본 504개를 504개로 세면 p값이 과소평가된다.",
      "effective_n = n / h\n월간(h=21) → 유효 표본 24개로 검정", "trader.py"],
    ["다중검정", "250종목 중 최고 성적을 고르면, 전부 동전이어도 누군가는 63% 를 낸다.",
      "정확도 63.5%, 단독 p = 0.022\n250종목 보정 후 p = 0.996", "nasdaq100.py"],
  ];
  traps.forEach(([t, problem, fix, file], i) => {
    const x = M + i * 4.05;
    card(s, x, 2.3, 3.75, 3.9);
    s.addShape(p.ShapeType.ellipse, { x: x + 0.3, y: 2.55, w: 0.44, h: 0.44, fill: { color: CORAL } });
    s.addText(String(i + 1), {
      x: x + 0.3, y: 2.55, w: 0.44, h: 0.44, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 14, bold: true, color: WHITE, align: "center", valign: "middle",
    });
    s.addText(t, {
      x: x + 0.9, y: 2.58, w: 2.6, h: 0.38, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 16, bold: true, color: INK, valign: "middle",
    });
    s.addText(problem, {
      x: x + 0.3, y: 3.15, w: 3.15, h: 1.25, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 11.5, color: MUTED, lineSpacing: 16,
    });
    s.addShape(p.ShapeType.roundRect, {
      x: x + 0.3, y: 4.45, w: 3.15, h: 1.05, rectRadius: 0.06, fill: { color: WHITE },
    });
    s.addText(fix, {
      x: x + 0.45, y: 4.6, w: 2.85, h: 0.8, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 11.5, bold: true, color: INK, lineSpacing: 16,
    });
    s.addText(file, {
      x: x + 0.3, y: 5.65, w: 3.15, h: 0.3, isTextBox: true, margin: 0,
      fontFace: "Consolas", fontSize: 9.5, color: MUTED,
    });
  });
  s.addText("셋 다 '성적이 좋아 보이는' 방향으로만 틀린다 — 처리하지 않으면 반드시 과대평가된다.", {
    x: M, y: 6.45, w: 11.9, h: 0.35, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 13, bold: true, color: INK,
  });
  s.addNotes("논문에는 이 세 항목이 없습니다. 특히 다중검정 — 250종목을 돌리고 1등을 자랑하면 안 됩니다. 보정하면 p가 0.996, 즉 전혀 유의하지 않습니다.");
}

/* ───────────── 9. 실측 결과 (차트) ───────────── */
{
  const s = light("실측 결과 — 정확도 분포는 우연의 분포와 겹친다", "RESULT");
  s.addText("나스닥 249종목 · 종목당 표본 외 63일 · 앙상블 예측 정확도", {
    x: M, y: 1.72, w: 8.0, h: 0.35, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 13, color: MUTED,
  });
  s.addChart(p.ChartType.bar, [
    { name: "실측 종목 수", labels: ["~40%", "40-45%", "45-50%", "50-55%", "55-60%", "60-65%", "65%~"], values: [17, 44, 58, 72, 41, 17, 0] },
    { name: "우연일 때 기대값", labels: ["~40%", "40-45%", "45-50%", "50-55%", "55-60%", "60-65%", "65%~"], values: [16.2, 39.8, 68.5, 68.5, 39.8, 13.4, 2.8] },
  ], {
    x: M, y: 2.2, w: 7.6, h: 3.9,
    barDir: "col", barGrouping: "clustered", barGapWidthPct: 40,
    chartColors: [INK, ICE],
    showTitle: false, showLegend: true, legendPos: "t",
    legendFontFace: KR, legendFontSize: 11, legendColor: MUTED,
    catAxisLabelFontFace: KR, catAxisLabelFontSize: 10.5, catAxisLabelColor: MUTED,
    valAxisLabelFontFace: KR, valAxisLabelFontSize: 10.5, valAxisLabelColor: MUTED,
    valGridLine: { color: "E4E8F0", size: 0.75 }, catGridLine: { style: "none" },
  });
  const facts = [
    ["49.95%", "249종목 전체를 합친 정확도\n(p = 0.55 — 유의하지 않음)", INK],
    ["0.511", "평균 AUC\n동전 던지기는 0.500", INK],
    ["13 / 164", "월간 트랙에서 '항상 다수 클래스'\n기준선을 넘긴 종목 수", CORAL],
  ];
  facts.forEach(([v, l, c], i) => {
    const y = 2.35 + i * 1.32;
    card(s, 8.7, y, 3.9, 1.15);
    s.addText(v, {
      x: 8.95, y: y + 0.12, w: 1.5, h: 0.5, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 22, bold: true, color: c,
    });
    s.addText(l, {
      x: 8.95, y: y + 0.62, w: 3.4, h: 0.5, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 10.5, color: MUTED, lineSpacing: 14,
    });
  });
  s.addText("이 겹침이 실패의 증거가 아니라, 검증 장치가 제대로 작동한다는 증거다 — 그래서 결론이 '거래하지 않는다'로 이어진다.", {
    x: M, y: 6.35, w: 11.9, h: 0.4, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 13, bold: true, color: INK,
  });
  s.addNotes("가장 중요한 슬라이드입니다. 파란 막대가 실측, 연한 막대가 '전부 동전 던지기였다면' 기대되는 분포입니다. 거의 포개집니다. 논문 수준의 정확도 표만 보여줬다면 '55% 넘는 종목이 58개나 된다'고 자랑했을 텐데, 우연 기대치가 53개입니다.");
}

/* ───────────── 10. 거래 게이트 ───────────── */
{
  const s = dark();
  s.addText("DESIGN DECISION", {
    x: M, y: 0.6, w: 11.9, h: 0.3, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 12, bold: true, color: MINT, charSpacing: 1.5,
  });
  s.addText("예측력이 없으면 주문하지 않는다", {
    x: M, y: 0.95, w: 11.9, h: 0.7, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 32, bold: true, color: WHITE,
  });
  s.addText("모델이 확률을 뱉는다고 곧바로 주문으로 넘기지 않는다. 종목·트랙별로 표본 외 성적이 우연과 구분되는지 먼저 검정하고, 통과하지 못하면 그 종목·그 기간은 HOLD 로 고정한다.", {
    x: M, y: 1.75, w: 11.9, h: 0.5, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 14, color: ICE, lineSpacing: 20,
  });
  const gates = [
    ["게이트 1 — 유의성", "이항검정 단측 p < 0.05\n겹침 보정 유효 표본 기준", MINT],
    ["게이트 2 — 임계값", "표본 외 예측 위에서 그리드 탐색,\n샤프 최대 지점 채택", MINT],
    ["게이트 3 — 추세 필터", "종가 ≥ SMA20 일 때만 매수\n아래면 매도 쪽으로", MINT],
    ["게이트 4 — 예산 상한", "주간 트랙 매수여력 10%\n월간 트랙 25% 로 고정", MINT],
  ];
  gates.forEach(([h2, b], i) => {
    const x = M + i * 3.05;
    s.addShape(p.ShapeType.roundRect, {
      x, y: 2.5, w: 2.75, h: 1.75, rectRadius: 0.1, fill: { color: NAVY2 },
    });
    s.addText(h2, {
      x: x + 0.25, y: 2.72, w: 2.25, h: 0.5, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 13, bold: true, color: MINT, lineSpacing: 17,
    });
    s.addText(b, {
      x: x + 0.25, y: 3.25, w: 2.3, h: 0.85, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 11, color: ICE, lineSpacing: 15,
    });
  });
  s.addText("두 트랙으로 나눈 이유", {
    x: M, y: 4.6, w: 5.6, h: 0.35, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 15, bold: true, color: WHITE,
  });
  s.addText("익일 방향은 노이즈가 지배적이라 라벨로 쓰기 나쁘고, 매일 신호를 갈아타면 거래비용만 쌓인다. 그래서 보유 기간을 명시적으로 잡았다.\n\n주간 h=5 · 예산 10% · 12 walks (유효 표본 ~50)\n월간 h=21 · 예산 25% · 24 walks (유효 표본 ~24)", {
    x: M, y: 5.0, w: 5.6, h: 1.6, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 12, color: ICE, lineSpacing: 17,
  });
  s.addShape(p.ShapeType.roundRect, {
    x: 6.9, y: 4.6, w: 5.7, h: 2.05, rectRadius: 0.1, fill: { color: NAVY2 },
  });
  s.addText("실제로 무엇을 반영했나", {
    x: 7.2, y: 4.8, w: 5.1, h: 0.32, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 13, bold: true, color: MINT,
  });
  s.addText("· 거래비용 10bp — 토스증권 실계좌 조회값(수수료 0.100%)\n· 왕복 20bp, 스프레드·환전은 미반영이라 이 값도 낙관적\n· 백테스트는 walk-forward 표본 외 예측만으로 계산\n· 기본은 dry-run — 주문은 --live 를 명시할 때만 나간다", {
    x: 7.2, y: 5.2, w: 5.1, h: 1.3, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 11.5, color: ICE, lineSpacing: 18,
  });
  s.addNotes("결과가 안 나왔으니 시스템이 실패한 게 아니라, 결과가 안 나왔다는 걸 시스템이 스스로 판정하고 주문을 막는 것이 설계 목표였습니다.");
}

/* ───────────── 11. 대시보드 ───────────── */
{
  const s = light("실시간 대시보드 — FastAPI + 단일 페이지", "SYSTEM 1/3");
  const tabs = [
    ["① 종목 분석", [
      "캔들 + MA20/60 + 볼린저 + 신호",
      "1/5/20일 예상 주가와 ±σ√h 밴드",
      "표본 외 백테스트: 전략 vs 단순 보유",
      "현재가 15초 폴링 · 1시간 자동 재분석",
    ]],
    ["② 나스닥 스캔", [
      "250종목에 고속 파이프라인 일괄 적용",
      "표본 외 정확도 내림차순 정렬",
      "진행률·중지·디스크 캐시",
      "행 클릭 → 해당 종목 정밀 분석으로",
    ]],
    ["③ AI 코드 감사", [
      "GAF 수학 불변식·데이터 무결성 진단",
      "VAE 사후분포 붕괴 · 분류기 건전성",
      "실패 시 Claude API 패치 → 재진단",
      "개선 없으면 자동 롤백",
    ]],
  ];
  tabs.forEach(([t, items], i) => {
    const x = M + i * 4.05;
    card(s, x, 1.85, 3.75, 2.55);
    s.addText(t, {
      x: x + 0.3, y: 2.1, w: 3.15, h: 0.38, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 16, bold: true, color: INK,
    });
    s.addText(items.map((it, k) => ({
      text: it, options: { bullet: true, breakLine: k < items.length - 1 },
    })), {
      x: x + 0.3, y: 2.6, w: 3.2, h: 2.1, isTextBox: true, margin: 0, valign: "top",
      fontFace: KR, fontSize: 11.5, color: MUTED, paraSpaceAfter: 6,
    });
  });
  s.addText("주요 엔드포인트", {
    x: M, y: 4.7, w: 5.0, h: 0.32, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 14, bold: true, color: INK,
  });
  const eps = [
    "POST /api/analyze          종목 분석 시작 (백그라운드)",
    "GET  /api/scan/stats       스캔 결과 유의성 검정",
    "POST /api/trade/plan       주간·월간 매매 계획 산출",
    "POST /api/trade/execute    계획 실행 (live=false 기본)",
    "GET  /api/toss/summary     계좌·보유·매수여력 조회",
  ];
  s.addText(eps.join("\n"), {
    x: M, y: 5.15, w: 7.6, h: 1.2, isTextBox: true, margin: 0,
    fontFace: "Consolas", fontSize: 11, color: MUTED, lineSpacing: 16,
  });
  card(s, 8.7, 4.7, 3.9, 1.65, "E8F7F2");
  s.addText("무거운 작업은 전부 백그라운드 스레드 + 상태 폴링이다. 계획 산출은 VAE 2회 학습과 36 walks 라 수 분이 걸리는데, HTTP 를 막지 않는다.", {
    x: 8.95, y: 4.95, w: 3.4, h: 1.2, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 11.5, color: MUTED, lineSpacing: 16,
  });
  s.addNotes("탭 세 개, 엔드포인트는 이 정도입니다. 시연에서 이 화면을 직접 보여드립니다.");
}

/* ───────────── 12. 토스 API ───────────── */
{
  const s = light("토스증권 Open API 연동", "SYSTEM 2/3");
  s.addText("예측이 실제 주문으로 이어지는 마지막 구간. OAuth2 client_credentials 로 토큰을 받아 만료 60초 전에 자동 갱신하고, 429 는 Retry-After 를 따라 지수 백오프로 재시도한다.", {
    x: M, y: 1.72, w: 11.9, h: 0.5, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 13.5, color: MUTED, lineSpacing: 19,
  });
  const groups = [
    ["시세 · 종목", ["prices()", "orderbook()", "candles()", "exchange_rate()", "us_market_calendar()"], "토큰만 필요"],
    ["계좌 · 자산", ["accounts()", "holdings()", "buying_power()", "commissions()", "sellable_quantity()"], "계좌 헤더 필요"],
    ["주문", ["buy() / sell()", "create_order()", "cancel_order()", "modify_order()", "orders()"], "실행은 --live 만"],
  ];
  groups.forEach(([t, fns, note], i) => {
    const x = M + i * 4.05;
    const isOrder = i === 2;
    s.addShape(p.ShapeType.roundRect, {
      x, y: 2.4, w: 3.75, h: 2.75, rectRadius: 0.1,
      fill: { color: isOrder ? INK : CARD }, shadow: sh(),
    });
    s.addText(t, {
      x: x + 0.3, y: 2.62, w: 3.15, h: 0.35, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 15, bold: true, color: isOrder ? WHITE : INK,
    });
    s.addText(fns.join("\n"), {
      x: x + 0.3, y: 3.05, w: 3.15, h: 1.35, isTextBox: true, margin: 0,
      fontFace: "Consolas", fontSize: 11, color: isOrder ? ICE : MUTED, lineSpacing: 16,
    });
    s.addText(note, {
      x: x + 0.3, y: 4.6, w: 3.15, h: 0.3, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 11, bold: true, color: isOrder ? CORAL : MINT,
    });
  });
  card(s, M, 5.4, 11.9, 1.3, "FDEEEF");
  s.addText("안전장치", {
    x: M + 0.35, y: 5.58, w: 2.0, h: 0.3, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 13, bold: true, color: CORAL,
  });
  s.addText("기본 실행은 dry-run — 수량·가격까지 계산해 거래 일지(trade_journal.jsonl)에만 기록하고 주문은 내지 않는다.\n실제 주문은 --live 를 명시할 때만, 그것도 유의성 게이트를 통과한 트랙에 한해서 나간다. 매수 수량은 매수여력 × 트랙 예산 비중으로 제한된다.", {
    x: M + 2.4, y: 5.55, w: 9.2, h: 0.95, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 12, color: INK, lineSpacing: 18,
  });
  s.addNotes("주문 API는 다 붙여놨지만, 기본값이 dry-run이라는 게 핵심입니다. 실수로 주문이 나가지 않게 --live 플래그를 명시적으로 요구합니다.");
}

/* ───────────── 13. AI 감사 ───────────── */
{
  const s = light("AI 코드 감사 에이전트 — 자기 자신을 점검하는 계층", "SYSTEM 3/3");
  s.addText("수치 파이프라인의 버그는 조용하다. 틀려도 그럴듯한 숫자가 나온다. 그래서 수학적 불변식을 코드로 박아두고, 실패하면 Claude API 에 진단 리포트와 소스를 보내 최소 패치를 받는다.", {
    x: M, y: 1.72, w: 11.9, h: 0.5, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 13.5, color: MUTED, lineSpacing: 19,
  });
  const steps = [
    ["진단", "코드 컴파일 · GAF 불변식(GADF 반대칭·대각 0·범위, GASF 대칭)\n데이터셋 무결성(NaN·라벨 균형·최신성) · VAE 사후분포 붕괴\n분류기·Walk-forward 건전성 · gs-quant 지표"],
    ["분석 · 패치", "실패 항목이 있으면 진단 리포트 + 소스를 Claude API 로 전송,\n원인 분석과 최소 패치를 JSON 으로 수신"],
    ["적용 · 재검증", "화이트리스트 파일만 .py.bak 백업 후 패치 → 컴파일 검사\n→ 모듈 리로드 → 재진단"],
    ["롤백", "개선이 없으면 자동 롤백. 최대 2회 시도 후 중단"],
  ];
  steps.forEach(([t, b], i) => {
    const y = 2.4 + i * 1.02;
    s.addShape(p.ShapeType.ellipse, { x: M, y: y + 0.06, w: 0.5, h: 0.5, fill: { color: i === 3 ? CORAL : MINT } });
    s.addText(String(i + 1), {
      x: M, y: y + 0.06, w: 0.5, h: 0.5, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 15, bold: true, color: WHITE, align: "center", valign: "middle",
    });
    if (i < 3) s.addShape(p.ShapeType.line, {
      x: M + 0.25, y: y + 0.56, w: 0, h: 0.46, line: { color: ICE, width: 1.5 },
    });
    s.addText(t, {
      x: M + 0.75, y: y + 0.05, w: 2.0, h: 0.35, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 15, bold: true, color: INK,
    });
    s.addText(b, {
      x: M + 2.85, y: y + 0.02, w: 6.3, h: 0.85, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 11.5, color: MUTED, lineSpacing: 16,
    });
  });
  card(s, 10.0, 2.4, 2.6, 3.65, INK);
  s.addText("패치 허용\n파일만", {
    x: 10.2, y: 2.65, w: 2.2, h: 0.6, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 13, bold: true, color: MINT, lineSpacing: 18,
  });
  s.addText("gaf.py\nvae.py\nmodels.py\nquant.py\nbacktest.py\npipeline.py\nnasdaq100.py\ndata.py", {
    x: 10.2, y: 3.3, w: 2.2, h: 2.2, isTextBox: true, margin: 0,
    fontFace: "Consolas", fontSize: 11, color: ICE, lineSpacing: 17,
  });
  s.addText("toss.py · server.py 는\n제외 — 주문 경로는\nAI 가 건드리지 않는다", {
    x: 10.2, y: 5.4, w: 2.2, h: 0.6, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 10, color: CORAL, lineSpacing: 14,
  });
  s.addNotes("주문이 나가는 경로인 toss.py와 server.py는 화이트리스트에서 빼뒀습니다. AI가 자동으로 고칠 수 있는 범위를 계산 코드로 한정한 것입니다.");
}

/* ───────────── 14. 시연 (영상) ───────────── */
{
  const s = dark();
  s.addText("LIVE DEMO", {
    x: M, y: 0.6, w: 11.9, h: 0.3, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 12, bold: true, color: MINT, charSpacing: 1.5,
  });
  s.addText("시연 — 예측에서 주문까지", {
    x: M, y: 0.95, w: 11.9, h: 0.7, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 32, bold: true, color: WHITE,
  });
  // 영상 자리
  s.addShape(p.ShapeType.roundRect, {
    x: M, y: 1.9, w: 7.5, h: 4.35, rectRadius: 0.1, fill: { color: NAVY2 },
  });
  s.addShape(p.ShapeType.ellipse, { x: 3.85, y: 3.45, w: 1.2, h: 1.2, fill: { color: MINT } });
  s.addText("▶", {
    x: 3.85, y: 3.45, w: 1.2, h: 1.2, isTextBox: true, margin: 0,
    fontFace: "Arial", fontSize: 30, color: NAVY, align: "center", valign: "middle",
  });
  s.addText("영상 삽입 위치", {
    x: M, y: 4.85, w: 7.5, h: 0.35, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 16, bold: true, color: WHITE, align: "center",
  });
  s.addText("PowerPoint 리본 → 삽입 → 비디오 → 이 디바이스 →  demo.mp4 를 이 사각형 위에 올려 크기를 맞춥니다", {
    x: M + 0.5, y: 5.25, w: 6.5, h: 0.6, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 11, color: MUTED, align: "center", lineSpacing: 16,
  });
  // 시연 순서
  s.addText("시연 순서", {
    x: 8.6, y: 1.9, w: 4.0, h: 0.35, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 16, bold: true, color: WHITE,
  });
  const demo = [
    ["00:00", "make account", "실계좌 연결 확인 — 잔고·현재가 조회"],
    ["00:30", "make open", "대시보드 3개 탭 · 캔들과 신호 · 스캔 결과 정렬"],
    ["01:30", "make plan TICKER=…", "주간·월간 계획 산출, 유의성 검정과 판정 출력"],
    ["02:30", "판정 읽기", "p ≥ 0.05 → '우연과 구분 불가' → 주문 없음"],
    ["03:00", "make journal", "거래 일지와 현재가 대조 — 기록은 남는다"],
  ];
  demo.forEach(([t, cmd, desc], i) => {
    const y = 2.4 + i * 0.78;
    s.addText(t, {
      x: 8.6, y, w: 0.75, h: 0.28, isTextBox: true, margin: 0,
      fontFace: "Consolas", fontSize: 11, bold: true, color: MINT,
    });
    s.addText(cmd, {
      x: 9.4, y, w: 3.2, h: 0.28, isTextBox: true, margin: 0,
      fontFace: "Consolas", fontSize: 11.5, bold: true, color: WHITE,
    });
    s.addText(desc, {
      x: 9.4, y: y + 0.28, w: 3.2, h: 0.5, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 10.5, color: ICE, lineSpacing: 14,
    });
  });
  s.addText("녹화 전: .env 자격증명 · make status · 글꼴 16pt 이상", {
    x: 8.6, y: 6.35, w: 4.0, h: 0.5, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 10, italic: true, color: MUTED, lineSpacing: 14,
  });
  s.addNotes("시연 스크립트는 deck/DEMO_SCRIPT.md 에 명령어와 나레이션까지 정리해뒀습니다. 하이라이트는 마지막 판정 — 시스템이 스스로 '거래하지 않는다'고 말하는 장면입니다.");
}

/* ───────────── 15. 정리 ───────────── */
{
  const s = light("정리 — 무엇을 만들었고 무엇이 남았나", "WRAP-UP");
  const done = [
    "논문의 GAF·VAE 파이프라인을 재현하고 2채널로 개선 (AUC 0.53 → 0.64)",
    "학습 시간을 4시간 50분에서 수십 초로 줄여 실시간 재분석과 250종목 스캔을 가능하게 함",
    "라벨 누수·표본 겹침·다중검정을 처리한 표본 외 검증 체계 구축",
    "유의성 게이트를 통과한 신호만 토스증권 API 로 주문하는 경로 완성 (기본 dry-run)",
    "수학 불변식 진단 + AI 패치·롤백 루프로 계산 코드 자가 점검",
  ];
  const next = [
    "특징을 잠재변수 15차원에만 의존 — 거래량·섹터·거시 변수 결합",
    "라벨을 방향 이진에서 초과수익·변동성 조정 수익으로 교체",
    "지표 블렌딩 가중치 0.3 고정 — 백테스트 기반 최적화 필요",
    "스프레드·환전 비용 미반영 — 체결 시뮬레이션 정교화",
    "실계좌 소액 페이퍼 트레이딩으로 전 구간 검증",
  ];
  card(s, M, 1.85, 5.85, 4.55, "E8F7F2");
  s.addText("완료", {
    x: M + 0.35, y: 2.1, w: 3.0, h: 0.35, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 17, bold: true, color: INK,
  });
  s.addText(done.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < done.length - 1 } })), {
    x: M + 0.35, y: 2.6, w: 5.15, h: 3.6, isTextBox: true, margin: 0, valign: "top",
    fontFace: KR, fontSize: 12.5, color: INK, paraSpaceAfter: 20, lineSpacing: 19,
  });
  card(s, 6.85, 1.85, 5.75, 4.55);
  s.addText("남은 과제", {
    x: 7.2, y: 2.1, w: 3.0, h: 0.35, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 17, bold: true, color: CORAL,
  });
  s.addText(next.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < next.length - 1 } })), {
    x: 7.2, y: 2.6, w: 5.05, h: 3.6, isTextBox: true, margin: 0, valign: "top",
    fontFace: KR, fontSize: 12.5, color: MUTED, paraSpaceAfter: 20, lineSpacing: 19,
  });
  s.addText("본 프로그램은 연구·교육용이며 투자 판단의 근거가 될 수 없습니다.", {
    x: M, y: 6.6, w: 11.9, h: 0.35, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 11, italic: true, color: MUTED,
  });
  s.addNotes("성과를 한 줄로 요약하면 — 예측 모델을 만든 게 아니라, 예측이 되는지 안 되는지를 판정하고 그에 맞게 행동하는 시스템을 만들었습니다.");
}

/* ───────────── 16. 마무리 ───────────── */
{
  const s = dark();
  s.addShape(p.ShapeType.ellipse, { x: -1.8, y: 4.2, w: 5.0, h: 5.0, fill: { color: NAVY2 } });
  s.addText("감사합니다", {
    x: M, y: 2.9, w: 8.0, h: 1.0, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 46, bold: true, color: WHITE,
  });
  s.addText("질문 환영합니다", {
    x: M, y: 3.95, w: 8.0, h: 0.4, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 16, color: MINT,
  });
  s.addShape(p.ShapeType.line, { x: M, y: 4.65, w: 2.2, h: 0, line: { color: MINT, width: 2 } });
  s.addText("GAF · VAE Quant   ·   논문: 조예서(2022) 한양대학교", {
    x: M, y: 4.9, w: 8.0, h: 0.32, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 12, color: MUTED,
  });
  s.addNotes("예상 질문: 왜 정확도가 안 나오는가 / 왜 그런데도 시스템을 만들었나 / 실제 돈을 넣었나(아니오, dry-run) / 논문과 결과가 다른 이유는 무엇인가.");
}

p.writeFile({ fileName: "deck/GAF-VAE-Quant-발표자료.pptx" })
  .then((f) => console.log("wrote", f));
