import {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, WidthType, BorderStyle, HeadingLevel, ImageRun,
  Header, Footer, PageNumber, ShadingType,
  TableLayoutType, VerticalAlign, PageBreak,
} from "docx";
import * as fs from "fs";

// ── Style constants ─────────────────────────────────────────────────────────
const ACCENT = "1A5276";
const ACCENT_LIGHT = "D4E6F1";
const GREEN = "27AE60";
const RED = "E74C3C";
const ORANGE = "E67E22";
const GRAY_BG = "F8F9FA";
const FONT = "Calibri";
const CODE_FONT = "Consolas";

// ── Load data ───────────────────────────────────────────────────────────────
const results = JSON.parse(fs.readFileSync("chord_comparison_results.json", "utf-8"));
const img1 = fs.readFileSync("chord_iter1_comparison_cropped.png");
const img2 = fs.readFileSync("chord_iter2_comparison_cropped.png");
const img3 = fs.readFileSync("chord_iter3_comparison_cropped.png");

// ── Helpers ─────────────────────────────────────────────────────────────────
const t = (text, opts = {}) => new TextRun({ text, font: opts.font || FONT, size: opts.size || 22, ...opts });
const p = (runs, opts = {}) => new Paragraph({
  children: (Array.isArray(runs) ? runs : [runs]).map(r => typeof r === "string" ? t(r) : r),
  spacing: { after: opts.after || 120, before: opts.before || 0 },
  alignment: opts.alignment || AlignmentType.LEFT,
  ...opts,
});
const heading = (text, level = HeadingLevel.HEADING_1) => new Paragraph({
  children: [t(text, { bold: true, color: ACCENT, size: level === HeadingLevel.HEADING_1 ? 32 : 26 })],
  heading: level,
  spacing: { before: 240, after: 120 },
});

function codeBlock(lines) {
  return lines.map(line => new Paragraph({
    children: [t(line, { font: CODE_FONT, size: 18 })],
    spacing: { after: 0 },
    shading: { type: ShadingType.CLEAR, fill: GRAY_BG },
    indent: { left: 200 },
  }));
}

function hCell(text, width) {
  return new TableCell({
    children: [new Paragraph({ children: [t(text, { bold: true, color: "FFFFFF", size: 20 })], alignment: AlignmentType.CENTER, spacing: { after: 0 } })],
    shading: { type: ShadingType.CLEAR, fill: ACCENT },
    verticalAlign: VerticalAlign.CENTER,
    width: width ? { size: width, type: WidthType.PERCENTAGE } : undefined,
  });
}

function dCell(text, opts = {}) {
  return new TableCell({
    children: [new Paragraph({
      children: [t(String(text), { size: 20, bold: opts.bold, color: opts.color || "000000" })],
      alignment: opts.align || AlignmentType.CENTER,
      spacing: { after: 0 },
    })],
    shading: opts.shading,
    verticalAlign: VerticalAlign.CENTER,
  });
}

const yesCell = () => dCell("YES", { bold: true, color: GREEN, shading: { type: ShadingType.CLEAR, fill: "D5F5E3" } });
const noCell = (label = "NO") => dCell(label, { bold: true, color: RED, shading: { type: ShadingType.CLEAR, fill: "FADBD8" } });
const missCell = () => dCell("MISS", { bold: true, color: RED, shading: { type: ShadingType.CLEAR, fill: "FADBD8" } });
const rowBg = (i) => ({ type: ShadingType.CLEAR, fill: i % 2 === 0 ? "FFFFFF" : "F2F3F4" });

// ── Chord names for ground truth table ──────────────────────────────────────
const GT_CHORDS = [
  { name: "C major", notes: ["C3", "E3", "G3"] },
  { name: "G major", notes: ["G2", "B2", "D3", "G3"] },
  { name: "A minor", notes: ["A2", "E3", "A3"] },
  { name: "F major", notes: ["F2", "A2", "C3", "F3"] },
];

// ── Build document ──────────────────────────────────────────────────────────
const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 22 } } } },
  sections: [{
    properties: {
      page: { margin: { top: 1440, bottom: 1440, left: 1260, right: 1260 } },
    },
    headers: {
      default: new Header({
        children: [p([t("Chord Performance Comparison | GuitarScribe", { size: 18, color: ACCENT, italics: true })], { alignment: AlignmentType.RIGHT })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, color: "666666" })],
          alignment: AlignmentType.CENTER,
        })],
      }),
    },
    children: [
      // ═══════════ TITLE PAGE ═══════════
      ...Array(6).fill(null).map(() => p([], { after: 200 })),
      p([t("GuitarScribe", { size: 56, bold: true, color: ACCENT })], { alignment: AlignmentType.CENTER, after: 100 }),
      p([t("Chord Performance Comparison", { size: 36, color: ACCENT })], { alignment: AlignmentType.CENTER, after: 200 }),
      p([t("\u2500".repeat(60), { color: ACCENT_LIGHT, size: 20 })], { alignment: AlignmentType.CENTER, after: 200 }),
      p([t("Polyphonic Benchmark: 4-Chord Progression", { size: 24, italics: true, color: "555555" })], { alignment: AlignmentType.CENTER, after: 100 }),
      p([t("C major \u2192 G major \u2192 A minor \u2192 F major  (14 ground truth notes)", { size: 22, color: "555555" })], { alignment: AlignmentType.CENTER, after: 400 }),
      p([t("Iteration 1: Rule-Based (librosa pyin) \u2014 monophonic", { size: 22 })], { alignment: AlignmentType.CENTER, after: 60 }),
      p([t("Iteration 2: ML Model (CNN + BiGRU) \u2014 polyphonic", { size: 22 })], { alignment: AlignmentType.CENTER, after: 60 }),
      p([t("Iteration 3: Smart Pipeline (ML + music21) \u2014 polyphonic + key filter", { size: 22 })], { alignment: AlignmentType.CENTER, after: 400 }),
      p([t("March 2026", { size: 22, color: "888888" })], { alignment: AlignmentType.CENTER }),
      p([new PageBreak()]),

      // ═══════════ INTRODUCTION ═══════════
      heading("1. Introduction"),
      p([
        t("This document evaluates GuitarScribe\u2019s three transcription approaches on "),
        t("polyphonic chord audio", { bold: true }),
        t(" \u2014 a fundamentally harder task than monophonic scales. The test audio contains a 4-chord progression (C\u2013G\u2013Am\u2013F) synthesised with FluidSynth using the FluidR3 GM SoundFont at 22,050 Hz. Each chord sustains for 1.5 seconds with 0.2-second gaps. The 14 ground truth notes span MIDI 41\u201357 (F2\u2013A3 sounding pitch)."),
      ]),
      p([
        t("Why chords matter: ", { bold: true }),
        t("The pyin algorithm (iteration 1) tracks a single dominant pitch per analysis frame. When multiple notes sound simultaneously, it can only report one \u2014 typically the loudest or most harmonically dominant. The ML model (iterations 2 and 3), trained on polyphonic GuitarSet recordings, predicts per-string activations and should detect multiple concurrent notes. This test quantifies that advantage."),
      ]),

      // ═══════════ ITER 1 ═══════════
      heading("2. Iteration 1: Rule-Based (pyin)"),
      p([
        t("Result: ", { bold: true }),
        t(`${results.iteration1.metrics.tp}/14 true positives. `),
        t(`Precision: ${(results.iteration1.metrics.precision * 100).toFixed(1)}%, `, { color: RED }),
        t(`Recall: ${(results.iteration1.metrics.recall * 100).toFixed(1)}%, `, { color: RED }),
        t(`F1: ${(results.iteration1.metrics.f1 * 100).toFixed(1)}%`, { bold: true, color: RED }),
      ]),
      ...codeBlock([
        "$ python detect_pitches.py test/audioChords.wav",
        `  Detected: ${results.iteration1.detected_notes.join(", ")}`,
        `  Count: ${results.iteration1.note_count} notes (expected: 14)`,
        `  TP=${results.iteration1.metrics.tp}  FP=${results.iteration1.metrics.fp}  FN=${results.iteration1.metrics.fn}`,
      ]),
      p([]),
      p([t("Sheet Music Comparison:", { bold: true, color: ACCENT })]),
      p([new ImageRun({ data: img1, transformation: { width: 500, height: 134 }, type: "png" })], { alignment: AlignmentType.CENTER, after: 120 }),
      p([
        t("As expected, pyin captures only one pitch per frame. It locks onto the lowest or most energetic note in each chord, producing a monophonic melody that bears little resemblance to the polyphonic input."),
      ]),

      // ═══════════ ITER 2 ═══════════
      heading("3. Iteration 2: ML Model (CNN + BiGRU)"),
      p([
        t("Result: ", { bold: true }),
        t(`${results.iteration2.metrics.tp}/14 true positives. `),
        t(`Precision: ${(results.iteration2.metrics.precision * 100).toFixed(1)}%, `, { color: ORANGE }),
        t(`Recall: ${(results.iteration2.metrics.recall * 100).toFixed(1)}%, `, { color: GREEN }),
        t(`F1: ${(results.iteration2.metrics.f1 * 100).toFixed(1)}%`, { bold: true, color: ORANGE }),
      ]),
      ...codeBlock([
        "$ python -m model.predict test/audioChords.wav",
        `  Detected: ${results.iteration2.note_count} notes (expected: 14)`,
        `  TP=${results.iteration2.metrics.tp}  FP=${results.iteration2.metrics.fp}  FN=${results.iteration2.metrics.fn}`,
        `  Recall 5x higher than pyin (${(results.iteration2.metrics.recall * 100).toFixed(1)}% vs ${(results.iteration1.metrics.recall * 100).toFixed(1)}%)`,
      ]),
      p([]),
      p([t("Sheet Music Comparison:", { bold: true, color: ACCENT })]),
      p([new ImageRun({ data: img2, transformation: { width: 500, height: 136 }, type: "png" })], { alignment: AlignmentType.CENTER, after: 120 }),
      p([
        t("The ML model successfully identifies 10 of 14 ground truth notes across all four chords, demonstrating genuine polyphonic capability. The 4 missed notes (G2 in G major, A3 in Am, A2 and C3 in F major) are lower-register pitches that may be masked by upper harmonics. The 21 false positives include octave doublings (C4, E4), sympathetic string activations, and tail-end noise detections after the audio ends."),
      ]),

      // Page break
      p([new PageBreak()]),

      // ═══════════ ITER 3 ═══════════
      heading("4. Iteration 3: Smart Pipeline (ML + music21)"),
      p([
        t("Result: ", { bold: true }),
        t(`${results.iteration3.metrics.tp}/14 true positives. `),
        t(`Precision: ${(results.iteration3.metrics.precision * 100).toFixed(1)}%, `),
        t(`Recall: ${(results.iteration3.metrics.recall * 100).toFixed(1)}%, `),
        t(`F1: ${(results.iteration3.metrics.f1 * 100).toFixed(1)}%`, { bold: true }),
      ]),
      p([
        t("Key Detection: ", { bold: true }),
        t(`${results.iteration3.key_detected} (confidence: ${results.iteration3.key_confidence.toFixed(3)}). `),
        t(`Notes filtered: ${results.iteration3.filtered_notes}. `),
        t("The detected key of F major shares most pitch classes with the actual chord progression (C\u2013G\u2013Am\u2013F all use diatonic pitches), so the key filter does not remove any notes at tolerance=1."),
      ]),
      ...codeBlock([
        "$ python test/transcribe_smart.py test/audioChords.wav --backend ml",
        `  Key: ${results.iteration3.key_detected} (conf: ${results.iteration3.key_confidence.toFixed(3)})`,
        `  Raw: ${results.iteration3.raw_notes_count} → After filter: ${results.iteration3.note_count}`,
        `  TP=${results.iteration3.metrics.tp}  FP=${results.iteration3.metrics.fp}  FN=${results.iteration3.metrics.fn}`,
      ]),
      p([]),
      p([t("Sheet Music Comparison:", { bold: true, color: ACCENT })]),
      p([new ImageRun({ data: img3, transformation: { width: 530, height: 136 }, type: "png" })], { alignment: AlignmentType.CENTER, after: 120 }),

      // ═══════════ NOTE TABLE ═══════════
      heading("5. Note-by-Note Comparison"),
      p([
        t("Each row represents one ground truth note. "),
        t("YES", { bold: true, color: GREEN }),
        t(" = correctly detected within 50ms onset tolerance. "),
        t("MISS", { bold: true, color: RED }),
        t(" = not found by that approach."),
      ]),
      p([]),

      new Table({
        width: { size: 100, type: WidthType.PERCENTAGE },
        layout: TableLayoutType.FIXED,
        rows: [
          new TableRow({ children: [
            hCell("Chord"), hCell("GT Note"), hCell("Iter 1"), hCell("Iter 2"), hCell("Iter 3"),
          ]}),
          ...results.iteration1.note_matches.map((m1, i) => {
            const m2 = results.iteration2.note_matches[i];
            const m3 = results.iteration3.note_matches[i];
            // Determine chord name
            let chordName = "";
            if (i < 3) chordName = i === 0 ? "C major" : "";
            else if (i < 7) chordName = i === 3 ? "G major" : "";
            else if (i < 10) chordName = i === 7 ? "A minor" : "";
            else chordName = i === 10 ? "F major" : "";

            return new TableRow({ children: [
              dCell(chordName, { bold: true, color: ACCENT, align: AlignmentType.LEFT, shading: rowBg(i) }),
              dCell(m1.gt_note, { bold: true, shading: rowBg(i) }),
              m1.match ? yesCell() : missCell(),
              m2.match ? yesCell() : missCell(),
              m3.match ? yesCell() : missCell(),
            ]});
          }),
        ],
      }),

      p([]),

      // ═══════════ SUMMARY TABLE ═══════════
      heading("6. Summary"),

      new Table({
        width: { size: 100, type: WidthType.PERCENTAGE },
        layout: TableLayoutType.FIXED,
        rows: [
          new TableRow({ children: [hCell("Metric", 30), hCell("Iter 1 (pyin)", 23), hCell("Iter 2 (ML)", 23), hCell("Iter 3 (Smart)", 24)] }),
          ...[
            ["Notes Detected", results.iteration1.note_count, results.iteration2.note_count, results.iteration3.note_count],
            ["True Positives", results.iteration1.metrics.tp, results.iteration2.metrics.tp, results.iteration3.metrics.tp],
            ["False Positives", results.iteration1.metrics.fp, results.iteration2.metrics.fp, results.iteration3.metrics.fp],
            ["False Negatives", results.iteration1.metrics.fn, results.iteration2.metrics.fn, results.iteration3.metrics.fn],
            ["Precision", `${(results.iteration1.metrics.precision*100).toFixed(1)}%`, `${(results.iteration2.metrics.precision*100).toFixed(1)}%`, `${(results.iteration3.metrics.precision*100).toFixed(1)}%`],
            ["Recall", `${(results.iteration1.metrics.recall*100).toFixed(1)}%`, `${(results.iteration2.metrics.recall*100).toFixed(1)}%`, `${(results.iteration3.metrics.recall*100).toFixed(1)}%`],
            ["F1 Score", `${(results.iteration1.metrics.f1*100).toFixed(1)}%`, `${(results.iteration2.metrics.f1*100).toFixed(1)}%`, `${(results.iteration3.metrics.f1*100).toFixed(1)}%`],
            ["Key Detection", "N/A", "N/A", results.iteration3.key_detected],
            ["Polyphonic", "No", "Yes", "Yes"],
          ].map((row, i) => new TableRow({
            children: row.map((cell, j) => dCell(cell, {
              bold: j === 0,
              color: j === 0 ? ACCENT : "000000",
              align: j === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
              shading: rowBg(i),
            })),
          })),
        ],
      }),

      p([]),

      // ═══════════ ANALYSIS ═══════════
      heading("7. Analysis & Conclusion"),
      p([
        t("Polyphonic Capability Gap. ", { bold: true, color: ACCENT }),
        t("The chord benchmark reveals the fundamental limitation of the rule-based approach. With an F1 of 13.3%, pyin is effectively unable to transcribe polyphonic audio \u2014 it detects only 2 of 14 notes, both single bass notes that happened to dominate their respective chords. The ML model achieves 5\u00D7 higher recall (71.4% vs 14.3%), correctly identifying 10 of 14 notes across all four chords. This validates the core "),
        t("Analysis (S2) ", { bold: true, italics: true }),
        t("finding: monophonic algorithms are insufficient for real guitar music."),
      ]),
      p([
        t("Precision vs Recall Trade-off. ", { bold: true, color: ACCENT }),
        t("While the ML model\u2019s recall is strong, its precision (32.3%) indicates significant false positive generation \u2014 21 spurious notes including octave doublings, sympathetic string activations, and post-audio noise. This is a known challenge with frame-level sigmoid classifiers. For the "),
        t("Advise (S2) ", { bold: true, italics: true }),
        t("competency, the recommendation is clear: the ML model should be paired with a post-processing stage that deduplicates octave doublings and enforces temporal bounds (ignore detections outside the audio duration)."),
      ]),
      p([
        t("Smart Pipeline Limitations. ", { bold: true, color: ACCENT }),
        t("Iteration 3\u2019s key-based filtering was designed to remove chromatic errors, not to address the octave-doubling and noise issues that dominate the false positives here. The detected key (F major, confidence 0.689) is reasonable given the C\u2013G\u2013Am\u2013F progression. Since all false positives are diatonic to F major, the filter correctly retains them. This demonstrates a "),
        t("Realise (S2) ", { bold: true, italics: true }),
        t("insight: targeted post-processing layers must match the specific failure mode. A \"duplicate pitch suppression\" layer would be more effective here than key filtering."),
      ]),
      p([
        t("Iterative Problem-Solving (PS-2). ", { bold: true, color: ACCENT }),
        t("The three-iteration progression shows clear value: iteration 1 fails on polyphony (by design), iteration 2 adds polyphonic capability but introduces precision issues, and iteration 3\u2019s music-theory layer provides the right framework but needs refinement for this specific failure mode. The 4 missed notes (G2, A3, A2, C3) are consistently lower-register pitches, suggesting the CQT-based model has weaker sensitivity in the bass range \u2014 a finding that directly informs future training data augmentation."),
      ]),
      p([]),
      p([
        t("This polyphonic benchmark confirms that GuitarScribe\u2019s ML model is the correct approach for real-world guitar transcription. The next development priority should focus on reducing false positives through octave deduplication and temporal boundary enforcement, which would significantly improve precision while maintaining the strong recall.", { italics: true }),
      ]),
    ],
  }],
});

const buffer = await Packer.toBuffer(doc);
fs.writeFileSync("chord_comparison.docx", buffer);
console.log(`Generated: chord_comparison.docx (${(buffer.length / 1024).toFixed(1)} KB)`);
