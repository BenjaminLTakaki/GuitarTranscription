import {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, WidthType, BorderStyle, HeadingLevel, ImageRun,
  Header, Footer, PageNumber, NumberFormat, ShadingType,
  TableLayoutType, VerticalAlign, PageBreak,
} from "docx";
import * as fs from "fs";

// ── Style constants (matching iteration1_performance_analysis.pdf) ──────────
const ACCENT = "1A5276";
const ACCENT_LIGHT = "D4E6F1";
const GREEN = "27AE60";
const RED = "E74C3C";
const ORANGE = "E67E22";
const GRAY_BG = "F8F9FA";
const FONT = "Calibri";
const CODE_FONT = "Consolas";

// ── Load results ────────────────────────────────────────────────────────────
const results = JSON.parse(fs.readFileSync("comparison_results.json", "utf-8"));

// ── Load images ─────────────────────────────────────────────────────────────
const img1 = fs.readFileSync("iter1_comparison_cropped.png");
const img2 = fs.readFileSync("iter2_comparison_cropped.png");
const img3 = fs.readFileSync("iter3_comparison_cropped.png");

// ── Helper functions ────────────────────────────────────────────────────────
function makeText(text, opts = {}) {
  return new TextRun({
    text,
    font: opts.font || FONT,
    size: opts.size || 22,  // 11pt
    bold: opts.bold || false,
    italics: opts.italics || false,
    color: opts.color || "000000",
    ...opts,
  });
}

function makeParagraph(runs, opts = {}) {
  const runArray = Array.isArray(runs) ? runs : [runs];
  return new Paragraph({
    children: runArray.map(r => typeof r === "string" ? makeText(r) : r),
    spacing: { after: opts.afterSpacing || 120, before: opts.beforeSpacing || 0 },
    alignment: opts.alignment || AlignmentType.LEFT,
    ...opts,
  });
}

function heading(text, level = HeadingLevel.HEADING_2) {
  return new Paragraph({
    children: [makeText(text, { bold: true, color: ACCENT, size: level === HeadingLevel.HEADING_1 ? 32 : 26 })],
    heading: level,
    spacing: { before: 240, after: 120 },
  });
}

function codeBlock(lines) {
  return lines.map(line => new Paragraph({
    children: [makeText(line, { font: CODE_FONT, size: 18 })],
    spacing: { after: 0, before: 0 },
    shading: { type: ShadingType.CLEAR, fill: GRAY_BG },
    indent: { left: 200 },
  }));
}

function yesNo(match) {
  return makeText(match ? "YES" : "NO", {
    bold: true,
    color: match ? GREEN : RED,
  });
}

function cellShading(match) {
  return { type: ShadingType.CLEAR, fill: match ? "D5F5E3" : "FADBD8" };
}

function tableCell(children, opts = {}) {
  const childArray = Array.isArray(children) ? children : [children];
  return new TableCell({
    children: childArray.map(c => {
      if (c instanceof Paragraph) return c;
      if (c instanceof TextRun) return new Paragraph({ children: [c], spacing: { after: 0, before: 0 } });
      return new Paragraph({ children: [makeText(String(c))], spacing: { after: 0, before: 0 } });
    }),
    verticalAlign: VerticalAlign.CENTER,
    shading: opts.shading,
    width: opts.width ? { size: opts.width, type: WidthType.PERCENTAGE } : undefined,
    borders: opts.headerCell ? {
      bottom: { style: BorderStyle.SINGLE, size: 2, color: ACCENT },
    } : undefined,
  });
}

function headerCell(text, width) {
  return tableCell(
    [new Paragraph({
      children: [makeText(text, { bold: true, color: "FFFFFF", size: 20 })],
      spacing: { after: 0, before: 0 },
      alignment: AlignmentType.CENTER,
    })],
    {
      shading: { type: ShadingType.CLEAR, fill: ACCENT },
      width,
    }
  );
}

// ── Build document ──────────────────────────────────────────────────────────

const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: FONT, size: 22 },
      },
    },
  },
  sections: [
    // ════════════════════════════════════════════════════════════════════
    // TITLE PAGE
    // ════════════════════════════════════════════════════════════════════
    {
      properties: {
        page: {
          margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
        },
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            children: [makeText("Three-Way Performance Comparison | GuitarScribe", {
              size: 18, color: ACCENT, italics: true,
            })],
            alignment: AlignmentType.RIGHT,
          })],
        }),
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            children: [
              new TextRun({
                children: [PageNumber.CURRENT],
                font: FONT, size: 18, color: "666666",
              }),
            ],
            alignment: AlignmentType.CENTER,
          })],
        }),
      },
      children: [
        // Spacer
        ...Array(6).fill(null).map(() => new Paragraph({ children: [], spacing: { after: 200 } })),

        // Title
        new Paragraph({
          children: [makeText("GuitarScribe", { size: 56, bold: true, color: ACCENT })],
          alignment: AlignmentType.CENTER,
          spacing: { after: 100 },
        }),
        new Paragraph({
          children: [makeText("Three-Way Performance Comparison", { size: 36, color: ACCENT })],
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
        }),
        // Horizontal rule effect
        new Paragraph({
          children: [makeText("─".repeat(60), { color: ACCENT_LIGHT, size: 20 })],
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
        }),
        // Subtitle info
        new Paragraph({
          children: [makeText("C Major Scale Benchmark (audioCMajor.mp3)", { size: 24, italics: true, color: "555555" })],
          alignment: AlignmentType.CENTER,
          spacing: { after: 100 },
        }),
        new Paragraph({
          children: [makeText("Ground Truth: C4 – D4 – E4 – F4 – G4 – A4 – B4 – C5 (written pitch)", { size: 22, color: "555555" })],
          alignment: AlignmentType.CENTER,
          spacing: { after: 400 },
        }),
        // Three approaches listed
        new Paragraph({
          children: [makeText("Iteration 1: Rule-Based (librosa pyin)", { size: 22 })],
          alignment: AlignmentType.CENTER,
          spacing: { after: 60 },
        }),
        new Paragraph({
          children: [makeText("Iteration 2: ML Model (CNN + BiGRU)", { size: 22 })],
          alignment: AlignmentType.CENTER,
          spacing: { after: 60 },
        }),
        new Paragraph({
          children: [makeText("Iteration 3: Smart Pipeline (ML + music21)", { size: 22 })],
          alignment: AlignmentType.CENTER,
          spacing: { after: 400 },
        }),
        // Date
        new Paragraph({
          children: [makeText("March 2026", { size: 22, color: "888888" })],
          alignment: AlignmentType.CENTER,
        }),
        // Page break
        new Paragraph({
          children: [new PageBreak()],
        }),

        // ════════════════════════════════════════════════════════════════
        // INTRODUCTION
        // ════════════════════════════════════════════════════════════════
        heading("1. Introduction", HeadingLevel.HEADING_1),
        makeParagraph([
          makeText("This document compares GuitarScribe's three transcription approaches against a known ground truth: "),
          makeText("a C major scale played on acoustic guitar", { italics: true }),
          makeText(" (8 quarter notes, C4\u2013C5 in written pitch). The guitar uses a treble 8vb clef, meaning sounding pitches are one octave lower (C3\u2013C4). Each approach processes the same 44.1 kHz MP3 recording."),
        ]),

        heading("Approach Overview", HeadingLevel.HEADING_2),
        makeParagraph([
          makeText("Iteration 1 \u2014 Rule-Based (librosa pyin): ", { bold: true }),
          makeText("Monophonic pitch detection using librosa\u2019s probabilistic YIN algorithm. Frames are grouped into note segments with a minimum 50ms duration filter. Detects sounding pitch, requiring +12 semitone offset for written pitch."),
        ]),
        makeParagraph([
          makeText("Iteration 2 \u2014 ML Model (CNN + BiGRU): ", { bold: true }),
          makeText("A convolutional neural network with bidirectional GRU trained on GuitarSet (300 epochs, F1 = 0.746). Predicts tablature classes (6 strings \u00D7 21 frets = 126 classes) from CQT spectrograms. Uses Schmitt-trigger post-processing with onset/sustain dual thresholds."),
        ]),
        makeParagraph([
          makeText("Iteration 3 \u2014 Smart Pipeline (ML + music21): ", { bold: true }),
          makeText("Combines the ML model\u2019s raw output with music21\u2019s Krumhansl-Schmuckler key detection algorithm and scale-based note filtering. Adds guitar fingering optimisation to assign ergonomic (string, fret) positions."),
        ]),

        // Page break before comparisons
        new Paragraph({ children: [new PageBreak()] }),

        // ════════════════════════════════════════════════════════════════
        // ITERATION 1 COMPARISON
        // ════════════════════════════════════════════════════════════════
        heading("2. Iteration 1: Rule-Based (pyin)", HeadingLevel.HEADING_1),
        makeParagraph([
          makeText("Result: ", { bold: true }),
          makeText(`${results.iteration1.correct}/8 notes correct (${results.iteration1.accuracy}% accuracy). `),
          makeText(`Detected ${results.iteration1.note_count} notes total.`),
        ]),
        ...codeBlock([
          "$ python detect_pitches.py audioCMajor.mp3 -o output/iter1_pyin.mid",
          `  Sounding: ${results.iteration1.sounding_notes.join(", ")}`,
          `  Written:  ${results.iteration1.written_notes.join(", ")}`,
          `  Count:    ${results.iteration1.note_count} notes`,
        ]),
        new Paragraph({ children: [], spacing: { after: 120 } }),

        // Sheet music image
        makeParagraph([makeText("Sheet Music Comparison (Ground Truth vs Detected):", { bold: true, color: ACCENT })]),
        new Paragraph({
          children: [new ImageRun({
            data: img1,
            transformation: { width: 400, height: 230 },
            type: "png",
          })],
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
        }),

        // ════════════════════════════════════════════════════════════════
        // ITERATION 2 COMPARISON
        // ════════════════════════════════════════════════════════════════
        heading("3. Iteration 2: ML Model (CNN + BiGRU)", HeadingLevel.HEADING_1),
        makeParagraph([
          makeText("Result: ", { bold: true }),
          makeText(`${results.iteration2.correct}/8 ground truth pitches found (${results.iteration2.accuracy}% pitch accuracy). `),
          makeText(`Detected ${results.iteration2.note_count} notes total `),
          makeText(`(${results.iteration2.extra_notes.length} extra notes: ${results.iteration2.extra_notes.join(", ")}).`, { color: ORANGE }),
        ]),
        ...codeBlock([
          "$ python -m model.predict audioCMajor.mp3 -o output/iter2_ml.mid",
          `  Sounding: ${results.iteration2.sounding_notes.join(", ")}`,
          `  Written:  ${results.iteration2.written_notes.join(", ")}`,
          `  Count:    ${results.iteration2.note_count} notes (8 correct + 3 extra)`,
        ]),
        new Paragraph({ children: [], spacing: { after: 120 } }),

        makeParagraph([makeText("Sheet Music Comparison:", { bold: true, color: ACCENT })]),
        new Paragraph({
          children: [new ImageRun({
            data: img2,
            transformation: { width: 490, height: 230 },
            type: "png",
          })],
          alignment: AlignmentType.CENTER,
          spacing: { after: 120 },
        }),
        makeParagraph([
          makeText("Note: ", { bold: true }),
          makeText("The ML model detects all 8 ground truth pitches, plus 3 duplicate activations on adjacent strings (G3 on string 2 fret 5, B3 on string 4 fret 0, and C4 on string 4 fret 1). These are valid guitar positions for the same pitches, indicating the polyphonic model picks up sympathetic resonance."),
        ], { afterSpacing: 200 }),

        // Page break
        new Paragraph({ children: [new PageBreak()] }),

        // ════════════════════════════════════════════════════════════════
        // ITERATION 3 COMPARISON
        // ════════════════════════════════════════════════════════════════
        heading("4. Iteration 3: Smart Pipeline (ML + music21)", HeadingLevel.HEADING_1),
        makeParagraph([
          makeText("Result: ", { bold: true }),
          makeText(`${results.iteration3.correct}/8 ground truth pitches found (${results.iteration3.accuracy}% pitch accuracy). `),
          makeText(`Detected ${results.iteration3.note_count} notes total. `),
        ]),
        makeParagraph([
          makeText("Key Detection: ", { bold: true }),
          makeText(`${results.iteration3.key_detected} (confidence: ${results.iteration3.key_confidence.toFixed(3)}). `),
          makeText(`Notes filtered: ${results.iteration3.filtered_notes} (tolerance: \u00B11 semitone).`),
        ]),
        ...codeBlock([
          "$ python test/transcribe_smart.py audioCMajor.mp3 --backend ml",
          `  Detected key:  ${results.iteration3.key_detected}`,
          `  Raw notes:     ${results.iteration3.raw_notes_count}`,
          `  After filter:  ${results.iteration3.note_count} (dropped ${results.iteration3.filtered_notes})`,
          `  Written:       ${results.iteration3.written_notes.join(", ")}`,
        ]),
        new Paragraph({ children: [], spacing: { after: 120 } }),

        makeParagraph([makeText("Sheet Music Comparison:", { bold: true, color: ACCENT })]),
        new Paragraph({
          children: [new ImageRun({
            data: img3,
            transformation: { width: 490, height: 230 },
            type: "png",
          })],
          alignment: AlignmentType.CENTER,
          spacing: { after: 120 },
        }),
        makeParagraph([
          makeText("Note: ", { bold: true }),
          makeText(`The smart pipeline detects the key as "${results.iteration3.key_detected}" (the relative minor of C major, sharing the same pitch classes). With tolerance=1, all notes from the ML model pass the filter since every detected pitch belongs to the A natural minor / C major scale. The 3 extra notes persist.`),
        ], { afterSpacing: 200 }),

        // Page break
        new Paragraph({ children: [new PageBreak()] }),

        // ════════════════════════════════════════════════════════════════
        // NOTE-BY-NOTE COMPARISON TABLE
        // ════════════════════════════════════════════════════════════════
        heading("5. Note-by-Note Comparison", HeadingLevel.HEADING_1),
        makeParagraph([
          makeText("The table below compares all three approaches against the 8-note ground truth. For each approach, \u201c"),
          makeText("YES", { bold: true, color: GREEN }),
          makeText("\u201d indicates the pitch was correctly detected and \u201c"),
          makeText("NO", { bold: true, color: RED }),
          makeText("\u201d indicates a mismatch or missing note."),
        ]),
        new Paragraph({ children: [], spacing: { after: 120 } }),

        // Build the comparison table
        new Table({
          width: { size: 100, type: WidthType.PERCENTAGE },
          layout: TableLayoutType.FIXED,
          rows: [
            // Header row
            new TableRow({
              children: [
                headerCell("#", 6),
                headerCell("Ground Truth", 14),
                headerCell("Iter 1 (pyin)", 16),
                headerCell("Match", 10),
                headerCell("Iter 2 (ML)", 16),
                headerCell("Match", 10),
                headerCell("Iter 3 (Smart)", 16),
                headerCell("Match", 10),
              ],
            }),
            // Data rows
            ...Array.from({ length: 8 }, (_, i) => {
              const m1 = results.iteration1.matches[i];
              const m2 = results.iteration2.matches[i];
              const m3 = results.iteration3.matches[i];
              const rowBg = i % 2 === 0 ? "FFFFFF" : "F2F3F4";

              return new TableRow({
                children: [
                  tableCell([new Paragraph({
                    children: [makeText(`${i + 1}`, { bold: true, size: 20 })],
                    alignment: AlignmentType.CENTER,
                    spacing: { after: 0, before: 0 },
                  })], { shading: { type: ShadingType.CLEAR, fill: rowBg } }),

                  tableCell([new Paragraph({
                    children: [makeText(m1.expected, { bold: true, size: 20 })],
                    alignment: AlignmentType.CENTER,
                    spacing: { after: 0, before: 0 },
                  })], { shading: { type: ShadingType.CLEAR, fill: rowBg } }),

                  tableCell([new Paragraph({
                    children: [makeText(m1.detected, { size: 20 })],
                    alignment: AlignmentType.CENTER,
                    spacing: { after: 0, before: 0 },
                  })], { shading: { type: ShadingType.CLEAR, fill: rowBg } }),

                  tableCell([new Paragraph({
                    children: [yesNo(m1.match)],
                    alignment: AlignmentType.CENTER,
                    spacing: { after: 0, before: 0 },
                  })], { shading: cellShading(m1.match) }),

                  tableCell([new Paragraph({
                    children: [makeText(m2.detected, { size: 20 })],
                    alignment: AlignmentType.CENTER,
                    spacing: { after: 0, before: 0 },
                  })], { shading: { type: ShadingType.CLEAR, fill: rowBg } }),

                  tableCell([new Paragraph({
                    children: [yesNo(m2.match)],
                    alignment: AlignmentType.CENTER,
                    spacing: { after: 0, before: 0 },
                  })], { shading: cellShading(m2.match) }),

                  tableCell([new Paragraph({
                    children: [makeText(m3.detected, { size: 20 })],
                    alignment: AlignmentType.CENTER,
                    spacing: { after: 0, before: 0 },
                  })], { shading: { type: ShadingType.CLEAR, fill: rowBg } }),

                  tableCell([new Paragraph({
                    children: [yesNo(m3.match)],
                    alignment: AlignmentType.CENTER,
                    spacing: { after: 0, before: 0 },
                  })], { shading: cellShading(m3.match) }),
                ],
              });
            }),
          ],
        }),

        new Paragraph({ children: [], spacing: { after: 200 } }),

        // ════════════════════════════════════════════════════════════════
        // SUMMARY TABLE
        // ════════════════════════════════════════════════════════════════
        heading("6. Summary", HeadingLevel.HEADING_1),

        new Table({
          width: { size: 100, type: WidthType.PERCENTAGE },
          layout: TableLayoutType.FIXED,
          rows: [
            new TableRow({
              children: [
                headerCell("Metric", 30),
                headerCell("Iter 1 (pyin)", 23),
                headerCell("Iter 2 (ML)", 23),
                headerCell("Iter 3 (Smart)", 24),
              ],
            }),
            ...[
              ["Notes Detected", String(results.iteration1.note_count), String(results.iteration2.note_count), String(results.iteration3.note_count)],
              ["Expected Notes", "8", "8", "8"],
              ["Correct Pitches", `${results.iteration1.correct}/8`, `${results.iteration2.correct}/8`, `${results.iteration3.correct}/8`],
              ["Extra Notes", String(results.iteration1.extra_notes.length), String(results.iteration2.extra_notes.length), String(results.iteration3.extra_notes.length)],
              ["Pitch Accuracy", `${results.iteration1.accuracy.toFixed(1)}%`, `${results.iteration2.accuracy.toFixed(1)}%`, `${results.iteration3.accuracy.toFixed(1)}%`],
              ["Key Detection", "N/A", "N/A", results.iteration3.key_detected],
              ["Polyphonic", "No", "Yes (6-string)", "Yes (6-string)"],
              ["Post-Processing", "Min duration filter", "Schmitt trigger", "Schmitt + key filter"],
            ].map((row, i) => new TableRow({
              children: row.map((cell, j) => tableCell(
                [new Paragraph({
                  children: [makeText(cell, {
                    size: 20,
                    bold: j === 0,
                    color: j === 0 ? ACCENT : "000000",
                  })],
                  alignment: j === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
                  spacing: { after: 0, before: 0 },
                })],
                { shading: { type: ShadingType.CLEAR, fill: i % 2 === 0 ? "FFFFFF" : "F2F3F4" } }
              )),
            })),
          ],
        }),

        new Paragraph({ children: [], spacing: { after: 200 } }),

        // ════════════════════════════════════════════════════════════════
        // ANALYSIS & CONCLUSION
        // ════════════════════════════════════════════════════════════════
        heading("7. Analysis & Conclusion", HeadingLevel.HEADING_1),

        makeParagraph([
          makeText("Pitch Detection Accuracy. ", { bold: true, color: ACCENT }),
          makeText("All three approaches successfully identify the 8 ground truth pitches of the C major scale, achieving 100% pitch accuracy. This confirms that even the simplest rule-based approach (pyin) performs flawlessly on clean, monophonic single-note recordings \u2014 a useful baseline for the "),
          makeText("Analysis (S2) ", { bold: true, italics: true }),
          makeText("competency, establishing that evaluation must use increasingly complex audio to differentiate approach quality."),
        ]),

        makeParagraph([
          makeText("False Positives & Polyphonic Leakage. ", { bold: true, color: ACCENT }),
          makeText("The ML-based approaches (iterations 2 and 3) detect 3 extra notes beyond the 8 ground truth pitches. These are duplicate activations of the same pitches (G3, B3, C4) on adjacent guitar strings, reflecting the polyphonic model\u2019s sensitivity to sympathetic string resonance. The pyin approach, being inherently monophonic, avoids this issue entirely. This trade-off between recall and precision illustrates the "),
          makeText("Advise (S2) ", { bold: true, italics: true }),
          makeText("competency: recommending pyin for simple monophonic passages and the ML model for polyphonic scenarios."),
        ]),

        makeParagraph([
          makeText("Smart Pipeline Effectiveness. ", { bold: true, color: ACCENT }),
          makeText("Iteration 3\u2019s music21 key detection correctly identifies the key as "),
          makeText("A minor", { italics: true }),
          makeText(" (the relative minor of C major, sharing identical pitch classes). With a tolerance of \u00B11 semitone, no notes are filtered \u2014 all detected pitches belong to the A natural minor / C major scale. On this benchmark, the smart pipeline\u2019s key-based filtering does not reduce false positives because the extra notes are harmonically valid. This demonstrates an important insight for "),
          makeText("Realise (S2)", { bold: true, italics: true }),
          makeText(": key filtering is most effective for removing chromatic errors, not sympathetic resonance duplicates. A future deduplication step (grouping simultaneous activations of the same pitch across strings) would address this."),
        ]),

        makeParagraph([
          makeText("Problem-Solving Reflection (PS-2). ", { bold: true, color: ACCENT }),
          makeText("The three-iteration approach follows a structured problem-solving methodology. Iteration 1 establishes a simple, reliable baseline. Iteration 2 adds machine learning for polyphonic capability but introduces new failure modes (duplicate activations). Iteration 3 applies domain knowledge (music theory) as a corrective layer. The C major scale benchmark reveals that each iteration addresses specific shortcomings while sometimes introducing others \u2014 a characteristic pattern in iterative system design. The next step is to run these comparisons on polyphonic audio (chords, arpeggios) where the ML model\u2019s strengths over pyin become more pronounced."),
        ]),

        new Paragraph({ children: [], spacing: { after: 120 } }),

        makeParagraph([
          makeText("Overall, this benchmark validates that GuitarScribe\u2019s pitch detection core is accurate for simple monophonic passages across all three implementation strategies. The differentiating factor for real-world use will be polyphonic performance, where the ML-based approaches (iterations 2 and 3) are expected to significantly outperform the rule-based approach.", { italics: true }),
        ]),
      ],
    },
  ],
});

// ── Generate ────────────────────────────────────────────────────────────────
const buffer = await Packer.toBuffer(doc);
fs.writeFileSync("three_way_comparison.docx", buffer);
console.log("Generated: three_way_comparison.docx");
console.log(`File size: ${(buffer.length / 1024).toFixed(1)} KB`);
