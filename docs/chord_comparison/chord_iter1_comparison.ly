\version "2.24.0"

\header {
  title = "Chord Benchmark — Iteration 1 (pyin)"
  tagline = ##f
}

\paper {
  indent = 0\mm
  top-margin = 5\mm
  bottom-margin = 5\mm
  left-margin = 10\mm
  right-margin = 10\mm
}

\score {
  <<
    \new Staff \with { instrumentName = "Ground Truth" } {
      \clef "treble_8"
      \time 4/4
      <c e g>4 <g, b, d g>4 <a, e a>4 <f, a, c f>4
      \bar "|."
    }
    \new Staff \with { instrumentName = "Detected" } {
      \clef "treble_8"
      \time 4/4
      e,4 g4 gis4 g4 |
      a,4 e,4 e,4 f,4 |
      f4 f,4 f4 f,4 |
      f4 f,4 f4 f,4
      \bar "|."
    }
  >>
  \layout { }
}
