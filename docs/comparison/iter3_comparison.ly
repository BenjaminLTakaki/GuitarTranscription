\version "2.24.0"

\header {
  tagline = ##f
}

\paper {
  indent = 0\mm
  top-margin = 5\mm
  bottom-margin = 5\mm
  left-margin = 10\mm
  right-margin = 10\mm
  system-system-spacing.basic-distance = #12
}

\score {
  <<
    \new Staff \with {
      instrumentName = "Ground Truth"
      shortInstrumentName = "GT"
    } {
      \clef "treble_8"
      \key c \major
      \time 4/4
      c'4 d'4 e'4 f'4 |
      g'4 a'4 b'4 c''4 |
    }
    \new Staff \with {
      instrumentName = "Iteration 3"
      shortInstrumentName = "Smart"
    } {
      \clef "treble_8"
      \key c \major
      \time 4/4
      c'4 c''4 d'4 e'4 |
      f'4 g'4 g'4 a'4 |
      b'4 b'4 c''4
    }
  >>
  \layout {
    \context {
      \Score
      \override SpacingSpanner.common-shortest-duration = \musicLength 4
    }
  }
}
