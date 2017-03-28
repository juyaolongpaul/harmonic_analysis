\version "2.10.33"

\header {
  title = "215 - Verleih' uns Frieden gnädiglich"
  composer = "J. S. Bach"
}

global = {
  \time 4/4
  \key g \minor
}

soprano = \relative c'' {
  \partial 4 g4
  g g fis g8 a
  bes4 a g\fermata bes
  c c d bes
  c2 d4\fermata d
  d c d8 c bes4
  c8 bes a4 g\fermata g8 a
  bes4 bes c bes8 a
  g2 g4\fermata c
  d c g8 a bes4
  a8 g a4 g\fermata g
  g f bes c
  d8 c bes4 c c
  d2\fermata ees
  d4 c c4. bes8
  bes2.\fermata d4
  d bes c8 d ees4
  d2\fermata f4 d
  ees d8 c d4 d
  ees d f8 ees d4
  c2 bes\fermata
  d4 d bes c
  bes a g2\fermata
  f4 g8 a bes4 g
  g2.\fermata f4
  g a bes a
  g2~g4 fis
  g2.\fermata
}

alto = \relative c' {
  \partial 4 d4
  ees8 d c4 d d8 fis
  g4 fis d d
  f f f f
  g2 fis4 fis
  g g d d
  g4. fis8 d4 ees
  f8 ees d4 g f8 ees
  d f ees16 d ees8 d4 f
  f ees bes8 c d4
  d8 g4 fis8 d4 d
  e d d8 e f4
  f g g f
  f2 ees
  f4 g f8 d ees4
  d2. f8 g
  a4 g g g
  g2 f4 f
  bes,8 c d ees f4 f
  g8 f f ees d ees f4
  f4. ees8 d2
  f8 g a4 d, c
  g'4. fis8 g2
  d8 c bes c d c d4
  c2. a4
  bes8 c16 d ees4 d4. ees16 f
  g4. f16 ees d2
  d2.
}

tenor = \relative c' {
  \partial 4 bes4
  bes a8 g a4 bes8 c
  d4 d8 c bes4 bes
  a a bes bes
  bes a8 g a4 a
  bes c8 bes a4 bes8 a
  g4 d' bes bes8 c
  d c bes a g a bes c
  d4 c16 b c8 b4 c
  bes?8 a g4 g f8 g
  a bes c4 bes bes
  a a g a
  bes8 c d4 g,8 a16 bes a4
  bes2 bes
  bes4 bes bes a
  f2. bes4
  a d g, c
  b2 c4 bes8 aes
  g4 f bes8 aes4 g8
  g a? bes4 bes2~bes8 g a4 f2
  bes4 a bes8 a g4
  ees'4. d16 c bes2
  bes8 a g4 f8 ees f4
  ees2. f4 
  ees4. d16 ees f8 g a4
  b8 c16 d g,8 c bes8 a16 g a4
  b2.
}

baixo = \relative c {
  \partial 4 g'4
  c,8 d ees4 d8 c bes a
  g4 d' g,\fermata g'
  f8 ees d c bes c d bes
  ees2 d4\fermata d
  g8 f e4 fis g8 f
  ees c d4 g,\fermata ees'
  d4 g8 f ees4 d8 c
  b4 c g\fermata a
  bes4 c8 d ees4 d8 e
  fis g d4 g,\fermata g'
  cis, d g f
  bes8 a g f ees4 f
  bes,2\fermata g'
  f4 ees f f,
  bes2.\fermata bes'4
  fis g8 f ees d c4
  g2\fermata a4 bes
  bes( bes bes bes
  bes) bes8 c d c d ees
  f4 f, bes2\fermata
  bes'8 a g fis g f ees d
  ees d c d ees2\fermata
  bes4 ees d c8 b
  c2.\fermata d4~d c d8 ees f4~f8 ees16 d ees8 c d2
  g,2.\fermata
}
\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}