\version "2.10.33"

\header {
  title = "48 - Ach wie nichtig, ach wie flüchtig"
  composer = "J. S. Bach"
}

global = {
  \time 4/4
  \key c \major
}

soprano = \relative c'' {
  a b c c
  c d e e
  e e d c
  b4. a8 a2
  e'4 e f f
  e e d d
  d d e e
  d d s c
  c d e d8 c
  b4 b a a
}

alto = \relative c' {
  e e e e
  e g g g
  g c4. b4 a8
  a4 gis e2
  a4 a a bes
  bes a8 g g e fis4
  g g g a
  a g8 f f d e4
  f f e e
  fis e8 d d b cis4
}

tenor = \relative c' {
  c b a a
  a b c c
  c8 d c b a b c e
  f4 e8 d cis2
  cis8 d e cis d4 d
  d cis d a
  b8 c d b c4 c
  c b c g
  a a b a
  a gis a e
}

baixo = \relative c {
  a' gis a a,
  a' g c c,
  c'8 b a g fis gis a4
  d, e a,2
  a'8 b cis a d c bes a
  g e a4 d, d
  g8 a b g c b a g
  f d g4 c, c
  f e8 d gis e a4
  dis, e a a,
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