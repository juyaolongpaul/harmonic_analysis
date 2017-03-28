\version "2.10.33"

\header {
  title = "127 - Dies sind die heiligen zehn Gebot"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key c \major
}


soprano = \relative c'' {
  \partial 4 g4 
    g4 g4 g4 g4 
  g8 a8 b4 c4 c4 
  d4 d4 c4 g4 
  g8 a8 b4 c4 c4 
  d4 d4 c4 g4   %5
  a4 g4 f4 g4 
  a4 b4 c2 
  ais8 a8 g8 a8 ais4 a4 
  g4 g4 g2
}

alto = \relative c' {
  \partial 4 d4 
    e4 f4 e4 e4 
  d4. e16 f16 e4 f8 g8 
  a8 g8 f4 f4. e8 
  d4 g8 f8 e4 f4 
  f4 f4 f4 e4   %5
  e8 f8 e4 d4 e4 
  f4 f8 e8 e2 
  fis4 g8 d8 d4 f4 
  f4 e4 d2 
}

tenor = \relative c' {
  \partial 4 b4 
    c4 d4 d4 c4 
  b8 a8 g4 g4 a8 c8 
  f4 a,8 b8 c4 c4 
  b8 a8 g4 g4 a4 
  a4 d4 g,4 c8 ais8   %5
  a4 ais8 a8 a4 c4 
  c8 d8 d4 c2 
  d8 c8 ais8 a8 g4 a4 
  d4. c8 c8 b16 a16 b4
}

baixo = \relative c' {
  \partial 4 g8  f8 
    e8 d8 c8 b8 c8 d8 e8 f8 
  g4 g,4 c4 f8 e8 
  d8 e8 f8 g8 a8 b8 c4 
  g8 f8 e8 d8 c4 f8 e8 
  d8 c4 b8 c8 d8 e8 d8   %5
  cis8 d4 cis8 d4 c8 b8 
  a8 a'4 gis8 a2 
  d,4 e8 fis8 g4 d8 c8 
  b8 g8 c4 g2
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