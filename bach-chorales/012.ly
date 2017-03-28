\version "2.10.33"

\header {
  title = "12 - Puer natus in Bethlehem"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key c \major
}


soprano = \relative c'' {
  \partial 4 a4 
    a2 a4 
  b2 b4 
  c4 b4 a4 
  g2. 
  c2 d4   %5
  c2 b4 
  c2 c4 
  c2 c4 
  b2 a4 
  a4 b4 gis4   %10
  a2 a4 
  g2 a4 
  b2 b4 
  c2 b4 
  a2 gis4   %15
  a2 
}

alto = \relative c' {
  \partial 4 e4 
    e2 a4 
  g2 g4 
  g2 f4 
  e2. 
  g2 gis4   %5
  a2 g4 
  g2 g4 
  a2 fis4 
  g2 g4~ 
  g4 fis4 e4   %10
  e2 d4 
  d4 e4 d4 
  d2 f4 
  e4 f4 g4~ 
  g4 fis4 e4   %15
  e2 
}

tenor = \relative c' {
  \partial 4 c4 
    c2 d4 
  d2 d4 
  c4 e4 c4 
  c2. 
  e2.~   %5
  e4 d4 d4 
  e2 e4 
  e4 d4 d4 
  d4 e4 e4 
  d4 b4 b4   %10
  c2 a4 
  b4 c4 a4 
  g4 b4 d4 
  c4 d2 
  c2 b4   %15
  cis2 
}

baixo = \relative c {
  \partial 4 a4 
    a'4 g4 fis4 
  g4 d4 f4 
  e4 c4 f4 
  c2. 
  c4 c'4 b4   %5
  a4 fis4 g4 
  c,2 c'4 
  a4 fis4 d4 
  g4 e4 cis4 
  d4 dis4 e4   %10
  a,2 f'4~ 
  f4 e4 fis4 
  g4 fis4 gis4 
  a4 d,4 e4 
  f4 dis4 e4   %15
  a,2 
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