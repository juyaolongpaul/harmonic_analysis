\version "2.10.33"

\header {
  title = "81 - Christus, der uns selig macht"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c'' {
  e4 e4 e4 e4 
  d4 c4 b2 
  c4 d4 e4 e4 
  d4 c4 b2 
  a4 b4 c4 a4   %5
  a8 g8 f4 e2 
  f4 g4 a4 a4 
  g4 f4 e2 
  a4 gis4 a4 b4 
  c4 b4 a2   %10
  c4 d4 e4 e4 
  d4 c4 b2 
  a4 b4 c4 a4 
  a8 g8 f4 e2 
  f4 g4 a4 a4   %15
  g4 f4 e2~ 
  e1 
  
}

alto = \relative c'' {
  gis4 gis4 a4 b4 
  a4 a4 gis2 
  e8 fis8 g4 g4 f8 e8 
  f4 e4 e2 
  e4 e4 e4 f4   %5
  e4 d4 cis2 
  d4 e4 ees4 d4 
  d2 cis2 
  d4 e4 e4 fis8 gis8 
  a4 g4 fis2   %10
  g4 g4 g4 f4 
  f4 e8 fis8 g2 
  a4 e4 e4 e4 
  a,4 d4 cis2 
  d4 e4 f4 f4   %15
  e4 d2 c4 
  b1 
  
}

tenor = \relative c' {
  b4 b4 c4 b8 c8 
  d8 e8 f4 b,2 
  a4 bes4 bes4 a4 
  a2 gis2 
  c4 b4 a8 b8 c8 d8   %5
  e4 a,4 a2 
  a4 c4 c4 c4 
  bes8 c8 d4 a2 
  a4 b4 c4 d4 
  e4 d4 d2   %10
  c4 bes4 a4 a4 
  a8 b8 c4 d2 
  d8 c8 b4 a8 b8 c8 d8 
  e4 a,4 a2 
  a4 c4 c4 c4   %15
  c8 b8 a4 gis4 a4 
  gis1 
  
}

baixo = \relative c {
  e4 e4 a4 g4 
  f4 e8 d8 e2 
  a4 g4 cis,4 cis4 
  d4 a4 e'2 
  a4 gis4 a4 f4   %5
  cis4 d4 a2 
  d4 c4 f,4 fis4 
  g4 gis4 a2 
  f'4 e8 d8 c4 b4 
  a4 b8 c8 d2   %10
  ees4 d4 cis4 d8 e8 
  f8 g8 a4 g2 
  fis4 gis4 a4 a,8 b8 
  cis4 d4 a2 
  d4 c4 f,8 g8 a8 b8   %15
  c4 d4 e2~ 
  e1 
  
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