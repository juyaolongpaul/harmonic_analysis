\version "2.10.33"

\header {
  title = "83 - Jesu Leiden, Pein und Tod"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \major
}


soprano = \relative c'' {
  cis4 b4 a4 b4 
  cis4 d4 e2 
  fis4 e4 d4 cis8 b8 
  b2 a2 
  cis4 e4 d4 cis4   %5
  b4 a4 gis2 
  a4 b4 cis4 cis8 d16 e16 
  d2 cis2 
  cis8 b8 cis8 d8 e4 d4 
  cis4 b4 cis2   %10
  b4 b4 cis8 dis8 e4 
  e4 dis4 e2 
  e4 cis4 fis4. e8 
  d4 cis8 b8 b2 
  cis4 cis8 d16 e16 d4 cis8 b8   %15
  b2 a2 
  
}

alto = \relative c'' {
  a4 gis4 fis4 e4 
  e4 fis4 g2 
  fis4 gis4 a8 b8 e,8 a8 
  a4 gis4 e2 
  a4 b4 a4 a4   %5
  gis4 fis4 eis2 
  fis4 e4 e4 fis4 
  fis8 a8 gis8 fis8 gis2 
  fis4 fis4 cis8 gis'8 a8 b8 
  a4 gis4 a2   %10
  fis4 e4 a4 b4 
  cis4 b8 a8 gis2 
  gis4 a4 a4 g4 
  g8 fis8 e4 d2 
  fis4 e8 fis8 gis4 a4   %15
  a8 fis8 gis4 e2 
  
}

tenor = \relative c' {
  fis4 cis4 cis4 b4 
  a4 a4 a2 
  d4 cis4 cis8 b8 cis8 d8 
  e4 d8 cis8 cis2 
  e4 e4 fis4 e4   %5
  d4 cis4 cis2 
  cis4 b4 a8 b8 cis4 
  cis4 b8 a8 b2 
  a4 a'4 gis4 fis4 
  e4 e4 e2   %10
  dis4 e4 e8 fis8 g4 
  fis4 b,4 b2 
  cis4 cis4 c4 b8 ais8 
  b4 ais4 b2 
  a4 a4 b4 e,4   %15
  e'4 d16 cis16 d8 cis2 
  
}

baixo = \relative c {
  fis4 eis4 fis4 gis4 
  a4 fis4 cis2 
  d4 e4 fis8 gis8 a4 
  e2 a,2 
  a'4 gis4 fis8 gis8 a4   %5
  eis4 fis4 cis2 
  fis4 gis4 a4 ais4 
  b2 eis,2 
  fis8 gis8 a8 b8 cis4 fis,8 gis8 
  a4 e4 a,2   %10
  a'4 gis4 g4 fis8 e8 
  ais4 b4 e,2 
  cis4 fis4 d4 g4 
  e4 fis4 b,2 
  fis'8 e8 d8 cis8 b4 cis8 d8   %15
  e2 a,2 
  
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