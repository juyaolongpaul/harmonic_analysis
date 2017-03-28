\version "2.10.33"

\header {
  title = "106 - Jesu Kreuz, Leiden und Pein"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \major
}


soprano = \relative c'' {
  cis4 b4 a4 b4 
  cis4 dis4 e2 
  fis4 e4 d4 cis8 b8 
  b2 a2 
  cis4 e4 d4 cis4   %5
  b4 a4 gis2 
  a4 b4 cis4 cis8 d16 e16 
  d2 cis2 
  cis4 d4 e4 d4 
  cis4 b4 cis2   %10
  b4 b4 cis8 dis8 e4 
  e4 dis4 e2 
  e4 cis4 fis4. e8 
  d4 cis8 b8 b2 
  cis4 e4 d4 cis8 b8   %15
  b2 a2 
  
}

alto = \relative c'' {
  a4 gis4 a4 gis4 
  a8 gis8 fis4 gis2 
  b4 b4 gis4 a4 
  a4 gis4 e2 
  a4 b4 a4 gis4   %5
  fis4 fis4 f2 
  cis4 e4 e4 e4 
  a4 b4 a2 
  a4 a4 a4 a4 
  a4 gis4 a2   %10
  gis4 e4 a4 gis4 
  fis2 gis2 
  g4 g4 fis8 gis8 a4 
  a8 gis8 a4 gis2 
  a4 b4 a4 a4   %15
  a4 gis4 e2 
  
}

tenor = \relative c' {
  e4 e4 e4 e4 
  e4 b4 b2 
  d4 b4 e4 e4 
  fis4 e8 d8 cis2 
  e4 e4 fis4 cis4   %5
  d4 d4 gis,2 
  fis4 gis4 a4 b4 
  fis'4 e4 e2 
  e4 d4 cis4 b4 
  cis8 d8 e4 e2   %10
  e4 b4 a4 b4 
  cis4 b4 b2 
  ais4 a4 a8 b8 cis4 
  d4 e4 e2 
  e4 e4 fis4 fis4   %15
  fis4 e8 d8 cis2 
  
}

baixo = \relative c' {
  a4 e4 cis4 e4 
  a,4 b4 e2 
  b'4 gis4 e4 a4 
  d,4 e4 a,2 
  a'4 gis4 fis4 e4   %5
  d4 cis8 b8 cis2 
  fis4 e4 a4 gis4 
  fis4 gis4 a2 
  a4 fis4 cis4 d4 
  e4 e4 a,2   %10
  e'8 fis8 gis4 fis4 e4 
  a4 b4 e,2 
  g4 a4 d,4 cis4 
  b4 a4 e'2 
  a4 gis4 fis4 e4   %15
  dis4 e4 a,2 
  
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