\version "2.10.33"

\header {
  title = "141 - Seelenbräutigam, Jesu, Gottes Lamm"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \major
}


soprano = \relative c'' {
  a4 a4 gis4 a4 
  b2 a4 b4 
  cis4 b4 a2 
  cis4 d4 e4 d4 
  cis4 b4 cis2   %5
  b2 a4 b4 
  cis4 b4 a4 gis4 
  a4. gis8 gis2 
  a4 a4 gis4 a4 
  b2 a4 b4   %10
  cis4 b8 a8 a2 
  
}

alto = \relative c' {
  e4 e4 e4 e8 fis8 
  gis2 fis4 e4 
  e4 e4 e2 
  a4 a4 a4. b8~ 
  b8 a8 gis8 b4 a16 gis16 a4~   %5
  a gis fis gis 
  a gis4. fis4 eis8
  fis8 eis fis4~ fis eis
  fis e e e
  e2 cis8 fis e4~
  e8 a gis4 e2
}

tenor = \relative c' {
  cis4 cis4 b4 cis4 
  e2 cis8 d8 cis8 b8~ 
  b8 a4 gis8 cis2 
  e4 fis4 e4 fis8 e8 
  e4 e4 e2   %5
  e2 cis4 e 
  e4. d8 cis4 cis
  cis2 cis2
  cis4 cis4 b4 a4~ 
  a4 gis4 a8 d cis b   %10
  a8 e' e8. d16 cis2 
  
}

baixo = \relative c {
  a8 b cis d e4 a,
  e'2 fis4 gis
  a4 e4 a,2 
  a'8 gis fis4 cis fis8 gis 
  a4 e8 gis a4 a,   %5
  e'2 fis4 e 
  a e fis cis
  fis,8 gis8 a8 b8 cis2 
  fis4 cis8 d8 e4 cis8 a8 
  e'2 fis4 gis4   %10
  a4 e4 a,2 
  
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