\version "2.10.33"

\header {
  title = "194 - Liebster Immanuel, Herzog der Frommen"
  composer = "J. S. Bach"
}


global = {
  \time 3/2
  \key b \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    d2 d2 d2 
    cis2. b4 a2 
    b2 g2 fis4 e4 
    e1 d2 
    a'2 a2 fis2     %5
    b2. cis4 d2 
    d,4 e4 e2. fis4 
    fis1. }
  fis2 fis2 g2 
  a2 fis1   %10
  b2 ais2 b2 
  cis2 ais1 
  d2 cis2 b2 
  ais2. fis4 b2 
  cis4 d4 ais2. b4   %15
  b1. 
  
}

alto = \relative c' {
  \repeat volta 2 {
    fis2 fis2 eis4 fis4 
    gis2 eis2 fis2 
    g2 e2 d2 
    d2 cis2 a2 
    d2 e2 d2     %5
    d2 g2 fis2 
    fis2 e4 d4 cis4 b4 
    cis1. }
  d2 d2 d2 
  e2 d1   %10
  fis2 e2 fis2 
  fis2 fis1 
  fis2 g2 e2 
  fis2. fis4 fis2 
  g2 fis2 fis2   %15
  fis1. 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    b2 b2 b4 a4 
    gis2 cis2 cis2 
    d2 e4 a,4 a2 
    a2. g4 fis2 
    a2 a2 a2     %5
    g4 a4 b2 b2 
    b2 b2 e2 
    ais,1. }
  b2 b2 b2 
  a2 a1   %10
  b2 cis2 d2 
  ais2 cis1 
  b2 ais2 b2 
  cis2. cis4 b2 
  e4 d4 cis2 cis2   %15
  dis1. 
  
}

baixo = \relative c {
  \repeat volta 2 {
    b2 b'4 a4 gis4 fis4 
    eis2 cis2 fis2 
    b,2 cis2 d2 
    a'2 a,2 d2 
    fis2 cis2 d2     %5
    g2 e2 b2 
    a2 g2 g'2 
    fis1. }
  b,2 d2 e2 
  cis2 d1   %10
  d2 cis2 b2 
  fis'2 fis1 
  b2 e,2 g2 
  fis2 e2 d2 
  e2 fis2 fis,2   %15
  b1. 
  
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