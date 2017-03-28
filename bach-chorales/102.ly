\version "2.10.33"

\header {
  title = "102 - Ermuntre dich, mein schwacher Geist"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key g \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    g4 g4 a4 
    b4. c8 d4 
    d4 d4 cis4 
    d2 b4 
    c2 b4     %5
    a2. 
    g4 fis2 
    g2. }
  a4 a4 b4 
  c2 c4   %10
  b4. a8 b4 
  a2. 
  b4 b4 b4 
  c4. d8 e4 
  e4 dis4 dis4   %15
  e2 fis4 
  g2 b,4 
  c2 b4 
  a4 a4 d4 
  e4. d8 c4   %20
  b4 a2 
  g2. 
  
}

alto = \relative c' {
  \repeat volta 2 {
    d4 e4 fis4 
    g2. 
    g4 e4 e4 
    fis2 g4 
    e2 d4     %5
    d2. 
    b4 d2 
    d2. }
  d4 d4 d4 
  c2 e4   %10
  e2 e4 
  e2. 
  e4 gis4 gis4 
  a2. 
  a4 fis4 fis4   %15
  g2 a4 
  b2 g4 
  g2 g4 
  fis4 fis4 g4 
  g4. fis8 e4   %20
  d4 d2 
  d2. 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    b4 b4 d4 
    d4. c8 b4 
    b4 a4 a4 
    a2 g4 
    g2 g4     %5
    fis2. 
    e4 a2 
    b2. }
  fis4 fis4 gis4 
  a2 a4   %10
  gis4. a8 gis4 
  a2. 
  gis4 b4 e4 
  e4. d8 c4 
  c4 b4 b4   %15
  b2 d4 
  d2 d4 
  e2 d4 
  d4 d4 d4 
  c4. a8 g4   %20
  g4 g4 fis4 
  b2. 
  
}

baixo = \relative c' {
  \repeat volta 2 {
    g4 e4 d4 
    g4. a8 b4 
    g4 a4 a,4 
    d2 g4 
    c,2 g4     %5
    d'2. 
    e4 d2 
    g,2. }
  d'4 d4 b4 
  a2 c4   %10
  e2 e4 
  a,2. 
  e'4 e4 e4 
  a4. b8 c4 
  a4 b4 b4   %15
  e,2 d4 
  g2 g4 
  c,2 g4 
  d'4 d4 b4 
  c4. d8 e4   %20
  g4 d2 
  g,2. 
  
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