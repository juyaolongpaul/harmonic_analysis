\version "2.10.33"

\header {
  title = "153 - Alle Menschen müssen sterben"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    a4 d4 a4 b4 
    a4. g8 fis4 fis4 
    a4 a4 g4 fis4 
    e4. d8 d2 }
  e4 e4 fis8 gis8 a4   %5
  a4. gis8 a4 a4 
  b4. cis8 d4 d4 
  cis4. cis8 b4 b4 
  fis4 fis4 b4 a4 
  a4 gis4 a2   %10
  b4 a4 g4 fis4 
  e4. d8 d2 
  
}

alto = \relative c' {
  \repeat volta 2 {
    d4 d4 d4 d4 
    d4 cis4 d4 d4 
    d4 d8 cis8 b8 cis8 d4~ 
    d4 cis4 d2 }
  cis4 cis4 d4 cis4   %5
  fis4 e8 d8 cis4 cis4 
  e8 fis8 g4 fis4 fis4 
  fis4. e8 d4 d4 
  d4 d4 d4 cis4 
  d4 d4 cis2   %10
  d4 d8 cis8 b8 cis8 d4 
  d4 cis4 d2 
  
}

tenor = \relative c {
  \repeat volta 2 {
    fis4 fis8 g8 a4 g8 fis8 
    e8 d8 e4 d4 d4 
    fis4 fis4 g4 a4 
    b4 a8. g16 fis2 }
  a4 a4 a4 a4   %5
  b4 b4 a4 a4 
  g4. a8 b2~ 
  b4 ais4 b2 
  a4 a4 g8 fis8 e4 
  d8 fis8 e8 d8 e2   %10
  g4 fis4 g4 a4 
  b4 a8. g16 fis2 
  
}

baixo = \relative c {
  \repeat volta 2 {
    d4 b4 fis4 g4 
    a4 a4 d,4 d4 
    d'4 d4 e4 fis4 
    g8 e a8 a,8 d2 }
  a4 a4 d4 fis8 e8   %5
  d4 e4 a,2 
  e'2 b8 cis8 d8 e8 
  fis2 b,2 
  d4 d4 g,4 a4 
  b2 a2   %10
  g4 d'4 e4 fis4 
  g8 e8 a8 a,8 d2 
  
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