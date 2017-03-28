\version "2.10.33"

\header {
  title = "10 - Aus tiefer Not schrei' ich zu dir"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    b2 e,4 b'4 
    c4 b8 a8 g4 a4 
    b2 b2 
    c4 d4 c8 b8 a4 
    g4 f4 e2 }  %5
  a2 g4 c4 
  b4 a4 d8 c8 b4 
  a2 c2 
  b4 c4 d4 g,4 
  b4 a4 g2   %10
  g2 c4 b4 
  a4 e4 g4 f4 
  e1 
  
}

alto = \relative c' {
  \repeat volta 2 {
    e2 e4 gis4 
    a4 g8 fis8 e4 fis4 
    gis2 gis2 
    a4 b4 c4 f,4 
    e4 d8 c8 b2 }  %5
  e2 d4 e8 fis8 
  g4 a4 b8 a8 gis4 
  e2 a2 
  g4 g4 g8 f8 e4 
  d4 c4 b2   %10
  e2 e4 e8 d8 
  c4 c4 d2~ 
  d4 c4 b2 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    gis2 a4 d4 
    e4 d4 e8 d8 c4 
    b2 e2 
    e4 f4 g4 c,4 
    c8 b8 a4 gis2 }  %5
  a2 b4 c4 
  d4 d8 e8 f4 b,4 
  c2 e2 
  e4 e4 d4 c4 
  fis,8 g4 fis8 g2   %10
  g2 a4 gis4 
  a4 g4 g4 a4 
  b4 a4 gis2 
  
}

baixo = \relative c {
  \repeat volta 2 {
    d2 c4 b4 
    a4 b4 c8 b8 a4 
    e'2 e2 
    a4 g8 f8 e4 f4 
    c4 d4 e2 }  %5
  c'2 b4 a4 
  g4 f8 e8 d4 e4 
  a,2 a'2 
  e4 d8 c8 b4 c4 
  d4 d4 g,2   %10
  c2 a4 e'4 
  f4 c4 bes4 a4 
  gis4 a4 e'2 
  
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