\version "2.10.33"

\header {
  title = "233 - Werde munter, mein Gemüte"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    cis4 d4 e4 e4 
    d4 cis4 b4 b4 
    cis4 d4 e4 cis4 
    b4 b4 a2 }
  b4 cis4 d4 d4   %5
  cis4 cis4 b4 b4 
  d4 e4 fis4 fis4 
  e4 e4 d4 d4 
  cis4 d4 e4 e4 
  d4 cis4 b2   %10
  cis4 d4 e8 d8 cis4 
  b4 b4 a2 
  
}

alto = \relative c'' {
  \repeat volta 2 {
    a4 a4 gis4 a4 
    a8 gis8 a4 gis4 gis4 
    a4 a4 b4 a4 
    a4 gis4 e2 }
  gis4 ais4 b4 b4   %5
  b4 ais4 fis4 fis4 
  fis4 a4 a4 a4 
  a8 e8 a8 g8 fis4 fis4 
  e4 a4 e8 fis8 gis4 
  a4 a4 gis2   %10
  a4 a4 a8 gis8 a4 
  a4 gis4 e2 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    e4 fis4 e8 d8 cis4 
    d4 e4 e4 e4 
    e4 fis4 b,4 cis8 d8 
    e8 b8 e8 d8 cis2 }
  e4 e4 fis4 fis4   %5
  fis8 cis8 fis8 e8 d4 d4 
  d4 cis4 d4 d4 
  d4 cis4 a4 a4 
  a8 b8 cis4 b4 e4 
  fis4 e4 e2   %10
  e4 fis4 e4 e4 
  e8 b8 e8 d8 cis2 
  
}

baixo = \relative c' {
  \repeat volta 2 {
    a4 fis4 cis4 fis4 
    b,4 cis8 d8 e4 e4 
    a8 gis8 fis4 gis4 a4 
    e4 e4 a,2 }
  e'4 d8 cis8 b8 cis8 d8 e8   %5
  fis4 fis4 b,4 b4 
  b'4 a8 g8 fis8 e8 d4 
  a'4 a,4 d4 d4 
  a'8 gis8 fis4 gis4 cis4 
  fis,8 gis8 a4 e2   %10
  a8 fis8 e8 d8 cis8 b8 a4 
  e'4 e4 a,2 
  
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