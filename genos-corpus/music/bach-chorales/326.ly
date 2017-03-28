\version "2.10.33"

\header {
  title = "326 - Allein Gott in der Höh sei Ehr"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 a8  b8 
    cis4 d4 e4 d4 
    cis4 b4 cis4 cis4 
    cis4 b8 cis8 d8 cis8 b4 
    a4 b4 a4 } a4 
  b4 cis4 d4 cis4   %5
  b4 cis4 b4 b4 
  cis4 d4 e4 d4 
  cis4 b4 cis4 a4 
  b4 cis4 d8 cis8 b4 
  a8 fis8 gis4 a4   %10
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    a4 a4 gis4 fis8 gis8 
    a4 gis4 a4 e4 
    fis4 fis8 e8 d8 e8 e4 
    e8 a8 gis8 fis8 e4 } e4 
  e4 e4 d8 e8 fis4   %5
  fis8 e8 e8 d16 cis16 d4 e4 
  e4 fis4 gis4 fis4 
  e4 d8 fis16 eis16 fis4 fis4 
  fis4 e4 d4 e4 
  e8 fis8 e4 e4   %10
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 cis8  d8 
    e4 fis4 b,8 cis8 d4 
    e4 e4 e4 a,4 
    a2. d8 b8 
    e8 d16 cis16 d4 cis4 } cis4 
  gis4 ais4 b4 ais4   %5
  b4 ais4 fis4 b4 
  a4 a4 b4 a8 b8 
  cis4 fis,8 gis8 a4 cis4 
  fis,8 gis8 a4 a4 gis4 
  a8 d8 b4 cis4   %10
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 a4 
    a'8 gis8 fis4 e4 b4 
    cis8 d8 e4 a,4 a'8 gis8 
    fis8 e8 d8 cis8 b8 a8 gis4 
    cis8 a8 e'4 a,4 } a4 
  e'4 d8 cis8 b4 e8 fis8   %5
  g4 fis4 b,4 gis'4 
  a8 gis8 fis4 e4 fis8 gis8 
  a4 b4 fis4 fis8 e8 
  d4 cis4 b8 cis8 d4 
  cis8 d8 e4 a,4   %10
  
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