\version "2.10.33"

\header {
  title = "42 - Du Friedensfürst, Herr Jesu Christ"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 cis4 
    a4 b4 cis4 e4 
    d4 d4 cis4 e4 
    d4 cis4 b4 b4 
    cis2. } b4 
  b4 b4 cis4 b4   %5
  a4 b4 gis4 gis4 
  a4 b4 cis4 b8 cis8 
  d4 cis4 b2 
  a2.
  
}

alto = \relative c'' {
  \repeat volta 2 {
    \partial 4 a4 
    fis4 gis4 a4 b4 
    a4 b4 a4 a4 
    a4 a4 a4 gis4 
    a2. } gis4 
  e4 gis4 a4 gis4   %5
  fis4 gis4 eis4 eis4 
  fis4 gis4 a4 e4 
  a4 a4 a4 gis4 
  e2. 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    d4 d4 e4 e4 
    fis4 e4 e4 e4 
    fis4 e4 fis4 e4 
    e2. } e4 
  b4 e4 e4 e8 d8   %5
  cis4 d4 cis4 cis4 
  cis4 e4 e2 
  d4 e4 fis4 e8 d8 
  cis2. 
  
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 a4 
    d8 cis8 b4 a4 gis4 
    fis4 gis4 a4 cis,4 
    fis8 gis8 a4 d,4 e4 
    a,2. } e'8 fis8 
  gis4 e4 a4 e4   %5
  fis4 b,4 cis4 cis4 
  fis4 e4 a4 gis4 
  fis4 e4 d4 e4 
  a,2.
  
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