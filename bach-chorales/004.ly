\version "2.10.33"

\header {
  title = "4 - Es ist das Heil uns kommen her"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key e \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 b4 
    b4 b4 b4 d4 
    cis4 b4 a4 b4 
    gis4 e8 fis8 gis4 ais4 
    b4 cis4 b4 } b4 
  e4 dis4 cis4 dis4   %5
  e8 dis8 cis4 b4 b4 
  e4 b4 cis4 gis8 a8 
  b4 a4 gis4 gis4 
  fis4 a4 gis4 fis4 
  cis4 dis4 e4 
}

alto = \relative c'' {
  \repeat volta 2 {
    \partial 4 gis4 
    fis8 gis8 a4 gis4 fis4 
    e8 a8 a8 gis8 e4 fis4 
    e4 e8 dis8 e4 cis4 
    fis8 e16 dis16 e4 dis4 } e8 fis8 
  gis8 ais8 b4 ais4 b4   %5
  b4 fis4 fis4 gis4 
  e8 fis8 gis4 a4 e8 fis8 
  gis4 fis4 eis4 e4 
  dis4 cis4 b4 b4 
  cis4 b4 b4 
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    fis4 e8 dis8 e4 a,8 b8 
    cis16 d16 e4 d8 cis4 b4 
    b4 b8 a8 gis4 fis4 
    fis8 b4 ais8 fis4 } gis4 
  cis4 fis,4 fis'4 fis4   %5
  b,4 ais4 dis4 e4 
  b4 e4 e4 cis4 
  d4 cis4 cis4 b4 
  b4 e4 e4 dis8 b8 
  fis2 gis4 
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 e4 
    dis4 b4 e4 fis8 gis8 
    a4 e4 a,4 dis4 
    e8 fis8 gis4 cis,4 fis8 e8 
    dis8 cis16 b16 fis'4 b,4 } e4 
  cis4 dis8 e8 fis4 b4   %5
  gis8 e8 fis4 b,4 e4 
  gis4 e4 a8 b8 cis4 
  eis,4 fis4 cis4 e4 
  b4 cis8 dis8 e4 b4 
  ais4 b4 e,4 
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