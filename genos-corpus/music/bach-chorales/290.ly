\version "2.10.33"

\header {
  title = "290 - Es ist das Heil uns kommen her"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key e \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 b4 
    b4 b4 b8 cis8 d4 
    cis4 b4 a4 b8 a8 
    gis8 fis8 e8 fis8 gis4 ais4 
    b4 cis4 b4 } b4 
  e4 dis4 cis4 dis4   %5
  e8 dis8 cis4 b4 b4 
  e4 b4 cis4 gis8 a8 
  b4 a4 gis4 gis4 
  fis8 gis8 a4 gis4 fis4 
  cis4 dis4 e4 
}

alto = \relative c'' {
  \repeat volta 2 {
    \partial 4 gis4 
    fis8 gis8 a4 gis8 a8 b4 
    b8 a8 d,8 e8 fis4 fis4 
    e4 b4 e4 cis4 
    fis4. e8 dis4 } e8 fis8 
  gis8 ais8 b4 ais8 gis8 fis4   %5
  b8 gis8 ais4 fis4 gis8 a8 
  b8 a8 gis8 fis8 eis8 fis8 gis8 cis,8 
  cis8 gis'8 gis8 fis8 fis8 eis8 eis4 
  fis8 e8 dis8 cis8 b8 cis8 dis4 
  e4 b4 b4 
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    b4 fis'4 e4 e4 
    e8 fis8 g4 d4 b4 
    b8 a8 gis4 cis4 fis,4 
    fis8 b8 ais4 fis4 } gis4 
  cis4 dis8 e8 fis4 b,4   %5
  b4 fis'8 e8 dis4 e4 
  b8 cis8 d4 cis8 dis8 eis8 fis8 
  gis8 cis,8 cis4 cis4 cis4 
  cis8 b8 a4 e'4 b8 a8 
  g4 fis8 a8 gis4 
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 e4 
    e4 dis4 e4 gis,4 
    a4 b8 cis8 d4 dis4 
    e4. dis8 cis8 dis8 e8 fis8 
    dis8 b8 fis'4 b,4 } e8 dis8 
  cis4 b4 fis'8 gis8 a8 fis8   %5
  gis8 e8 fis4 b,4 e8 fis8 
  gis8 a8 b4 b8 a8 gis8 fis8 
  eis4 fis4 cis4 cis'8 b8 
  a8 gis8 fis4 gis8 a8 b4 
  b8 ais8 b4 e,4 
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