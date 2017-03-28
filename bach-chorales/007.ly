\version "2.10.33"

\header {
  title = "7 - Nun lob', mein'Seel', den Herren"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key a \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 a4 
    a2 gis4 
    fis2 e4 
    a4 b2 
    cis2 cis4 
    cis4 b4 cis4     %5
    cis2 b4 
    a4 b2 
    a2 } a4 
  a4 b4 cis4 
  b2 cis4   %10
  a4 gis2 
  fis2 b4 
  b4 a4 gis4 
  a4 fis2 
  e2 e4   %15
  a2 a4 
  b2 b4 
  cis4. b8 cis4 
  a2 a4 
  d2 d4   %20
  cis4 b4 cis4 
  b2 b4 
  cis2 cis4 
  d2 d4 
  e2.   %25
  a,2 cis4 
  b4 a4 gis4 
  a4 fis2 
  e2 e4 
  a2 gis4   %30
  fis2 e4 
  b'4 cis2 
  b2 cis4 
  d2 cis4 
  b4 a8 gis8 a8 b8   %35
  cis4 b2 
  a2 
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    fis2 e4 
    d4 cis4 b4 
    e4 fis4 e4 
    e2 e8 fis8 
    gis2 eis4     %5
    fis2 gis4 
    a2 gis4 
    e2 } e4 
  fis4 gis4 a4 
  b4 a4 gis4   %10
  fis2 eis4 
  cis2 fis4 
  e2 e4 
  e2 dis4 
  b2 cis4   %15
  cis2 fis4 
  e4 fis4 gis4 
  a2 g4 
  fis2 a4 
  b2 b8 a8   %20
  g8 ais8 b4 e,4 
  dis2 e4 
  e2 fis4 
  fis2 g4 
  g4 fis4 e4   %25
  d2 e4 
  fis2 e4 
  e2 dis4 
  b2 cis4 
  cis4 dis4 e4   %30
  dis2 e4 
  fis4. gis8 a4 
  e2 gis4 
  a4 b4 a4 
  gis4 a2   %35
  a4 gis2 
  e2 
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 cis4 
    cis2 cis4 
    a2 gis4 
    a2 gis4 
    a2 cis4 
    cis2 b4     %5
    a2 e'4 
    e4 fis4 e4 
    cis2 } cis4 
  d2 e4 
  eis4 fis4 gis4   %10
  cis,4 d4 cis4 
  a2 a4 
  gis4 a4 b4 
  cis2 b8 a8 
  gis2 gis4   %15
  a2 d4 
  b2 e4 
  e2. 
  d2 e4 
  fis2 b,4   %20
  e4 d8 cis8 b8 ais8 
  b2 gis4 
  a2 a4 
  b2 b4 
  b4 a4 g4   %25
  fis2 a4 
  fis2 b4 
  cis4 b4 a4 
  gis2 gis4 
  a2 b4   %30
  b4 a4 gis4 
  d'4 a2 
  gis2 cis4 
  fis4 e4 e4 
  e8 d8 cis4 fis4   %35
  e2 d4 
  cis2 
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 a4 
    fis2 cis4 
    d2 d4 
    cis4 d4 e4 
    a,2 a'4 
    eis2 cis4     %5
    fis4 e4 d4 
    cis4 d4 e4 
    a,2 } a4 
  d2 a'4 
  gis4 fis4 eis4   %10
  fis4 b,4 cis4 
  fis,2 dis'4 
  e2 d4 
  cis4 a4 b4 
  e2 cis4   %15
  fis4 e4 fis4 
  gis4 fis4 e4 
  a2 a,4 
  d2 cis4 
  b4 b'8 a8 g8 fis8   %20
  e8 fis8 g4 fis4 
  b,2 e4 
  a4 gis4 fis4 
  b4 a4 g4 
  cis,2.   %25
  d2 a'4 
  dis,2 e4 
  a,4 b2 
  e2 cis4 
  fis2 e4   %30
  b2 cis4 
  d2 dis4 
  e2 eis4 
  fis4 gis4 a4 
  e4 fis4 d4   %35
  e2. 
  a,2 
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