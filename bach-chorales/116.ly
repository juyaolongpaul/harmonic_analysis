\version "2.10.33"

\header {
  title = "116 - Nun lob, mein Seel, den Herren"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key d \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 d4 
    d2 cis4 
    b2 a4 
    d4 e2 
    fis2 fis4 
    fis8 e8 fis2     %5
    fis2 e4 
    d8 e8 e2 
    d2 } d4 
  d4 e4 fis4 
  e2 fis8 e8   %10
  d8 e8 cis2 
  b2 e4 
  e4 d4 cis4 
  d8 cis8 b2 
  a2 a4   %15
  d2 d4 
  e2 e4 
  fis4 e4 fis4 
  d2 d4 
  g2 g4   %20
  fis8 e8 fis2 
  e2 e4 
  fis2 fis4 
  g2 g4 
  a4. g8 fis8 e8   %25
  d2 fis4 
  e4 d4 cis4 
  d8 cis8 b2 
  a2 a4 
  d2 cis4   %30
  b2 a4 
  e'4 fis2 
  e2 fis4 
  g4 fis4 e4 
  fis8 g8 e2   %35
  d2 
}

alto = \relative c'' {
  \repeat volta 2 {
    \partial 4 a4 
    b4 a4 a4 
    a4 g4 a8 g8 
    fis4 b4 a4 
    a2 a4 
    a2 cis4     %5
    fis,4 b4 e,4 
    a4 b4 a8 e8 
    fis2 } a4 
  b4 a4 a4 
  a8 b8 a8 g8 fis4   %10
  fis4 g8 fis8 e8 fis8 
  d2 e4 
  e2 e4 
  a2 gis4 
  e2 e4   %15
  d8 e8 fis4 b4 
  b4 a4 a4 
  a2.~ 
  a4 g4 g8 a8 
  b4 a8 g8 c8 b8   %20
  a8 g8 c4 b8 a8 
  g2 a4 
  a4 b2~ 
  b4 a8 g8 c8 b8 
  a8 g8 a2~   %25
  a4 g4 a4 
  b2 a4 
  a2 gis4 
  e2 e4 
  a8 gis8 a8 b8 e,4   %30
  fis4 cis4 cis8 d8 
  e4 a2 
  a2 a4 
  b4 a4 g4 
  fis4 g8 fis8 e4   %35
  fis2 
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 fis4 
    fis2 fis8 e8 
    d4 e8 d8 cis4 
    d2 cis4 
    d2 d4 
    d2 cis4     %5
    d2 cis4 
    d2 cis4 
    a2 } d4 
  g4 cis,4 d4 
  d4 cis4 cis4   %10
  b2 ais4 
  b2 gis8 a8 
  b8 gis8 a8 b8 cis8 b8 
  a4 fis'4 e8 d8 
  cis2 cis8 b8   %15
  a4 d2~ 
  d4 cis8 b8 cis4 
  d4 a4 d4 
  b2 b4 
  e2 e4   %20
  e2 dis4 
  b2 cis4 
  d2 d4 
  g,4 e'2~ 
  e4 d4. c8   %25
  b2 d8 cis8 
  b4 e2 
  d8 e8 fis4 e8 d8 
  cis2 cis4 
  cis4 b4 a4   %30
  a4 gis4 a4 
  a2 d4 
  cis2 d4 
  d2 a4 
  d2 cis4   %35
  a2 
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 d8  cis8 
    b4 fis2 
    g8 fis8 e4 fis4 
    b8 a8 g4 a4 
    d,2 d4 
    d'4. cis8 b8 ais8     %5
    b8 a8 g8 fis8 g8 e8 
    fis4 g4 a4 
    d,2 } fis4 
  g8 a8 g8 fis8 e8 d8 
  a'2 ais4   %10
  b4 e,4 fis4 
  b,2 cis4 
  gis'8 e8 fis8 gis8 a8 gis8 
  fis8 e8 d4 e4 
  a,2 a'8 g8   %15
  fis4 b8 a8 g8 fis8 
  g8 e8 a8 g8 fis8 e8 
  d8 e8 d8 cis8 d8 fis8 
  g2 g8 fis8 
  e4 fis8 g8 a8 b8   %20
  c4 a4 b4 
  e,2 a4 
  d8 cis8 b8 a8 g8 fis8 
  e8 d'8 c8 b8 a8 g8 
  fis8 g8 fis8 e8 d4   %25
  g2 d'4 
  gis,8 e8 fis8 gis8 a8 gis8 
  fis8 e8 d4 e4 
  a,2 a'8 gis8 
  fis8 e8 fis8 gis8 a4   %30
  dis,8 cis8 dis8 f8 fis8 e8 
  d8 cis8 d2 
  a2 d8 cis8 
  b8 a8 b8 d8 cis8 e8 
  d4 g,4 a4   %35
  d2 
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