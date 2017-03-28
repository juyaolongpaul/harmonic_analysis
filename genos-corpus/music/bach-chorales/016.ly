\version "2.10.33"

\header {
  title = "16 - Es woll' uns Gott genadig sein"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key b \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 cis4 
    d4 cis4 b4 cis8 d8 
    e4 fis4 e4 d4 
    cis2 r4 e4 
    d4 cis4 d4 b4 
    a4 fis8 g8 a4 g4     %5
    fis2. } a4 
  g4 fis4 e4 fis4 
  d4 e4 fis4 cis'4 
  d4 cis4 b4 cis8 d8 
  e4 d4 cis4 fis4   %10
  e8 d8 cis8 d8 e4 b4 
  a4 g4 fis4 e4 
  a4 b4 cis8 d8 e4 
  d4 cis4 b4 d4 
  cis4 b4 a8 g8 fis8 g8   %15
  a4 g4 fis2~ 
  fis2.
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 fis4 
    fis4 fis4 g4 a4 
    g4 fis8 gis8 ais4 b4 
    ais2 r4 cis4 
    b8 a8 g4 a4 g8 fis8 
    e4 d8 e8 fis4 e8 d8     %5
    cis2. } cis4 
  d4 d4 cis4 cis4 
  d8 cis8 b4 cis4 fis4 
  fis8 gis8 ais4 b4 ais8 b8 
  cis8 fis,8 fis4 fis4 a4   %10
  gis4 a4 e4 e4 
  e8 d8 e8 cis8 d4 cis4 
  cis8 dis8 e4 e8 fis8 g8 fis8 
  fis8 gis8 a4 gis4 fis4 
  e8 fis8 g4 fis8 e8 d4   %15
  e2. d4~ 
  d4 cis8 b8 cis4
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 ais4 
    b4 cis8 d8 e4 a,4 
    b8 cis8 d4 e4 fis4 
    fis2 r4 fis4 
    fis4 e4 d4 d4 
    e4 a,8 b8 c4 b4     %5
    ais2. } a4 
  b4 a4 a8 g8 fis4 
  fis4 b4 ais4 ais4 
  b4 e4 fis4 e8 d8 
  cis4 b4 ais4 b4   %10
  b4 a4 b8 a8 gis4 
  a4 a4 a4 a4 
  a4 gis4 ais8 b8 cis4~ 
  cis8 b8 e4 e4 a,8 b8 
  cis4 d4 d4 a4   %15
  a4 b4 cis4 b4~ 
  b4 ais8 gis8 ais4
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 fis4 
    b4 a4 g4 fis4 
    e4 d4 cis4 b4 
    fis'2 r4 ais4 
    b4 e,4 fis4 g4 
    cis,4 d4 dis4 e4     %5
    fis2. } fis4 
  b,8 cis8 d4 a4 ais4 
  b4 g4 fis4 fis'4 
  b4 cis4 d4 cis8 b8 
  ais4 b4 fis4 dis4   %10
  e4 fis4 gis8 fis8 e8 d8 
  cis8 b8 cis8 a8 d4 a'8 gis8 
  fis4 e8 d8 cis8 b8 ais4 
  b4 cis8 dis8 e4 fis8 gis8 
  a4 b8 cis8 d4 d,4   %15
  c4 b4 ais4 b4 
  fis2.
  
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