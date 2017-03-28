\version "2.10.33"

\header {
  title = "89 - O Haupt voll Blut und Wunden"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key b \minor
}


soprano = \relative c' {
  \repeat volta 2 {
    \partial 4 fis4 
    b4 a4 g4 fis4 
    e2 fis4 cis'4 
    d4 cis8 d16 e16 d4 cis8 b8 
    b2. } d4 
  cis8 b8 a4 b4 cis4   %5
  d2 d4 a4 
  b4 a4 b8 a8 g4 
  fis2. d'4 
  cis8 d8 e4 d4 cis4 
  b2 cis4 fis,4   %10
  g4 fis4 e4 a8 g8 
  fis2~ fis4
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    d8 e8 fis4 b,8 cis8 d4 
    d4 cis4 d4 fis4 
    fis4 g4 fis4 e8 d8 
    d2. } fis4 
  gis4 fis4 g4 g4   %5
  a4 g4 fis4 fis4 
  fis8 e8 dis8 e8 fis4. e8 
  e4 dis2 b'4 
  a4 a4 a8 b8 cis8 a8 
  a4 gis4 a4 d,4   %10
  e4 d4 e4 fis8 e8 
  e4 d cis 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 b4 
    b8 cis8 d4 e4 a,4 
    b4 a4 a4 cis4 
    b4 b4 b4 ais4 
    fis2. } b8 a8 
  gis4 cis4 d4 e4   %5
  d8 c4 bes8 a4 d8 c8 
  b4 c4 b4 b4 
  b2. e4 
  e4 e8 fis16 g16 fis4 e4 
  fis4 e4 e4 a,4   %10
  a4 a4 a8 b8 c4 
  cis4 b ais
  
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 b8  a8 
    g4 fis4 e4 d4 
    gis,4 a4 d4 ais4 
    b4 e4 fis4 fis4 
    b,2. } b'4 
  eis,4 fis f e   %5
  fis4 g4 d4 d4 
  g4 fis8 e8 dis4 e4 
  b2. gis'4 
  a8 b8 cis4 fis,8 gis8 a4 
  dis,4 e4 a,4 d4   %10
  cis4 d4 c8 b8 a4 
  ais b fis'
  
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