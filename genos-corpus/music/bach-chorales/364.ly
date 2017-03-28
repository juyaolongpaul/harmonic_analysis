\version "2.10.33"

\header {
  title = "364 - Von Gott will ich nicht lassen"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key b \minor
}


soprano = \relative c' {
  \repeat volta 2 {
    \partial 4 fis4 
    b4 cis4 d4 e4 
    cis4. b8 a4 a4 
    b4 b4 cis8 b8 cis4 
    fis,2. } fis'4 
  e4 d4 d4 cis4   %5
  d2. cis4 
  d4 e4 fis4 fis4 
  e4. d8 cis4 cis8 d16 e16 
  d4 cis8. b16 b4 b8 cis16 d16 
  cis4. b8 a4 g4   %10
  fis4 b4 b8 cis8 ais4 
  b2. 
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 cis4 
    fis4 e4 d8 fis8 b8 gis8 
    gis8 fis8 eis16 dis16 eis8 fis4 fis4 
    fis4 e8 d8 e4 e8 fis16 e16 
    d2. } b'4 
  b4 b4 a4 a4   %5
  a2. a4 
  a4 a4 a4. gis8 
  a8 fis8 gis4 e4 fis4 
  fis4 fis4 g8 fis8 g4 
  gis8 fis8 eis4 fis4 fis8 e8   %10
  d8 e8 fis8 g8 fis8 g16 fis16 e8 fis16 e16 
  dis2. 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 ais4 
    b4. ais8 b4. b8 
    b8 a8 gis8 cis8 cis4 cis4 
    b8 a8 g2 fis4 
    fis2. } d'4 
  e8 fis8 g4 fis8 e16 d16 e8 fis16 g16   %5
  fis2. e4 
  fis4 e4 d8 a8 d4 
  cis4 b16 a16 b8 a4 cis4 
  cis8 b8 cis8 dis8 e8 dis8 e8 b8 
  gis8 a16 b16 cis4 cis4 d8 a8   %10
  a4 d8 cis8 d8 e16 d16 cis8 fis,8 
  fis2. 
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 fis8  e8 
    d4 cis4 b8 a8 gis8 e'8 
    a,8 b8 cis4 fis,4 fis'8 e8 
    d8 fis8 g4 ais,4 ais4 
    b2. } b8 a8 
  g8 g'8 fis8 e8 a4 a,4   %5
  d2. a'8 g8 
  fis8 e8 d8 cis8 d8 e16 fis16 b,4 
  cis8 d8 e4 a,4 ais4 
  b8 b'4 a8 g8 a16 b16 e,4 
  eis8 fis16 gis16 cis,4 fis4 b,8 cis8   %10
  d8 cis8 d8 e8 fis8 e8 fis4 
  b,2. 
  
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