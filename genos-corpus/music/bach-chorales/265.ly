\version "2.10.33"

\header {
  title = "265 - Was mein Gott will, das"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key b \minor
}


soprano = \relative c' {
  \repeat volta 2 {
    \partial 4 fis4 
    a4 b4 a4 d4 
    d4 cis4 d4 d4 
    cis4 b4 e4. d8 
    cis4 d4 cis2 
    b2. } b4   %5
  b4 b4 cis4. b8 
  cis4 a4 a4 gis4 
  a4 a4 b4 cis4 
  d4 cis4 b2 
  cis2. fis,8 gis8   %10
  a4 b4 a4 d4 
  d4 cis4 d4 d4 
  cis4 b4 e4. d8 
  cis4 d4 cis2 
  b2.   %15
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    d4 d4 d4 d8 e8 
    fis8 g8 a4 a4 a4 
    a4 g4 g4 fis4 
    fis2~ fis8 g8 fis8 e8 
    d2. } fis4   %5
  e4 d4 cis4 dis8 eis8 
  fis4 fis4 fis4 e4 
  e4 e4 e4 e4 
  fis4 ais4 b4 gis4 
  ais2. cis,4   %10
  fis4 gis4 fis4 fis4 
  b4 a4 a4 gis4 
  a4 gis4 ais4. b8 
  cis4 b2 ais4 
  a g fis   %15
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 b4 
    a4 g4 fis8 g8 a8 b8 
    a4 e'4 fis4 fis4 
    e4 e4 b4 b4 
    ais4 b2 ais4 
    fis2. } d'8 cis8   %5
  b8 cis8 b8 a8 gis4 a8 b8 
  a8 b8 cis4 b4 b4 
  cis4 cis4 b4 ais4 
  b4 g'4 fis4 eis 
  fis2. a,8 b8   %10
  cis4 d8 cis8 cis4 d4 
  e4 e4 fis4 e8 d8 
  e4 e4 e8 fis8 g4~ 
  g8 fis16 e16 d8 cis16 b16 fis'8 cis8 fis8 e8 
  dis4 e dis   %15
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 b4 
    fis'4 g4 d8 e8 fis8 g8 
    a4 a,4 d4 d4 
    a'4 e8 fis8 g8 a8 b4 
    fis8 e8 d8 e8 fis2 
    b,2. } b'8 a8   %5
  gis8 a8 gis8 fis8 eis8 cis8 fis8 gis8 
  a8 gis8 fis8 e8 d8 b8 e4 
  a,4 a'4 gis4 g4 
  fis4 e4 d8 cis8 d8 b8 
  fis'2. fis,4   %10
  fis' eis fis b8 a8 
  gis4 a d, b 
  a e'8 d cis d cis b 
  ais g' fis eis fis2 
  b,2.  %15
  
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