\version "2.10.33"

\header {
  title = "24 - Valet will ich dir geben"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \major
}


soprano = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    a'4 a4 b4 cis4 
    d2 d4 fis4 
    e4 d4 d4 cis4 
    d2. } d8 e8 
  fis4 fis4 e4. d8   %5
  cis8 b8 cis4 a4 cis4 
  d4 cis4 b4 b4 
  a2. a4 
  fis8 g8 a4 b4 a4 
  a4 g8 fis8 fis4 a4   %10
  g4 fis4 e4 e4 
  d2. 
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 a4 
    d4 d4 d4 g4~ 
    g4 fis8 e8 fis4 a8 b8 
    cis4 b4 b4 a4 
    a2. } a4 
  d4 d8 cis8 b8 gis8 a8 b8   %5
  e,8 fis8 gis4 fis4 a4~ 
  a8 gis8 a2 gis4 
  e2. e4 
  d4 d4 d8 e8 fis4 
  fis4 e4 dis4 d4   %10
  d4 d2 cis4 
  a2. 
}

tenor = \relative c {
  \repeat volta 2 {
    \partial 4 fis8  e8 
    fis8 g8 a8 fis8 g4 g8 a8 
    b4 a8 g8 a4 a'4 
    g4 fis4 f4 e4 
    fis2. } fis8 g8 
  a4 d,4 e8 d8 cis8 b8   %5
  a4 gis4 cis4 fis8 e8 
  d4 e4 e8 fis8 e8 d8 
  cis2. a4 
  a4 d8 c8 b8 cis8 d8 c8 
  b2 b4 c4   %10
  b4 b4 b4 a8 g8 
  fis2. 
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 d8  cis8 
    d8 e8 fis8 d8 g8 fis8 e4 
    b8 cis8 d4 d,4 d''4 
    ais4 b8 a8 gis4 a4 
    d,2. } d4 
  d'8 cis8 b8 a8 gis8 e8 fis8 gis8   %5
  a8 gis8 fis8 eis8 fis4 fis4 
  b8 b,8 cis8 d8 e8 d8 e8 e,8 
  a2. cis4 
  d8 e8 fis8 d8 g4 fis8 e8 
  dis4 e4 b4 fis4   %10
  g8 a8 b8 a8 g8 e8 a4 
  d,2. 
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