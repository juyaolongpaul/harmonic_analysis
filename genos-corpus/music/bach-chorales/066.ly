\version "2.10.33"

\header {
  title = "66 - Christ, unser Herr, zum Jordan kam"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c' {
  \repeat volta 2 {
    \partial 4 d8  e8 
    f4 g4 a4 g4 
    c4 b8 a8 a4 g4 
    c4 b4 a4 g4 
    f4 e4 d4 } d4 
  g4 g4 e4 a4   %5
  a4 gis4 a4 a4 
  d4 d4 e8 d8 c4 
  d8 c8 b4 a4 b4 
  c4 b4 a4 g4 
  a8 g8 f4 e4 d8 e8   %10
  f4 g4 a8 g8 f4 
  g8 f8 e4 d4 d'4 
  d4 d4 e4 d4 
  c4( b4) a
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 a4 
    d4 e4 f4 e4 
    e4 e4 e4 e4 
    e4 e4 f8 e8 d4 
    a8 d4 cis8 a4 } a4 
  d4 c8 b8 c8 d8 e4   %5
  e4 e4 e4 e4 
  a4 g4 g4 a8 g8 
  f4 b,8 c16 d16 c4 e4 
  e4 d4 c8 d8 e4 
  a,4 d4 cis4 a4   %10
  d4 e4 e8 cis8 d4 
  e8 d8 cis4 a4 a'4 
  g4 g4 g4 a8 d,8 
  e4. d8 cis4
  
}

tenor = \relative c {
  \repeat volta 2 {
    \partial 4 f8  g8 
    a8 b8 c4 c4 c8 b8 
    a4 gis4 c4 b4 
    a4 b8 cis8 d4 d,8 e8 
    f8 g8 a4 f4 } f8 e8 
  d4 g4 g4 c4   %5
  b8 a8 b4 c4 c4 
  c4 b4 c8 b8 a4 
  b8 a8 gis4 e4 gis4 
  a8 g8 f8 g8 a8 b8 cis8 d8 
  e4 a,4 a4 f8 e8   %10
  d8 f8 bes4 a4 bes4~ 
  bes4 e,8 f16 g16 f4 d4 
  b'8 c8 d8 b8 c8 b8 a4 
  a4 gis4 e
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 d4 
    d4 c4 f4 c4 
    c8 d8 e4 a,4 e'4 
    a4 g4 f4 b,8 cis8 
    d4 a4 d4 } d8 c8 
  b8 a8 g4 c4 b8 a8   %5
  e'4 e,4 a4 a'8 g8 
  fis4 g4 c,4 f8 e8 
  d4 e4 a,4 e'4 
  a4 d,8 e8 f4 e8 d8 
  cis4 d4 a4 d8 c8   %10
  bes8 a8 g4 f4 bes8 a8 
  g4 a4 d,4 fis'4 
  g8 a8 b8 g8 c,4 f4 
  e8 d8 e4 a,
  
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