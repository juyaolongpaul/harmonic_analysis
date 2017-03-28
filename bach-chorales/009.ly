\version "2.10.33"

\header {
  title = "9 - Ermuntre dich, mein schwacher Geist"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 g4 
    g4 a4 b8 cis8 d4 
    d4 cis4 d4 b4 
    c4 b4 a4 b4 
    a2 g4 } a4 
  a4 b4 c4 c4   %5
  b8 a8 b4 a4 b4 
  b4 b4 c8 d8 e4 
  e4 dis4 e4 fis4 
  g4 b,4 c4 b4 
  a2 a4 d4   %10
  e4 d4 c4 b8 c8 
  a2 g4 
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    e4 fis4 d8 e8 fis4 
    e8 fis8 g4 fis4 fis4 
    e4 d8 e8 fis4 g4~ 
    g4 fis4 d4 } fis8 g8 
  a4. gis8 a8 gis8 a4   %5
  a4 gis4 e4 e8 dis8 
  e4 e4 e8 fis8 g8 fis8 
  fis8 g8 a4 g4 a4 
  g8 a8 b4~ b8 a8~ a8 g8 
  g2 fis4 g8 fis8   %10
  e4 fis4 g8 a8 d,4 
  e4 d4 d4 
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 b4 
    b8 c4 d8 b8 a8 a4 
    a4 a4 a4 b8 a8 
    g8 a8 b8 c8 d4 d4 
    d4. c8 b4 } d8 e8 
  fis8 e8 d4 e4 fis4   %5
  b,4 e8 d8 c4 g8 a8 
  b8 c8 d8 e8 c4 cis4 
  b4 b4 b4 d4 
  d4 g4 g,8 a8 b4 
  e8 d8 e8 cis8 d4 d4~   %10
  d8 c8 c8 b8 b8 a8 g4 
  g8 fis16 e16 fis4 b4 
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 g8  fis8 
    e4 d4 g4 fis8 g8 
    a4 a,4 d4 dis4 
    e8 fis8 g8 c,8 c'8 b8 a8 g8 
    d'4 d,4 g4 } d4 
  d'8 c8 b4 a4 dis,4   %5
  e4 e4 a,4 e'8 fis8 
  g4 gis4 a4 ais4 
  b4 b,4 e4 d'8 c8 
  b8 a8 g8 fis8 e8 fis8 g8 e8 
  cis8 b8 cis8 a8 d4 b4   %10
  c4 d4 e8 fis8 g4 
  c,4 d4 g,4 
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