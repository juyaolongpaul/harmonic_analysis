\version "2.10.33"

\header {
  title = "5 - An Wasserflüssen Babylon"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 d4 
    e4 d8 c8 b8 c8 d4 
    c8 b8 c4 b4 a4 
    b4 c4 d4 c8 b8 
    a8 g8 a4 g4 } g4 
  a4 b4 c8 b8 a4   %5
  b4 a4 g4 g8 a8 
  b4 c4 d4 e4 
  b4 cis4 d4 b8 c8 
  d4 e4 d8 c8 b4 
  c2 b4 e4   %10
  e4 e4 a,4 d4 
  c4 b4 a4 c4 
  b4 a4 g4 a8 g8 
  f4 e4 d4 d4 
  g4 a4 b4. c8   %15
  d4 c8 b8 a4 b8 c8 
  a2 g4 
}

alto = \relative c'' {
  \repeat volta 2 {
    \partial 4 g4 
    g4 fis4 g4 fis4 
    e8 g4 fis8 g4 a4 
    g4 g4 g4 g4 
    g4 fis4 d4 } d4 
  d8 e8 fis4 e4. d8   %5
  d4 c4 b4 e4 
  d4 e4 f4 e8 fis8 
  g8 fis8 g8 a8 fis4 d4 
  g4 g4 a4 b4 
  b8 gis8 a4 gis4 b4   %10
  a4 e8 c8 d8 e8 f4 
  e4 d4 c4 a'4 
  d,8 e8 f4 e8 d8 e4~ 
  e8 d4 cis8 d4 d8 c8 
  b8 a8 g8 fis8 g4 g'4   %15
  fis4 g4 g4 g4~ 
  g4 fis4 d4 
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 b4 
    c8 b8 a4 e'4 a,8 b8 
    c8 e8 d4 d4 d4 
    d4 g,8 a8 b8 g8 e'4 
    a,8 b8 c4 b4 } b4 
  a4 d4 g,4 a4~   %5
  a8 g4 fis8 g4 g4 
  g4 g4 g4 g4 
  g8 a16 b16 a4 a4 g4~ 
  g8 d'8 c8 b8 a4 e'4 
  e2 e4 e8 d8   %10
  c8 b8 a4. gis8 a8 b8 
  gis8 a4 gis8 a4 e8 fis8 
  g4 d'8 c8 bes4 a4 
  a4 g4 fis4 fis4 
  g4 d'4 d4 e4   %15
  d2 e8 d8 c4 
  d8 e8 d8 c8 b4 
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 g4 
    c,4 d4 e4 fis8 g8 
    a4 d,4 g4 fis4 
    g8 fis8 e4 b4 c4 
    d2 g,4 } g'4 
  fis4 e8 d8 e4 fis4   %5
  g4 d4 g,4 c4 
  g'8 f8 e8 d8 c8 b8 c8 d8 
  e4 a,4 d4 g8 a8 
  b4 c4 fis,4 gis4 
  a4 a,4 e'4 gis,4   %10
  a8 b8 c8 a8 f'4 e8 d8 
  e4 e,4 a4 a4 
  b8 cis8 d2 cis4 
  d4 a4 d4 b4 
  e4 d4 g8 fis8 e4   %15
  b8 a8 g4 c8 b8 a4 
  d8 c8 d4 g,4 
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