\version "2.10.33"

\header {
  title = "99 - Helft mir Gotts Güte preisen"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 a4 
    a4 b4 c4 d4 
    b4. c8 d4 e4 
    c4 a4 b4 b4 
    a2. } e'4 
  d4 c4 b4 a4   %5
  b2. b4 
  c4 d4 e4 e4 
  d2 b4 b4 
  c4 d4 e4 e4 
  d4. c8 b4 e4   %10
  d8 c8 b8 a8 b4 b4 
  a2. 
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    f4 f4 g4 a4 
    g2 a4 e4 
    e4 a4 a4 gis4 
    e2. } e8 fis8 
  gis4 a8 g8 f4 e8 fis8   %5
  gis2. gis4 
  a4 b4 c8 b8 a8 g8 
  a8 g8 fis4 g4 g4 
  g4 g8 f8 e8 f8 g4 
  f8 g8 a4 gis4 b4   %10
  a4 e4 f4 e8 d8 
  cis2. 
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 c4 
    c4 d4 g,4 d'4 
    d4 e4 d4 b4 
    c8 d8 e4 f4 e8 d8 
    c2. } g8 a8 
  b4 e8 c8 d8 b8 c4   %5
  b2. e4 
  e4 f4 g4 c,8 b8 
  a4 d4 d4 d4 
  g,4 a4 ais4 a8 g8 
  a2 e'4 e4   %10
  e,4 a4 a4 gis4 
  e2. 
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 a8  g8 
    f4 e8 d8 e4 fis4 
    g4 fis8 e8 fis4 gis4 
    a8 b8 c8 a8 d,4 e4 
    a,2. } c'4 
  b4 a4 gis4 a4   %5
  e2. e4 
  a4 g8 f8 c'4 e,4 
  fis4 e8 d8 g4 g8 f8 
  e4 d4 cis4 d8 e8 
  f4 e8 d8 e4 gis4   %10
  a4 c,4 d4 e4 
  a,2. 
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