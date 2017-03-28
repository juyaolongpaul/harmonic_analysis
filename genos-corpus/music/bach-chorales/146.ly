\version "2.10.33"

\header {
  title = "146 - Wer nur den lieben Gott läßt walten"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    a4 b4 c4 b4 
    a4 b4 gis4. fis8 
    e4 g4 g4 f4 
    e4 a4 a8 b8 gis4 
    a2. } b4   %5
  c4 d4 e4 e4 
  d8 e16 f16 d4 c4 e4 
  d4 c4 b4 a8 b8 
  c8 b8 b4 a
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 c8  d8 
    e4 e4 e8 fis8 gis4 
    a8 g8 f4 e2 
    b4 e4 e4 d4 
    d8 cis8 d8 dis8 e8 fis8 e4 
    e2. } gis4   %5
  a4 g4 g4 a4 
  a4 g8 f8 e4 e4 
  fis8 gis8 a4 g8 f8 e4 
  fis4 e8 d8 cis
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 a8  b8 
    c8 b8 a8 gis8 a4 d,4 
    e8 e'8 d8 c8 b4. a8 
    gis4 b4 c4 d8 c8 
    b8 a8 a8 b8 c8 b4 c16 d16 
    c2. } e4   %5
  e4 d4 c4 c4 
  c4 b4 g4 a4 
  a8 e'8 e8 ees8 d4 c8 b8 
  a4 gis4 e
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 a4 
    c,4 e4 a,4 b4 
    c4 d4 e2 
    e4 e4 a4 b8 a8 
    gis8 g8 fis8 f8 e8 dis8 e4 
    a,2. } e'4   %5
  a4 b4 c8 b8 a8 g8 
  f8 d8 g4 c,4 cis4 
  d8 e8 f8 fis8 g8 gis8 a4 
  dis,4 e4 a,
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