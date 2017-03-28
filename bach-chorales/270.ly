\version "2.10.33"

\header {
  title = "270 - Befiehl du deine Wege"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    a4 g4 f4 e4 
    d2 e4 b'4 
    c4 c4 b4 b4 
    a2. } c4 
  b8 c16 d16 g,4 a4 b4   %5
  c2 c4 g4 
  a4 g4 a4 f4 
  e2 r4 c'4 
  b4 d4 c4 b4 
  a2 b4 e,4   %10
  f4 e4 d4 g4 
  e2. 
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    e8 d8 d4 d4 c8 b8 
    a2 b4 e4 
    e4 f4 f4 e8 d8 
    cis4 d4. c8 } a'8 g8 
  f4 e4 e8 c8 f4   %5
  g4. f8 e4 c4 
  c4 c8 bes8 a4 a4 
  a2 r4 e'4 
  e4 d4 e8 fis8 g8 gis8 
  a4 a,4 e'4 e8 cis8   %10
  a8 b8 c4 d4 d8 c8 
  b4. b16 a16 b4 
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 g8  b8 
    a8 c8 b4 c8 g8 g4 
    a8 f8 d8 d'8 d8 b8 gis4 
    a4. c8 d8 d,8 g8 f8 
    e4 f8. g16 a4 } e'4 
  d4 d8 c8 c4 d4   %5
  g,4 a8 b8 c4 c4 
  f,4 g4 e4 f8 g8 
  a2 r4 a4 
  b4 b4 b8 a8 e'8 d8 
  c8 cis8 d4 g,4 a8 g8   %10
  f8 g8 a4 a4 b4 
  b8 gis8 e8 c'8 b4 
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 e4 
    f4 g4 a,8 b8 c4 
    f,2 e4 e'4 
    a8 g8 f8 e8 d8 b8 e4 
    a,2. } a'4 
  d,4 e4 f4 d4   %5
  c2 c4 e4 
  f4 c4 cis4 d4 
  a2 r4 a4 
  e'4 b4 c8 d8 e4 
  f2 e4 a4   %10
  d,4 a8 g8 f8 fis8 g4 
  gis4 a4 e4 
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