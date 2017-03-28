\version "2.10.33"

\header {
  title = "345 - O Haupt voll Blut und Wunden"
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
    c4 c4 b8 c16 d16 b4 
    a2. } c4 
  b8 a8 g4 a4 b4   %5
  c2 c4 g4 
  a4 g4 f8 e8 f4 
  e2. c'4 
  b8 c8 d4 c4 b4 
  a2 b4 e,4   %10
  f4 e4 d4 g8 f8 
  e2. 
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 e8  d8 
    c8 d8 d8 e8 e8 d8 d8 c8 
    c4 b8 a8 b4 e4 
    e4 a4 a4 gis4 
    e2. } a8 g8 
  f4 e8 d8 c4 f4   %5
  f4 e8 d8 e4 e4 
  f4 bes,8 a8 a8 g8 a8 d8 
  cis2. d4 
  d4 g4 g8 fis8 g4 
  g4 fis4 g4 g4   %10
  c,8 d8 c4 f,4 d'4 
  d4 c4 b4 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 gis4 
    a4 b4 c8 d8 g,4 
    a8 gis8 a4 gis4 gis4 
    a8 b8 c4 f4 e8 d8 
    c2. } e4 
  d8 c8 b4 a8 g8 f8 g8   %5
  a4 g8 f8 g4 c4 
  c8 d8 e4 d8 cis8 d8 gis,8 
  a2. a4 
  g8 a8 b4 c8 d8 e4 
  e4 d4 d4 c4   %10
  c8 b8 b8 a8 a4 g8 a8 
  b4 a4 gis4 
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 e4 
    f4 g4 a8 b8 c4 
    f,2 e4 e4 
    a8 g8 f8 e8 d8 b8 e4 
    a,2. } a'4 
  d,4 e4 f8 e8 d4   %5
  a8 b8 c4 c4 c4 
  f8 e8 d8 cis8 d4 d4 
  a2. fis'4 
  g4 g8 fis8 e4. d8 
  c8 a8 d4 g,4 c8 b8   %10
  a8 b8 c4. b4 a8 
  gis4 a4 e'4 
  
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