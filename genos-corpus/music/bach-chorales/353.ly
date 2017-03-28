\version "2.10.33"

\header {
  title = "353 - Der Herr ist mein getreuer Hirt"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 g8  a8 
    b4 c4 d4 c4 
    b4 a4 b4 b4 
    b4 a8 b8 c8 b8 a4 
    g8. a16 a4 g4 } g4 
  a4 b4 c4 b4   %5
  a4. gis8 a4 a4 
  b4 c4 d4 c4 
  b4 a4 b4 b4 
  b4 a8 b8 c8 b8 a4 
  g8. a16 a4 g4   %10
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 d8  fis8 
    g4 g4 fis4 e8 fis8 
    g4 g8 fis8 g4 g4 
    g4 fis8 g8 a8 g8 fis4 
    g4 g8 fis d4 } d4 
  d4 d4 e4 e4   %5
  e8 c8 f8 e8 e4 fis4 
  g4 g8 f e4 e8 fis8 
  g4 g8 fis16 e16 dis4 e4 
  d8 e8 fis8 g8 a8 g4 fis8 
  e8 d8 e8 d8 d4   %10
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 b8  c8 
    d4 e4 a,4 a4 
    d4 d4 d4 e4 
    d4 d4 e8 d8 d8 c8 
    b4 e8 d16 c16 b4 } b4 
  a4 g4 g4 gis4   %5
  a4 d4 c4 a4 
  d4 c4 b4 c8 d8 
  e4 e,4 fis4 g8 a8 
  b8 c8 d8 g,8 fis8 g16 fis16 e8 fis8 
  g4. fis8 b4   %10
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 g4 
    g'8 fis8 e4 d4 a4 
    b8 c8 d4 g,4 e'4 
    b8 c8 d4 a8 b8 c8 d8 
    e8 d8 c8 d8 g,4 } g'4 
  fis4 f4 e4 d4   %5
  c8 f8 d8 e8 a,4 d8 c8 
  b4 a4 gis4 a4 
  e'8 d8 c4 b4 e8 fis8 
  g4 fis8 e8 dis8 e8 cis8 dis8 
  e8 b8 c8 d8 g,4   %10
  
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