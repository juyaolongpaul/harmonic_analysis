\version "2.10.33"

\header {
  title = "248 - Sei Lob und Ehr' dem höchsten Gut"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 d4 
    d4 d4 d8 e8 f4 
    e4 d4 c4 d4 
    b8 a8 g8 a8 b4 cis4 
    d4 e4 d4 } d4 
  g4 fis4 e4 fis4   %5
  g8 fis8 e4 d4 d4 
  g4 d4 e4 b8 c8 
  d4 c4 b4 b4 
  a8 b8 c4 b4 a4 
  e4 fis4 g4   %10
  
}

alto = \relative c'' {
  \repeat volta 2 {
    \partial 4 a4 
    g4 g8 fis8 g4 c,8 d8 
    e8. f16 g4 g4 a4 
    g4 d4 g8 fis8 e4 
    a8 fis8 g4 fis4 } b4 
  b4 a8 b8 cis4 cis4   %5
  d4 a8 g8 fis4 g4 
  g4 g4 g4 g4 
  a8 b8 e,8 a8 gis4 g4 
  a4 g8 a8 b8 g8 e4 
  e4 d4 d4   %10
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 d8 c8 
    b4 a4 b4 a4 
    g8 c8 c8 b8 e4 d4 
    d8 c8 b8 a8 g4 a4 
    a8 d4 cis8 d4 } d4 
  d8 e8 fis8 g8 a8 g8 fis8 e8   %5
  d4 cis4 a4 b4 
  b8 c8 d4 c4 g8 e'8 
  d8 b8 c8 e16 dis16 e4 d4 
  d4 e4 d4 c8 b8 
  a2 b4   %10
  
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 fis,4 
    g4 d4 g,4 a8 b8 
    c4 g4 c4 fis4 
    g4. fis8 e8 fis8 g8 a8 
    fis8 d8 a'4 d,4 } g8 a8 
  b8 cis8 d4 a4 ais4   %5
  b8 g8 a4 d,4 g8 fis8 
  e8 d8 c8 b8 c8 d8 e4 
  fis8 gis8 a4 e4 g4 
  fis4 e8 fis8 g4 c,4 
  cis4 d4 g,4   %10
  
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