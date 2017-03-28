\version "2.10.33"

\header {
  title = "249 - Allein Gott in der Höh' sei Ehr'"
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
    g8 e8 fis4 g4 } g4 
  a4 b4 c4 b4   %5
  a8. b16 b4 a4 a4 
  b4 c4 d4 c4 
  b4 a4 b4 g4 
  a4 b4 c8 b8 a4 
  g8 e8 fis4 g   %10
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    g4 g4 fis4 e8 fis8 
    g8 fis8 e4 dis4 e4 
    d8 e8 fis4 g4. fis8 
    g4 d4 d4 } d8 e8 
  fis4 gis4 a8 a,8 d8 e8   %5
  a,8 a'4 gis8 e4 d4 
  d4 g4 fis8 gis8 a4 
  g8 fis8 e4 dis4 b8 cis8 
  d8 e8 fis4 e4. d16 c16 
  g'4 d d  %10
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 b8  c8 
    d4 e8 g,8 a8 b8 c8 d8 
    e8 b8 c4 fis,4 g8 a8 
    b8 c8 d4 e4. d16 c16 
    d8 c16 b16 a4 b4 } b4 
  a4 d4 e8 f4 e8   %5
  e16 d16 e8 f8 e16 d16 c4 a4 
  g4 g8 a8 b4 e,4 
  e'8 b8 c4 fis,4 g4 
  fis4 b8 a8 g4. fis8 
  d'8 c16 b16 a4 b   %10
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 g4 
    g'8 fis8 e4 d4 a4 
    e'8 d8 c4 b4 e8 fis8 
    g4 fis4 e8 d8 c8 a8 
    b8 c8 d4 g,4 } g'4 
  d8 d'8 c8 b8 a4 g4   %5
  f8 c8 d8 e8 a,4 fis'4 
  g8 fis8 e4 b4 c8 d8 
  e8 d8 c4 b4 e4 
  d4 dis4 e8 d8 c8 a8 
  b8 c8 d4 g,   %10
  
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