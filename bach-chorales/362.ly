\version "2.10.33"

\header {
  title = "362 - Es ist gewißlich an der Zeit"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 g4 
    g8 a8 b4 a4 g4 
    a4 a4 b4 g8 a8 
    b4 c4 d4 c8 b8 
    a8 g8 a4 g4 } b4 
  b4 a4 g4 fis4   %5
  g8 a8 b4 a4 a4 
  g4 fis4 g4 d4 
  g4 a4 b4 g8 a8 
  b4 c4 d4 c8 b8 
  a8 g8 a4 g4   %10
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    e8 fis8 g4 g8 fis8 g4 
    g4 fis4 g4 e8 fis8 
    g8 b4 a16 g16 fis4 g4 
    g4 fis4 d4 } g4 
  g4. fis8 fis8 e8 e8 dis8   %5
  e4 d4 d4 d4 
  d8 c8 d4 d8 c8 b4 
  b4 e4 dis4 b4 
  e4 e4 d8 e16 fis16 g4 
  g4 fis4 d4   %10
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 b4 
    b4 e4 e8 d8 d4 
    e8 c8 a8 d8 d4 e4 
    d4 g,8 a8 b8 c8 d4 
    d4 c8 b8 b4 } d4 
  d4 d8 c8 b8 c8 fis,8 b8   %5
  b8 a8 a8 g8 fis4 fis4 
  g8 a8 b8 c8 b8 a8 g4 
  g4 fis8 e8 fis4 e8 fis8 
  g4. a8 b8 c8 d4 
  d4 c8 b8 b4   %10
  
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 g8  fis8 
    e4. d8 c4. b8 
    c8 a8 d4 g,4 c4 
    g'8 fis8 e4 b'8 a8 g4 
    d'4 d,4 g4 } g8 a8 
  b8 c8 d4 g,8 a8 b4   %5
  e,8 fis8 g4 d4 d'8 c8 
  b4. a8 g4. fis8 
  e8 d8 c4 b4 e4 
  e8 d8 c4 b8 a8 g4 
  d'2 g,4   %10
  
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