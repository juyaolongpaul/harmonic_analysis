\version "2.10.33"

\header {
  title = "328 - Liebster Jesu, wir sind hier"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    b4 a8 g8 a4 d4 
    b4 g4 a2 
    g4 g4 g4 g4 
    a4 b4 a2 
    g1 }  %5
  d4 d4 e4 fis4 
  g4 a8 g8 fis8 e8 d4 
  g4 g4 g4 g4 
  a4 b4 a2 
  g1   %10
  
}

alto = \relative c' {
  \repeat volta 2 {
    d4 d4 d4 fis4 
    d4 d8 e8 fis2 
    e4 f4 e4 d4 
    e8 fis8 g4 g4 fis4 
    d1 }  %5
  b4 a4 b8 cis8 d4 
  e4 e4 d4 a4 
  d8 b8 c8 d8 e4 e4 
  e8 fis8 g4. e8 fis4 
  d1   %10
  
}

tenor = \relative c' {
  \repeat volta 2 {
    g8 a8 b4 a4 a4 
    b8 c8 d4 d2 
    b4 b4 c4 g4 
    c4 d4 e8 c8 a8 d16 c16 
    b1 }  %5
  g4 a4 g4 a4 
  b4 a4 a8 g8 fis4 
  g4. f8 e8 fis8 g8 a16 b16 
  c4 d4 d4. c8 
  b1   %10
  
}

baixo = \relative c {
  \repeat volta 2 {
    g4 g'4. fis8 e8 d8 
    g8 a8 b8 g8 d2 
    e,8 e'4 d8 c,8 c'4 b8 
    a4 g8 b8 c8 a8 d4 
    g,1 }  %5
  g8 g'4 fis8 e4 d4~ 
  d4 cis4 d4 d4 
  b8 g8 a8 b8 c8 d8 c8 b8 
  a4 g4 d'2 
  g,1   %10
  
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