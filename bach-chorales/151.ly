\version "2.10.33"

\header {
  title = "151 - Meinen Jesum laß'ich nicht, Jesus"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    b4 b4 b4 d4 
    c4 c4 b2 
    b4 b4 a4 b4 
    c4 b4 a2 
    g1 }  %5
  b4 b4 a4 a4 
  g4 g4 fis2 
  e'4 d4 c4 b4 
  a4 a4 g2 
  
}

alto = \relative c'' {
  \repeat volta 2 {
    g4 g8 a8 g4 g4~ 
    g8 fis16 e16 fis4 g2 
    g4 g4. fis8 g4 
    a4. g4 fis16 e16 fis4 
    d1 }  %5
  g4 g4 g4 fis4~ 
  fis4 e4 dis2 
  e8 fis8 g4. fis8 g4~ 
  g8 fis16 e16 fis4 d2 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    d4 e8 fis8 e4 d4 
    e8 c8 a8 d8 d2 
    d4 d4 d4 d4 
    e8 d8 d4 e8 a,8 d8 c8 
    b1 }  %5
  d4 d4 d4. c8 
  b4 b4 b2 
  g'8 a8 d,4 e8 c8 d4 
  e8 a,8 d8 c8 b2 
  
}

baixo = \relative c' {
  \repeat volta 2 {
    g8 fis8 e8 dis8 e8 d8 c8 b8 
    a4 d4 g,2 
    g8 a8 b8 c8 d4 g4~ 
    g8 fis8 g8 b,8 c4 d4 
    g,1 }  %5
  g8 a8 b8 c8 d8 e8 fis8 dis8 
  e8 fis8 g8 a8 b2 
  c,8 c'4 b8 a,8 a'4 g8 
  c,4 d4 g,2 
  
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