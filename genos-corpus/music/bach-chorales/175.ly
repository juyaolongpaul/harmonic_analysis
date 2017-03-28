\version "2.10.33"

\header {
  title = "175 - Jesus, meine Zuversicht"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key c \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    g4 e4 a4 b4 
    c4 c4 b2 
    a8 b8 c4 g4 e4 
    f4 e4 d2 
    c1 }  %5
  e4 fis4 gis4 a4 
  b4 b4 a2 
  b4 c4 d4 e4 
  d4. c8 c2 
  
}

alto = \relative c' {
  \repeat volta 2 {
    e4 c8 d8 e4 e4 
    e4 a2 gis4 
    a4 g8 e8 d4. g,8 
    c8 d4 c8 c4 b4 
    g1 }  %5
  c4 d4. f8 e8 c8 
  f8 d8 b8 e16 d16 c2 
  g'4 g4 g4 g4 
  a8 f8 d8 g16 f16 e2 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    c4 g4 c4 b4 
    a4 e'4 e2 
    c4. g4 b8 g8 e8 
    a8 g8 g4 a4 g8 f8 
    e1 }  %5
  g4 a4 b8 d8 c8 a8 
  a4. gis8 a2 
  d4 e4 d4 c4 
  c4 b4 g2 
  
}

baixo = \relative c {
  \repeat volta 2 {
    c4 c8 b8 a4 gis4 
    a8 b8 c8 d8 e2 
    f4 e4 b8 g8 c4 
    a8 b8 c4 f,4 g4 
    c,1 }  %5
  c'8 c,8 c'4 b4 c8 f8 
  d8 b8 e4 a,2 
  g'8 f8 e8 d8 c8 b8 c8 e8 
  f8 d8 g4 c,2 
  
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