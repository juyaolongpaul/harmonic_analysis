\version "2.10.33"

\header {
  title = "246 - Singt dem Herrn ein neues Lied"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    g4 g4 g4 d4 
    g4 a4 b2 
    g8 a8 b8 cis8 d4 fis4 
    e2 d2 }
  e4 d4 c4 e4   %5
  d4 c4 b2 
  a2 a4 gis4 
  a4 b4 c4 c4 
  b2 g4 a4 
  b4 c4 c4 b4   %10
  c2 d4. c8 
  b4 a4 g8 a8 b4 
  a2 g2 
  
}

alto = \relative c' {
  \repeat volta 2 {
    d8 c8 b8 c8 d4 d4 
    e16 fis16 g4 fis8 g2 
    d4 d8 g8 fis4 a4 
    b8 g8 e8 a16 g16 fis2 }
  g8 e8 fis8 gis8 a4 e4   %5
  a8 gis8 a4 a8 gis16 fis16 gis4 
  e2 e4 e4 
  e4 e4 e4 e4 
  e2 e4. d16 c16 
  d8 g4 f16 e16 a4 g4   %10
  g2 g4 fis4 
  d4 d4 e8 fis8 g4~ 
  g8 fis16 e16 fis4 d2 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    b8 c8 d8 c8 b8 c8 b8 a8 
    b4 e8 d8 d2 
    b8 a8 g4 a4 d4~ 
    d8 cis16 b16 cis4 a2 }
  c4 d4 e8 d8 c8 b8   %5
  a8 b8 c8 d16 e16 f8 d8 b8 e16 d16 
  c2 c4 b4 
  c4 b4. a16 gis16 a4~ 
  a4 gis4 b8 c16 b16 a4 
  g8 b8 a16 b16 c8 d4. f8   %10
  e2 d4 a4 
  g4 a4 b8 c8 d4 
  e8 c8 a8 d16 c16 b2 
  
}

baixo = \relative c' {
  \repeat volta 2 {
    g8 a8 b8 a8 g8 a8 g8 fis8 
    e8 d8 c8 d8 g,2 
    g'8 fis8 g8 e8 fis8 e8 d8 e16 fis16 
    g8 e8 a4 d,2 }
  c'4. b8 a4. g8   %5
  f4. e8 d8 b8 e4 
  a,2 a8 c8 e8 d8 
  c8 b8 a8 gis8 a8 b8 c8 d8 
  e2 e8 c8 f4~ 
  f8 e8 a8 g8 f8 d8 g4   %10
  c,2 b8 c8 d4 
  g,8 g'4 fis8 e8 d8 c8 b8 
  c8 a8 d4 g,2 
  
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