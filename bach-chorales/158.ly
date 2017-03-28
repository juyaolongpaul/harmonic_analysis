\version "2.10.33"

\header {
  title = "158 - Der Tag, der ist so freudenreich"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 g4 
    g4 g4 a4 b4 
    c8 b8 a4 g2 
    a4 a4 e4 fis4 
    g4 a4 g4 } g4 
  d'4 d4 e4 d8 c8   %5
  b4 cis4 d4 d4 
  d4 d4 e4 d8 c8 
  b4 a4 g2 
  e4 fis4 g4 e4 
  d4 cis4 d2   %10
  g4 g4 a4 b4 
  c8 b8 a4 g2 
  a4 a4 e4 fis4 
  g8 fis8 e4 d4 f4 
  e4 c4 e4 fis4   %15
  g4 a4 g
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    e4 e4. fis8 g4 
    g4. fis8 d2 
    fis4 e8 d8 e4. d16 c16 
    b8 g8 d'4 d4 } d4 
  d4 g8 a8 b8 c8 b8 a8   %5
  g4 g4 a4 a4 
  g4 g4 g4 a4~ 
  a8 g4 fis8 d2 
  c4 c4 d8 b4 a8 
  a8 b8 a4 a2   %10
  e'4 e4 e8 fis8 g4 
  g4. fis8 d2 
  d4 d4 cis4 d4 
  d4 cis4 a4 d4 
  c4 g4 c4 d4   %15
  d4. c8 b4
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 b4 
    b4 c4 c4 d8 f8 
    e4 a,8 d16 c16 b2 
    d4 c8 b4 a4 b16 a16 
    g4. fis8 b4 } b4 
  a8 c8 b8 a8 g4 d'4   %5
  d4 e4 fis4 d8 c8 
  b8 c8 d8 b8 c8 b8 a8 d8 
  d4 d8. c16 b2 
  g4 a4 d,8 g4 fis16 e16 
  fis4 e8 fis16 g16 fis2   %10
  b4 c4 c4 d8 f8 
  e4 a,8 d16 c16 b2 
  fis8 g8 a4 a4 a4 
  b4 e,8 a16 g16 fis4 g4 
  g4 e4 a4 a4   %15
  g4 fis4 g 
  
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 g8  fis8 
    e8 d8 c8 b8 a4 g4 
    c4 d4 g,2 
    d'4 a8 b8 c4 d4 
    e4 d4 g,4 } g'4 
  fis4 g8 fis8 e4 fis4   %5
  g8 fis8 e4 d4 fis4 
  g8 a8 b8 g8 c4 fis,4 
  g4 d4 g,2 
  c8 b8 a4 b4 cis4 
  d4 a4 d,2   %10
  e'8 d8 c8 b8 a4 g4 
  c4 d4 g,2 
  d8 e8 fis8 g8 a8 g8 fis4 
  e4 a4 d4 b4 
  c4. b8 a4 d8 c8   %15
  b8 g8 d'4 g,
  
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