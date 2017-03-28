\version "2.10.33"

\header {
  title = "1 - Aus meines Herzens Grunde"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key g \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 g4 
    g2 d'4 
    b4. a8 g4 
    g4. a8 b4 
    a2\fermata b4 
    d2 c4     %5
    b4 a2 
    g2\fermata } b4 
  b4 c4 d4 
  d4. c8 b4 
  a2\fermata g4   %10
  b2 c4 
  d2 c4 
  b2. 
  g2\fermata b4 
  d2 c4   %15
  b2 a4 
  g4. a8 b4 
  a2\fermata b4 
  d2 c4 
  b4 a2   %20
  g2\fermata 
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    d4 e4 d4 
    d2 b4 
    e8 d8 e8 fis8 g4 
    fis2 g4 
    d4 e4 fis4     %5
    g2 fis4 
    d2 } g4~ 
  g8 fis8 e8 fis8 g4~ 
  g8 a8 g8 fis8 g4 
  fis2 e4   %10
  e4 fis8 g8 a4 
  a4 g4. fis8 
  g2 f4 
  e2 g4 
  a4. g8 fis4   %15
  g2 fis4~ 
  fis8 e8 e8 fis8 g4 
  fis2 g4 
  a2 g8 fis8 
  g2 fis4   %20
  d2 
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 b4 
    b4 c8 b8 a4 
    g4 fis4 g4 
    c8 b8 c4 d4 
    d2 d4 
    a4 b4 c4     %5
    d4 e4 d8 c8 
    b2 } d4 
  d4 c4 b8 a8 
  b8 c8 d4 d4 
  d2 b4   %10
  g4 b4 e4 
  d2 d4 
  d2. 
  c2 d4 
  d8 c8 b4 c4   %15
  d2 d8 c8 
  b4 c4 d4 
  d2 d4 
  d2 e4 
  e2 d8 c8   %20
  b2 
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 g4 
    g'4 e4 fis4 
    g4 d4 e4 
    c4 b8 a8 g4 
    d'2\fermata g,4 
    fis4 g4 a4     %5
    b4 c4 d4 
    g,2\fermata } g4 
  g4 a4 b4 
  b4. a8 g4 
  d'2\fermata e4~   %10
  e4 d4 c4 
  b4. c8 d4 
  g,8 a8 b4 g4 
  c2\fermata g4 
  fis4 g4 a4   %15
  b4 g4 d'4 
  e8 d8 c8 b8 a8 g8 
  d'2\fermata g4~ 
  g4 fis4 e4 
  e8 d8 c4 d4   %20
  g,2\fermata 
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