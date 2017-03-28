\version "2.10.33"

\header {
  title = "14 - O Herre Gott, dein göttlich Wort"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 g4 
    g4 fis4 e4 d4 
    g4 a4 b4 a4 
    b8 c16 d16 c4 b4 a8 b8 
    c4 b4 a2 
    g2. } g4   %5
  a4 b4 g4 g4 
  a4 b4 g4 e4 
  a4. g8 fis4 g8 fis8 
  e2 d4 d4 
  g4 fis4 e4 d4   %10
  g4 a4 b4 a4 
  d4. c8 b4 a8 b8 
  c4 b4 a2 
  g2. 
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    e4 d4 c4 c4 
    b4 d4 d4 d4 
    d4 e4 d8 e8 fis4 
    g2. fis4 
    d2. } d4   %5
  d4 d4 e4 d4 
  e4 fis4 e4 b4 
  a8 b8 cis4 d4 d4 
  d4 cis4 a4 b4 
  b8 cis8 d4 c8 b8 a4   %10
  d4 d4 d4 d8 e8 
  fis4 g8 a8 d,8 e8 fis4 
  g2. fis4 
  d2. 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 b4 
    b4 b8 a8 g4 fis4 
    g4 fis4 g4 fis4 
    fis4 g8 a8 b8 c8 d4 
    c8 d8 e4 a,8 b8 c4 
    b2. } b4   %5
  fis4 g4 g4 b4 
  a8 g8 fis4 b4 g8 fis8 
  e4 a4 a4 g8 a8 
  b4 a8 g8 fis4 fis4 
  g4 a4. g8 fis4   %10
  g4 fis4 g4 fis4 
  b4. a8 b8 c8 d4 
  c8 d8 e4 a,8 b8 c4 
  b2. 
  
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 g4 
    e4 b4 c4 d4 
    e4 d4 g,4 d'4 
    b4 e,8 fis8 g4 fis4 
    e8 e'8 d8 c8 d2 
    g,2. } g'4   %5
  d4 g,4 c4 b4 
  cis4 dis4 e4 e8 d8 
  cis8 b8 a4 d8 cis8 b8 a8 
  g4 a4 d4 b4 
  e,4 fis8 g8 a8 b8 c4   %10
  b4 d4 g,4 d'8 c8 
  b4 e8 fis8 g4 fis4 
  e4 d8 c8 d2 
  g,2. 
  
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