\version "2.10.33"

\header {
  title = "148 - Danket dem Herrn, heuf und allzeit"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 b4 
    b8 c8 d4 b4 g4 
    a4 b4 c4 b4 
    a4 g4 g4. a8 
    b4 b4 a8 g8 a4 
    g2. } b4   %5
  a4 b4 g4 e4 
  fis4 g4 a4 b4 
  g4 g4 g4. a8 
  b4 b4 a8 g8 a4 
  g2. 
}

alto = \relative c'' {
  \repeat volta 2 {
    \partial 4 g4 
    g4 a4 g4 g4 
    f8 e8 d4 e4 d4~ 
    d8 c8 b4 e2 
    d4 d4 d4. c8 
    b2. } g'4   %5
  fis4 fis4 e8 d8 cis4 
  d4. cis8 d4 dis4 
  b4 e4 d4 e8 fis8 
  g4 d4 d4. c8 
  b2. 
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    d4 d4 d4 d8 c8 
    c4 g4 g4 g4 
    fis4 g4 c2 
    b8 a8 g4 g4 fis4 
    d2. } d'4   %5
  d8 c8 b4 b4 a4 
  a4 g4 fis4 fis4 
  g4 c4 b8 c8 b8 a8 
  g4 g2 fis4 
  d2. 
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 g4 
    g'4 fis4 g4 e4 
    f4 g4 c,4 g4 
    d'4 e8 d8 c8 d8 e8 fis8 
    g4 b,8 c8 d4 d4 
    g,2. } g4   %5
  d'4 dis4 e4 a8 g8 
  fis4 e4 d4 b4 
  e8 d8 e8 fis8 g8 a8 g8 fis8 
  e4 b8 c8 d2 
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