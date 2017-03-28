\version "2.10.33"

\header {
  title = "317 - Herr, wie du willst, so schick's mit mir"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key c \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 c4 
    b4 c4 d4 d4 
    c4 d4 e4 f4 
    e4 d4 c4 b8 a8 
    b4 c4 d2 
    c2. } c4   %5
  d8 e8 f4 e8 d8 c4 
  d8 c8 b4 a4 b4 
  g4 g4 d'4 e4 
  f8 e8 d4 c4 g'4 
  e4 f8 e8 d4. c8   %10
  b4 c4 d2 
  c2. 
  
}

alto = \relative c'' {
  \repeat volta 2 {
    \partial 4 g4 
    g4. fis8 g4 gis4 
    e8 c8 g'4 g4 g4 
    g4 g8 f8 e4 f8 e8 
    d4 g4 a4 g8 f8 
    e2. } g4   %5
  g4 a4 b4 a4 
  a4. g8 fis4 fis4 
  e4 e8 fis8 g4 e4 
  d4 e4 e4 g4 
  g4 f8 g8 a8 f8 g8 a8   %10
  g4 g4 g4. f8 
  e2. 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    d4 c4 b4 b4 
    c8 f8 b,4 c4 d4 
    c4 b4 c8 e8 d4 
    g4 c,2 b4 
    g2. } e'4   %5
  d4 c4 b4 e4 
  d4 d4 d4 b4 
  b4 b8 c8 d4 a4 
  a4 b4 a4 d4 
  c4 c8 b8 a4 e'4   %10
  d4 c4 c4 b4 
  g2. 
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 c4 
    g'4 a4 g8 f8 e4 
    a4 g4 c4 b4 
    c4 g4 a8 g8 f4~ 
    f4 e4 fis4 g4 
    c,2. } c'4   %5
  b4 a4 gis4 a8 g8 
  fis4 g4 d4 dis4 
  e8 fis8 g8 a8 b4 cis4 
  d4 gis,4 a4 b4 
  c8 b8 a8 g8 f8 d8 e8 f8   %10
  g8 f8 e8 c8 g2 
  c2. 
  
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