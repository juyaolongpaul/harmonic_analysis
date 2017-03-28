\version "2.10.33"

\header {
  title = "129 - Keinen hat Gott verlassen"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key e \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 b4 
    b4 b4 b4 e4 
    d4. c8 b4 b8 c8 
    d4 fis,4 g4 a4 
    b2. } a4 
  a4 a4 a4 a4   %5
  b2 a4 a4 
  b8 cis8 d4 d4 cis4 
  d2. d4 
  b4 b4 c4 d4 
  e2 b4 b4   %10
  a4 g8 fis8 fis4. e8 
  e1 
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    fis4 e8 dis8 e8 fis8 g4~ 
    g4 fis4 d4 d4 
    d4 d8 c8 b4 e4 
    dis2. } e4 
  d4 d8 cis8 d8 e8 fis4   %5
  g2 fis4 a4 
  g4 a4 a4. g8 
  fis2. fis4 
  d4 g4 g4 f4 
  e2 e4 e4   %10
  e4 e4 e4 dis4 
  b1 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 g4 
    fis8 g8 a4 g4 c8 b8 
    a8 g8 a4 g4 g4 
    a4 a4 g8 fis8 e4 
    fis2. } e4 
  fis8 e8 fis8 g8 a4 d4   %5
  d2 d4 d4 
  d4 d8 e8 fis4 e4 
  d2. a4 
  g4 d'4 e4 b4~ 
  b4 a4 g4 b4   %10
  e,4 c'4 b8 g8 a4 
  gis1 
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 e4 
    dis4 b4 e8 d8 c4 
    d2 g,4 g'4 
    fis8 e8 d4 e8 d8 c4 
    b2. } cis4 
  d4 d8 e8 fis4 d4   %5
  g,8 a8 b8 c8 d4 fis4 
  g4 fis8 g8 a4 a,4 
  d2. d4 
  g8 a8 g8 f8 e8 f8 e8 d8 
  c8 b8 c8 d8 e4 g,4   %10
  c4 b8 a8 b4 b4 
  e,1 
  
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