\version "2.10.33"

\header {
  title = "315 - O Gott, du frommer Gott"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key e \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 b4 
    b4 e4 d4 c16 b16 c8 
    b2. b8 c8 
    d4 g,4 a4 b4 
    a2 g4 } a4 
  a4 b4 c8 e8 d8 c8   %5
  b2. b4 
  a4 d8 e16 fis16 e4 d8 cis8 
  d2. d4 
  d4 b4 c4 d4 
  e2. a,4   %10
  b8 c16 d16 c8 b8 b4 a8 g8 
  g2. 
  
}

alto = \relative c'' {
  \repeat volta 2 {
    \partial 4 g4 
    fis4 g8 fis8 g4 fis4 
    g2. g4 
    g8 fis8 e8 fis16 g16 fis4 g4~ 
    g8 e8 fis4 d4 } e4 
  d4 fis4 g4 a4   %5
  g2. g8 fis8 
  e8 d16 cis16 d4 g8 b8 a8 g8 
  fis2. a4 
  g4 g4 g4 g4 
  g2. fis4   %10
  g4 g4 g4 fis4 
  d2. 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 g8  a8 
    b4. a8 b8 e8 d4 
    d2. d8 c8 
    b8 a8 g4 d'4 d4 
    d4. c8 b4 } cis4 
  d4 d4 c8 b8 a8 d8   %5
  d2. d4 
  cis8 a8 g8 fis8 b8 g'8 fis8 e8 
  d2. fis4 
  g4 d4 e4 f4 
  e2. d4   %10
  d4 d8 e8 d4 c4 
  b2. 
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 e4~
    e8 d4 c8 b8 c8 
    d4 g,2. 
    g8 a8 b4 c4. b8 
    a8 g8 d'2 g,4 } g'4~
  g8 fis e d e4   %5
  fis4 g2. 
  g,4 g'8 fis8 b8 a8 g8 e8 
  a4 d,2. 
  d'8 c8 b8 a8 g8 f8 e8 d8 
  c8 b8 c2.   %10
  d4 g,8 a8 b8 c8 d4 
  d4
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