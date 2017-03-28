\version "2.10.33"

\header {
  title = "67 - Freu' dich sehr, o meine Seele"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    g4 a4 b4 a4  
    g4 fis4 e4 d4  
    g4 a4 b4  c4 
    b4 a4 g2  }
  b4  c4 d4  c4   %5
  b4 a4 b2  
  d4 d4 e4 d4  
  c4 b4 a2  
  b8  c8 d4  c4 b8 a8  
  g4 a4 b2    %10
  g2 b8  c8 d4  
  c4 b4 a4 b4  
  a2 g2  
 }

alto = \relative c' {
  \repeat volta 2 {
    d4 d4 d4 d4  
    d8 cis8 d4 cis4 a4  
    e'4 d4 d4  c4
    d4 d8  c8 b2  }
  g'4 fis8 e8 d8 e16 f16 e4    %5
  e4 e4 dis2  
  d4 g4 g4 fis4  
  e4 d8 e8 fis2  
  g4 g4 g8 fis8 g8 fis8  
  e8 d8  c4 b8 cis8 dis4    %10
  e2 e4 f4  
  e8 fis8 g4 fis8 e8 d4  
  d4.  c8 b2  
}

tenor = \relative c' {
  \repeat volta 2 {
    b4 a4 g4 fis4  
    g4 a4 a8 g8 fis4  
    b4 a4 g8 fis8 g4  
    g4 fis4 d2  }
  d'4  c4 fis,8 gis8 a4    %5
  g4  c4 fis,2
  g8 a8 b4  c8 b8 a4  
  g8 a8 b8 cis8 d2  
  d8  c8 b4 a4 d,4
  e4 fis4 g4 fis4    %10
  e2 g4 gis4  
  a4 e4 fis4 g4  
  g4 fis4 d2 
}

baixo = \relative c' {
  \repeat volta 2 {
    g4 fis4 g4 d4  
    e4 fis8 g8 a8 a,8 d4  %
    e4 fis4 g4 e4  
    d8  c8 d4 g,2  }
  g4 a4 b4  c8 d8    %5
  e8 d8  c4 b2  
  b4 g4  c4 d4  
  e8 fis8 g4 d2  
  g4 g,4 a4 b4
  c8 b8 a4 g8 a8 b4    %10
  e,2 e'4 b4 
  c4 cis4 d8  c8 b8 g8  
  d'2 g,2
  
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