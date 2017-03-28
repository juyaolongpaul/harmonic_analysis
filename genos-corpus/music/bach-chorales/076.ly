\version "2.10.33"

\header {
  title = "76 - Freu' dich sehr, o meine Seele"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}

soprano = \relative c'' {
  \repeat volta 2 {
    g4 a4 b4 a4  
    g4 fis4 e2  
    d2 g4 a4  
    b4 c4 b4 a4  
    g1  }  %5
  b4 c4 d4 c4  
  b4 a4 b2  
  d4 d4 e4 d4  
  c4 b4 a2  
  b8 c8 d4 c4 b8 a8    %10
  g4 a4 b2  
  g2 b8 c8 d4  
  c4 b4 a4 b8. c16  
  a2 g2  
  
}

alto = \relative c' {
  \repeat volta 2 {
    d4 d4 g4 cis,4  
    d8 e8 e8 d8 d4 cis4  
    a2 e'4 e8 d8
    d4 a'4 a8 g8 g8 fis8
    d1  }  %5
  g4 g8 f8 f4 e8 fis8  
  gis4 a8 a,8 e'2
  d4 g8 f8 e4 fis4  
  g8 d4 e8 fis2  
  g4 a4 g4 fis4    %10
  fis8 e8 e8 fis8 g4 fis4  
  e2 gis4 a8 e8  
  e8 fis8 g4 fis4 g4 ~ 
  g8 fis16 e16 fis4 d2  
}

tenor = \relative c' {
  \repeat volta 2 {
    b4 d4 d8 e8 e8 fis8  
    b,8 a8 a4 b4 a8 g8 
    fis2 b4 a4  
    a8 g8 g8 fis8 d'4 d8 c8 
    b1  }  %5
  d4 a4 a8 gis8 a4  
  e8 e'4 d8 gis,2 
  g8 a8 b4 b8 a8 a8 b8  
  b8 a8 b8 cis8 d2  
  d4 d4 g,8 a8 b4    %10
  b4 c8 a8 e'4. dis8 
  b2 e4 a,8 b8  
  c4 d4 d4 d4  
  d4. c8 b2 
}

baixo = \relative c' {
  \repeat volta 2 {
    g4 fis4 g4 a4  
    b8 cis8 d4 g,4 a4  
    d,2 e4 fis4  
    g4 a4 b8 c8 d8 d,8  
    g1  }  %5
  g,4 a4 b4 c8 d8  
  e4 f4 e2  
  b8 a8 g4 c4 d4  
  e8 fis8 g4 d2  
  g4 g8 fis8 e4 dis4    %10
  e8 d8 c4 g'8 a8 b4
  e,2 e4 fis8 gis8  
  a4 b8 c8 d,8 c8 b8 g8  
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