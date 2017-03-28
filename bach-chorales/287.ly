\version "2.10.33"

\header {
  title = "287 - Herr, ich habe mißgehandelt"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    a4 e4 a4 b4 
    c4 d4 b2 
    a2 c4 b4 
    a4 b4 c4 d4 
    e1 }  %5
  e4 f4 g4 c,4 
  f4 e4 d2 
  c2 d4 d4 
  e4 a,4 d4 c4 
  b2 a2   %10
  
}

alto = \relative c' {
  \repeat volta 2 {
    e4 e4 e4 d4 
    e4 a4 a4 gis4 
    e2 a4 gis4 
    a4 e4 e4 a4 
    gis1 }  %5
  a4 a4 g4 a4 
  a8 aes8 g4 g4. f8 
  e2 g4 g4 
  g4 fis8 e8 fis8 gis8 a4 
  a4 gis4 e2   %10
  
}

tenor = \relative c' {
  \repeat volta 2 {
    c4 c4 c8 a4 b16 a16 
    gis4 a4 e'4. d8 
    c2 e4 d4 
    e4 e8 d8 c8 b8 a4 
    b1 }  %5
  c4 d4 d8 c8 c4 
  c8 b8 c4 c4 b4 
  g2 b4 b4 
  c8 g8 d'4 d4 e4 
  e4. d8 cis2   %10
  
}

baixo = \relative c {
  \repeat volta 2 {
    a4 a'8 g8 fis4 f4 
    e4 f8 d8 e2 
    a,2 a'4 b4 
    c4 gis4 a8 g8 f4 
    e1 }  %5
  a4 g8 f8 e4 f8 e8 
  d4 e8 f8 g2 
  c,2 g'8 f8 e8 d8 
  c8 e8 d8 c8 b4 a4 
  e'2 a,2   %10
  
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