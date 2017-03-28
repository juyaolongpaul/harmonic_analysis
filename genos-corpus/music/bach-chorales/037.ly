\version "2.10.33"

\header {
  title = "37 - Jesu, der du meine Seele"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    e4 e4 b4 c4 
    d4 c8 b8 b4 a4 
    c4 c4 b4 a4 
    gis4 a4 b2 }
  b4 c4 d8 c8 b4   %5
  c4 b8 a8 a4 g4 
  c4 d4 e8 d8 c4 
  f4 e4 d4 c4 
  b4 c4 d4 e4 
  d4 c4 b2   %10
  a4 b4 c8 b8 d4 
  c4 b8 a8 a2 
  
}

alto = \relative c'' {
  \repeat volta 2 {
    a8 b8 c4 b4 a8 gis8 
    a8 b8 a4. gis8 e4 
    e4 a8 g8 f8 e4 d8 
    d8 e16 f16 e8 fis8 gis2 }
    gis4 a4 a4 g4   %5
  g8 fis8 g4 fis4 d4 
  c4 g'4 g4 a8 g8 
  a8 b8 c4 b4 g4 
  g4 f8 e8 d4 g8 a8 
  b4 a4 gis2   %10
  a8 g8 f8 e8 e8 f8 e8 d8 
  e4~ e8. d16 cis2 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    c8 b8 a4 e'4 e4 
    d8 b8 c8 d8 e4 c4 
    c8 b8 a4. gis8 a4 
    b8 c16 d16 c8 e16 dis16 e2 }
  e4 e4 d4 d4   %5
  c8 d8 e4 d8 c8 b4 
  e4 b4 c8 b8 a4 
  d4 g,4 g'8 f8 e4 
  d4 c4 g'8 f8 e4 
  f4 e4 e2   %10
  e4 d4 c8 d8 gis,8 a8 
  a4 gis4 e2 
  
}

baixo = \relative c {
  \repeat volta 2 {
    a4 a'4 gis4 fis8 e8 
    fis8 gis8 a4 e4 a,4 
    a'8 g8 f8 e8 d8 e8 f4 
    b,4 c8 a8 e'2 }
  e4 a8 g8 fis8 d8 g8 fis8   %5
  e4 d8 c8 d4 g,4 
  a'4 g8 f8 e4 f8 e8 
  d4 e8 f8 g4 c,4 
  g'4 a4 b4 c8 b8 
  a8 gis8 a4 e2   %10
  c8 b8 a8 gis8 a8 d8 e8 f8 
  e8 d8 e4 a,2 
  
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