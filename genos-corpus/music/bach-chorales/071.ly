\version "2.10.33"

\header {
  title = "71 - Ich ruf' zu dir, Herr Jesu Christ"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key e \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 b4 
    g4 fis8 g16 a16 g8 fis8 e8 fis8 
    g8. a16 a4 b4 b8 c8 
    d8 c8 b8 a8 g4 a8 b8 
    c2 b4 } d4 
  e4 d8 c8 b8 a8 g8 a8   %5
  b8 c8 a4 g4 b4 
  b4 b4 a4 g8 fis8 
  fis2 e2 
  g4 fis4 e2 
  d4 d4 g4 g4   %10
  a4 a4 b4 c4 
  b4 a4 g8 fis8 e8 fis8 
  g4 fis4 e4
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 fis4 
    e4 dis4 e8 dis8 e8 dis8 
    e4. fis8 dis4 e4 
    d4 d4 e4 d4 
    c8 e4 dis8 e4 } d8 g16 f16 
  e4 fis4 fis4 g4   %5
  g8 a8 fis4 d4 d8 e8 
  fis8 a8 g8 fis8 e8 dis8 e4 
  e4 dis4 e2 
  e4. d8~ d8 cis16 b16 cis4 
  a4 a4 d4 c8 d8   %10
  e8 c8 e8 d8 e4 e4 
  e8 dis8 e8 fis8 e8 dis8 e4 
  e4 dis4 b4 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 b4 
    b8 c4 b16 a16 b2~ 
    b8 b8 a16 g16 fis16 e16 fis4 g4 
    a4 g8 f8 e4 fis8 gis8 
    a8 g8 fis4 g4 } g4 
  g8 c8 b8 a8 b4. c8   %5
  d8 e8 d8 c8 b4 g4 
  fis4 e4 a8 b8 c4 
  b8 a4 g8 g2 
  a4 a4 b4 a8 g8 
  fis4 fis4 g8 f8 e8 g8   %10
  c8 e8 a,4 gis4 a8 g8 
  fis4 e8 b'8 b4. c8 
  b4. a8 gis4 
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 dis4 
    e2~ e8 fis8 g8 fis8 
    e8 d8 c4 b4 e,4 
    fis4 g4 c8 d8 c8 b8 
    a2 e4 } b'4 
  c4 d4 dis4 e4   %5
  d8 c8 d4 g,4 g'4 
  dis4 e8 d8 c4 b8 a8 
  b2 c2 
  cis4 d4 g,4 a4 
  d4 d8 c8 b4 c8 b8   %10
  a8 g8 f4 e4 a4 
  b4 cis8 dis8 e8 fis8 g8 a8 
  b4 b,4 e,4
  
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