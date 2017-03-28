\version "2.10.33"

\header {
  title = "183 - Nun freut euch, lieben Christen, g'mein"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 g4 
    g4 d4 g4 c4 
    b4 a4 g4 g4 
    a8 b8 c4 b4 a4 
    e4 fis4 g4 } g4 
  c4 b4 a4 d4   %5
  d4 cis4 d4 d4 
  e4 c4 d4. c8 
  b4 a4 g4 b4 
  a4 d,4 g4 c4 
  b4 a4 g   %10
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    d4 a4 d4 a'4 
    d,4 d8. c16 b4 b8 cis8 
    d8 f8 e8 fis8 g4 d8 a16 b16 
    c2 b4 } d4 
  g8 a4 g16 fis16 e4 a8 fis8   %5
  e8 g4 fis16 e16 fis4 g4 
  g4 g4 a8 fis8 g8 a8~ 
  a8 g4 fis8 e4 e4~ 
  e8 d16 c16 b4 e8 d8 e8 fis8 
  g4. fis8 d4   %10
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 b8  a8 
    g4 fis4 g4. fis8 
    g4. fis8 d4 g4 
    fis8 gis8 a4 d,8 e8 fis4 
    g4 a4 d,4 } b'4 
  e8 d8 d4 cis4 d4   %5
  b8 e,8 a4 a4 d4~ 
  d8 c16 b16 c8 e8 d8 c8 b8 a8 
  b16 c16 d4 c8 b4 g4 
  a4 g8 a8 b4 a4 
  d4~ d8. c16 b4   %10
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 g8  a8 
    b8 c8 d8 c8 b4 a4 
    g8 b8 d4 g,4 e'4 
    d4 a4 b8 c8 d4 
    c8 b8 a4 g4 } g'8 f8 
  e8 fis8 g4. a16 g16 fis8 b8   %5
  g4 a8 a,8 d4 b4 
  c8 d8 e4 fis8 d8 e8 fis8 
  g4 d4 e4 e4 
  fis4 g8 fis8 e4 a,4 
  b8 c8 d4 g,  %10
  
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