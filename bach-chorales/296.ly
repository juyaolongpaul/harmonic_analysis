\version "2.10.33"

\header {
  title = "296 - Nun lob, mein Seel, den Herren"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key c \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 c4 
    c2 b4 
    a2 g4 
    c4 d2 
    e2 e4 
    e2 e4     %5
    e2 d4 
    c8 d8 d4. c8 
    c2 } c4 
  c4 d4 e4 
  d2 e4   %10
  c4 b8 a8 b4 
  a2 a4 
  d4 c4 b4 
  c4 a2 
  g2 g4   %15
  c2 c4 
  d4 e4 f4 
  e4 d4 e4 
  c2 c4 
  f2 f4   %20
  e4 d4 e4 
  d2 d4 
  e2 e4 
  f2 f4 
  g4. f8 e8 d8   %25
  c2 e4 
  d4 c4 b4 
  c4 a2 
  g2 g4 
  c2 b4   %30
  a2 g4 
  d'4 e4. d8 
  d2 e4 
  f4 e4 d4 
  c8 d8 d4. c8   %35
  c2. 
  
}

alto = \relative c'' {
  \repeat volta 2 {
    \partial 4 g4 
    a2 g4 
    c,4 d4 e8 f8 
    g4 a4 g4 
    g2 g4 
    b4 a4 gis4     %5
    a4 e8 f8 g4 
    g4 a4 g4 
    e2 } e4 
  f2 e4 
  gis4 a4 b4   %10
  a2 gis4 
  a2 f8 e8 
  d2 d4 
  e4 d4 c4 
  b2 e4   %15
  d2 a'4 
  g2 g4 
  g4 f4 e4 
  e2 a4 
  a2 a4   %20
  bes4 f4 g4 
  f2 g4 
  g2 a4 
  f4 a2 
  g2.   %25
  a2 g4 
  a2 g4 
  g2 fis4 
  d2 d4 
  e4 f4 g4   %30
  c,4 d4 e4 
  f4 e8 d8 e8 fis8 
  g2 g4 
  f4 g2 
  g4 a4 g4   %35
  e2. 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    e4 f4 g4 
    f2 e4 
    c2 b4 
    c2 c4 
    b4 c4 d4     %5
    c2 b4 
    c2 b4 
    g2 } g4 
  a4 b4 c4 
  d8 e8 f4 e4   %10
  e2 d4 
  c2 c4 
  a2 g4 
  g4 fis2 
  g2 b4   %15
  c4 d4 e4 
  d4 c4 b4 
  c2 b4 
  c2 f4 
  a,2 d4   %20
  d2 cis4 
  a2 b4 
  c2 c4 
  a2 d4 
  d4 b4 c4   %25
  c2 c8 b8 
  a2 d4 
  c4 e4 d8 c8 
  b2 b4 
  g4 d'4 e4   %30
  a,4 b4 c4 
  b4 c4 a4 
  b2 c4 
  c2 b4 
  c2 b4   %35
  g2. 
  
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 c4 
    a2 e4 
    f2 c4 
    e4 f4 g4 
    c,2 c'4 
    gis4 fis4 e4     %5
    a4 g4 f4 
    e4 f4 g4 
    c,2 } c4 
  f2 c'4 
  b4 a4 gis4   %10
  a4 e2 
  f2 f4 
  fis2 g4 
  c,4 d2 
  g,2 e'4   %15
  a4 b4 c4 
  b4 a4 g4 
  c2 gis4 
  a2 f4 
  d4 e4 f4   %20
  g4 a2 
  d,2 g4 
  c,4 b4 a4 
  d4 c4 b4 
  e4 d4 c4   %25
  f2 c'4 
  fis,2 g4 
  e4 c4 d4 
  g,2 g'8 f8 
  e4 d4 c4   %30
  f2 e4 
  d4 c2 
  g'2 c4 
  a4 g4 f4 
  e4 f4 g4   %35
  c,2. 
  
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