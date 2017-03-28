\version "2.10.33"

\header {
  title = "70 - Gott sei gelobet und gebenedeiet"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key c \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 g4 
    g4 g4 a2 
    g2 c4 d4 
    c4 b4 a2 
    g2 a8 b8 c4 
    g8 f8 e4 f4 e4     %5
    d2 c4 } g'4 
  g4 g4 a4. g8 
  fis8 g8 a4 g2 
  g4 g4 a4 b4 
  c4 b4 a2   %10
  g2 b4 c4 
  d4 d4 d4 a4 
  g4 e4 d2 
  e4 fis4 g8 f8 e4 
  fis2 g2   %15
  d8 e8 f4 e4 d4 
  e4 d4 c2 
  r4 g'4 g4 g4 
  a4. g8 fis8 g8 a4 
  g2.   %20
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    g8 f8 e2 d4 
    e2 g4 g4~ 
    g8 fis8 g4 g8 fis16 e16 fis4 
    d2 f4. e8 
    d4 c4. b8 c4~     %5
    c8 b16 a16 b4 g4 } e'4 
  d8 c8 d8 f8 e8 d16 c16 b8 cis8 
  d8 g4 fis8 d2 
  d4 e4 fis4 g4~ 
  g8 fis8 g4 g8 fis16 e16 fis4   %10
  d2 g4 g4 
  g4 g4. fis16 e16 fis8 f8 
  e8 d8 d8 cis8 a2 
  c4 c4 d4 c4~ 
  c8 e8 d8 c8 b2   %15
  b4 a4 g8 c8 b4 
  c4. b8 g2 
  r4 d'4 c4 g4 
  c8 d8 e4 d2 
  d2.   %20
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 b8  c8 
    d4 c8 b8 c4 b8 a8 
    b2 e4 f4 
    e4 d4 d4. c8 
    b2 d4 c4 
    d4 g,4 f4 g4     %5
    a4 g8 f8 e4 } c'4 
  g8 a8 b4 c8 fis,8 g8 a8 
  b4 a16 b16 c8 b2 
  b4. cis8 d4 d4 
  c4 d4 e8 c8 a8 d16 c16   %10
  b2 d4 e4 
  d8 c8 b4 c4 d8 c8 
  bes4 a8. g16 f2 
  g4 a4 g4 g4 
  a8 c8 b8 a8 g2   %15
  g4 c,8 d8 e8 g8 g4 
  g4 g8. f16 e2 
  r4 g8 f8 e4 d8 e8 
  f4 c'8 b8 a8 g4 fis8 
  b2.   %20
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 g8  a8 
    b4 e8 d8 c4 f4 
    e2 e8 d8 c8 b8 
    c4 g4 d'4 d,4 
    g2 d'4 a4 
    b4 c8 b8 a4 g4     %5
    f4 g4 c,4 } c'4 
  b8 a8 g4 c8 d8 e4 
  d4 d,4 g2 
  g'8 fis8 e4 d8 c8 b4 
  a4 b8 g8 c8 a8 d4   %10
  g,2 g'8 f8 e4 
  b4. e8 a,4 d4 
  g,4 a4 d2 
  c8 b8 a4 b4 c8 b8 
  a4 d4 g,2   %15
  g4 a8 b8 c8 e8 g8 f8 
  e8 c8 g'8 g,8 c2 
  r4 b4 c4 b4 
  a8 b8 c4 d2 
  g,2.   %20
  
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