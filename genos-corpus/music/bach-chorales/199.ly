\version "2.10.33"

\header {
  title = "199 - Hilf, Gott, laß mir's gelingen"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key d \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 g4 
    a2 g4 
    f2 d4 
    f2. 
    g2 a4 
    bes2 d4     %5
    c2 bes4 
    a2. 
    g2 } c4 
  c2 c4 
  d2 c4   %10
  bes4 a4 g4 
  f2 f4 
  bes2 a4 
  bes2 c4 
  d2 c4   %15
  bes4. a8 g4 
  fis2 a4 
  bes2 d4 
  c2 bes4 
  a2.   %20
  g2
 
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 d8  e8 
    f2~ f8 e8 
    d4 a4 bes4 
    c8 ees8 d8 c8 b4 
    e2 e8 fis8 
    g8 fis8 g2~     %5
    g4. fis8 g4 
    g2 fis4 
    d2 } g4~ 
  g8 f4 e8 f4~ 
  f8 e8 f8 g8 a8 g8   %10
  f2~ f8 e8 
  c2 d4 
  g2 a4~ 
  a8 d,8 g4 a4 
  bes8 a8 g4. fis8   %15
  g4 d4. c8 
  d2 fis4 
  g2 f4 
  e8 d8 e8 fis8 g4 
  g2 fis4   %20
  d2
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 bes4 
    c2~ c8. bes16 
    a4 f4. g8 
    a2 g8 f8 
    c'2 c4 
    d4. c8 bes8 a8     %5
    g8 fis8 g a d4 
    ees4 d8 a8 d8 c8 
    bes2 } c8 bes8 
  a4 bes4 c4~ 
  c4 bes4 c4   %10
  d2 g,8 c8 
  a2 bes8 c8 
  d8 bes8 ees4. d8 
  d2 ees4 
  f4 ees8 d8 ees8 c8   %15
  d8 ees8 f8 f,8 g4 
  a2 d4~ 
  d8 bes8 ees4 d8 bes8 
  g4 c8 a8 bes8 d8 
  ees8 d8 c8 bes8 c8 a8   %20
  b2
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 g4 
    f2 c4 
    d2 bes4 
    a2 d4 
    c2 a4 
    g2 bes4     %5
    ees2 d4 
    c4 d4 d,4 
    g2 } e'4 
  f4 g4 a4 
  bes2 f8 e8   %10
  d8 c8 bes4 c4 
  f,2 bes8 a8 
  g4 g'4 fis4 
  g4 f8 ees8 d8 c8 
  bes4 bes'4 a4   %15
  g4 f4 ees4 
  d2 d4 
  g4. ees8 bes'4~ 
  bes4 a4 g4 
  c,4 a4 d4   %20
  g,2
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