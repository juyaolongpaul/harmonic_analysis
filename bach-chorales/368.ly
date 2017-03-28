\version "2.10.33"

\header {
  title = "368 - Hilf, Herr Jesu, laß gelingen"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key f \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    c4 a2 
    g2 f4 
    bes4 c4. d16 ees16 
    d2. 
    c2.     %5
    d4 bes2 
    c2 f,4 
    bes4 a2 
    g2. }
  e'4 c2   %10
  a2 d4 
  cis4 d4. e16 f16 
  e2. 
  a,2. 
  d4 e4 f4   %15
  g4 c,4 bes4 
  a4 d4 bes4 
  g2. 
  f2. 
  
}

alto = \relative c'' {
  \repeat volta 2 {
    a4 f2 
    f4 e4 f4 
    d4 g4 f4 
    f2. 
    f2.     %5
    a4 g2 
    g4 f4 f4 
    f8 g8 g4 f4 
    e2. }
  g8 f8 g4 g4   %10
  g4 f4 f4 
  g4 a4 bes8 a8 
  g8 f8 g4. e8 
  f2. 
  f4 g4 a4   %15
  g8 f8 g4 e4 
  c4 bes4 f'4 
  f4 e8 d8 e4 
  c2. 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    f4 c2 
    c4. bes8 a4 
    bes2 a8 bes16 c16 
    c4 bes8 a8 bes4 
    a2.     %5
    f'4 d2 
    c8 bes8 a4 a4 
    d8 c8 c2 
    c2. }
  c2 e4   %10
  c2 bes4~ 
  bes4 a4 d8 c8 
  bes4 a8 g8 a4 
  a2. 
  bes4. g8 c4   %15
  c2 g4 
  f2 d'4 
  g,4 c4 bes4 
  a2. 
  
}

baixo = \relative c {
  \repeat volta 2 {
    f8 g8 a4 f4 
    c2 d4 
    g8 f8 ees4 f4 
    bes,8 c8 d4 bes4 
    f'2.     %5
    d4 g4 f4 
    e4 f8 e8 d8 c8 
    d8 e8 f4 f4 
    c2. }
  c8 d8 e4 c4   %10
  f8 e8 f4 bes4 
  e,4 f4 d4 
  d4 cis8 b8 cis4 
  d2. 
  bes'8 a8 g4 f4   %15
  e8 d8 e4 c4 
  f4 bes,2 
  c8 bes8 c2 
  f,2. 
  
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