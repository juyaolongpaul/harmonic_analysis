\version "2.10.33"

\header {
  title = "96 - Jesu, meine Freude"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    a4 a4 g4 f4 
    e2 d2 
    a'4 a4 b4 cis4 
    d2 cis2 
    d8 e8 f4 e4 e4     %5
    d1 }
  a4 a4 bes4 a4 
  g8 a16 bes16 g4 f2 
  a4 b4 c4 a4 
  d4 c8 b8 b2   %10
  a2 a4 d4 
  g,4 f8 e8 e2 
  d1 
  
}

alto = \relative c' {
  \repeat volta 2 {
    f4 f4 e4 d4 
    d4 cis4 a2 
    f'4 f4 f8 d8 g4 
    a4 g4 a2 
    a4 a4 bes4 a8 g8     %5
    f1 }
  f8 g8 a4 a8 g8 f4 
  f4 e4 c2 
  f4 f4 g4 e4 
  a8 e8 e8 f8 e8 f8 e8 d8   %10
  cis2 d4 d4 
  d8 cis8 d4 d8 cis16 b16 cis4 
  a1 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    d4 d4 d8 cis8 d4 
    a8 bes8 a8 g8 f2 
    d'4 d4 d8 b8 e4 
    d8 e16 f16 e8 d8 e2 
    a,4 d4 d4 cis4     %5
    a1 }
  d4 d4 d4 c4 
  c8 a8 bes8 g8 a2 
  c4 d4 e4 cis4 
  a8 b8 c8 a8 a4 gis4   %10
  a2 f8 g8 a8 f8 
  g4 a8 bes8 a8 e8 a8 g8 
  fis1 
  
}

baixo = \relative c {
  \repeat volta 2 {
    d8 e8 f8 d8 bes'4 a8 g8 
    a4 a,4 d2 
    d8 e8 f8 d8 g4 f8 e8 
    f8 g16 a16 bes4 a2 
    f8 e8 d8 f8 g8 e8 a4     %5
    d,1 }
  d8 e8 f8 d8 g4 a8 bes8 
  c4 c,4 f2 
  f8 e8 d8 f8 e8 f8 g8 e8 
  fis8 gis8 a8 d,8 e2   %10
  a,2 d8 e8 f8 d8 
  bes'4 a8 g8 a4 a,4 
  d1 
  
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