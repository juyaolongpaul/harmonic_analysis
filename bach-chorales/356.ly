\version "2.10.33"

\header {
  title = "356 - Jesu, meine Freude"
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
    a'4 b4 c4 a4 
    d2 cis2 
    d8 e8 f4 e4 e4     %5
    d1 }
  a4 a4 bes4 a4 
  g4 g4 f2 
  a4 b4 c8 b8 a4 
  d4 c8 b8 b2   %10
  a2 a4 a4 
  g4 f4 e2 
  d1 
  
}

alto = \relative c' {
  \repeat volta 2 {
    d4 d4 e4 d4 
    d4 cis4 a2 
    d4 g,4 g'8 f8 e4 
    a4 b4 a2 
    g4 a8 bes8 g8 bes8 a8 g8     %5
    f1 }
  f4 f8 ees8 d8 e8 f4 
  f4 e4 c2 
  c4 f4 e8 d8 c4 
  f4 e8 f8 e4. d8   %10
  cis2 d4~ d8 cis8 
  d8 e8~ e8 d8~ d8 cis16 b16 cis4 
  a1 
  
}

tenor = \relative c {
  \repeat volta 2 {
    f8 g8 a4 bes8 a8 a4 
    bes4 a8 g8 f2 
    f'8 e8 d4 c4 cis4 
    d8 e8 f4 e2 
    d4 d4 d4 cis4     %5
    a1 }
  d4 c4 bes8 c8 d4 
  d4 c8 bes8 a2 
  a4 g8 f8 g4 a4 
  a8 gis8 a2 gis4   %10
  a2 f8 g8 a4 
  d,8 a'8 a4 b8 e,8 a8 g8 
  fis1 
  
}

baixo = \relative c {
  \repeat volta 2 {
    d8 e8 f4 cis4 d4 
    g,4 a4 d,2 
    d'8 e8 f4 e8 f8 g4 
    fis8 gis8 gis4 a2 
    bes4 a8 g8 a4 a,4     %5
    d1 }
  d8 e8 f4 g4 d8 c8 
  bes8 g8 c4 f,2 
  f'4 e8 d8 e4 f4 
  b,4 c8 d8 e2   %10
  a,2 d8 e8 f4 
  b,8 cis8 d4 gis,4 a4 
  d,1 
  
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