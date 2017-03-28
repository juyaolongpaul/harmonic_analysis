\version "2.10.33"

\header {
  title = "261 - Christ lag in Todesbanden"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key e \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 b4
    ais4 b8 cis8 d4 e4
    d4 cis4 b4 b4
    g4 a4 b4 a8[ g]
    \partial 2.
    fis2 e4 }
  e4 g4 a4 e4   %5
  g4 a4 b4 b4 
  e4 dis4 e4 fis4 
  d4 cis4 b4 b8 cis8 
  d4 b4 d4 a4 
  g4 fis4 e2   %10
  b'4 a4 g4 fis4 
  e1
  
}

alto = \relative c'' {
  
  \repeat volta 2 {
    \partial 4 g4
    fis4 fis8 ais8 b4 cis4
    b4 ais4 fis4 fis4
    e4 d4 d4 e4
    \partial 2.
    e4 dis4 b4 }
  b4 e8 d8 e4 c4   %5
  b4 a4 g4 d'4 
  g4 fis4 e4 b'4 
  b4 ais4 fis4 g4 
  a4 d,4 d4 c4 
  b8 e4 d8 d4 c4   %10
  b8 cis8 dis4 e4 dis 
  b1
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 e8 d8
    cis4 d8 e8 fis4 fis4
    fis4 fis8 e8 dis4 b4
    b4 a4 g8 a8 b4
    \partial 2.
    c4 b8 a8 g4 } 
  g4 g4 c4 a4   %5
  g4 d'4 d4 b4 
  b4 a4 b4 b4 
  fis'4 fis8 e8 d4 b4 
  a4 g4 g4 fis4 
  g4 a4 a2   %10
  fis2 g8 e8 b'8 a8 
  gis1
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 e4
    fis8 e8 d8 cis8 b4 ais4
    b4 fis'4 b,4 dis4
    e4 fis4 g4 e4
    \partial 2.
    a,4 b4 e,4 } 
  e'8 d8 c8 b8 a8 b8 c8 d8   %5
  e4 fis4 g4 g4 
  e4 fis4 g4 d8 e8 
  fis4 fis,4 b4 e4 
  fis4 g4 b,8 c8 d4 
  e4 fis8. gis16 a2   %10
  dis,4 b4 e8 a,8 b4 
  e1
  
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